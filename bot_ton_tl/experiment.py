"""Stage 3: baselines vs. transfer learning on ToN-IoT with scarce labels.

    python -m bot_ton_tl.experiment --prepared prepared --out results/run1

For every seed the BoT-IoT model is pre-trained once (and a DANN model once); then for
every label budget k (k labelled ToN-IoT flows per class, drawn from the ToN training
pool) and task (binary / multiclass) these methods are trained and scored on the fixed,
group-split ToN-IoT test set:

    source_only   BoT-IoT model applied unchanged (binary: P(attack) = 1 - P(normal))
    dann_zero     DANN model applied unchanged (alignment used unlabeled ToN flows only)
    target_only   same network trained from scratch on the k-shot budget    (no transfer)
    rf            random forest on the k-shot budget                         (no transfer)
    frozen        pre-trained encoder frozen + new head (the repo's CNN-TL recipe)
    finetune      pre-trained encoder, LP-FT (head first, then everything at a 10x smaller
                  encoder learning rate - conservative, keeps the encoder close to BoT-IoT)
    dann          DANN-pre-trained encoder, LP-FT
    finetune_full pre-trained encoder trained exactly like target_only (same learning rate,
                  steps and early stopping): the ONLY difference from target_only is the
                  initialisation, so finetune_full - target_only is the clean transfer effect
    dann_full     same, starting from the DANN-pre-trained encoder

Model selection inside a budget uses 20% of that budget only; the large ToN validation
split is never used for few-shot runs (it would leak labels beyond the budget).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier

from .data import DomainData, holdout_split, load_domain, sample_budget
from .features import ALL_GROUPS, DEFAULT_GROUPS, Standardizer, build_features, continuous_mask
from .metrics import binary_metrics, multiclass_metrics, per_class_rates
from .models import count_params
from .schema import BOT, SHARED_CLASS_MAP, TON
from .train import TrainConfig, finetune, predict_proba, scaled, train_dann, train_supervised

ALL_METHODS = ("source_only", "dann_zero", "target_only", "rf", "frozen", "finetune", "dann",
               "finetune_full", "dann_full")
NEEDS_SOURCE = {"source_only", "frozen", "finetune", "finetune_full"}
NEEDS_DANN = {"dann_zero", "dann", "dann_full"}


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepared", type=Path, default=Path("prepared"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--tasks", nargs="+", default=["binary", "multiclass"], choices=["binary", "multiclass"])
    ap.add_argument("--methods", nargs="+", default=list(ALL_METHODS), choices=list(ALL_METHODS))
    ap.add_argument("--shots", nargs="+", type=int, default=[10, 50, 100, 500, 1000],
                    help="labelled ToN-IoT flows per class")
    ap.add_argument("--include-full", action="store_true", help="also train on the whole (capped) ToN pool (upper bound)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--arch", choices=["mlp", "cnn1d"], default="mlp")
    ap.add_argument("--features", nargs="+", default=list(DEFAULT_GROUPS), choices=list(ALL_GROUPS))
    ap.add_argument("--norm", choices=["shared", "source", "per_domain"], default="shared",
                    help="shared: one scaler fitted on BoT train + the unlabeled ToN pool (features keep the "
                         "same physical units in both datasets); per_domain: separate scalers (AdaBN-style "
                         "ablation; also absorbs label-shift, so use with care)")
    ap.add_argument("--merge", nargs="*", default=[], metavar="A+B",
                    help="merge classes in both datasets, e.g. dos+ddos (single flows cannot tell DoS from "
                         "DDoS: in BoT-IoT almost all label-conflicting vectors are dos|ddos)")
    ap.add_argument("--unseen-classes", nargs="*", default=[],
                    help="ToN classes never labelled (zero-day study); binary detection reported per class")
    ap.add_argument("--source-cap", type=int, default=50_000, help="BoT-IoT training rows per class")
    ap.add_argument("--target-pool-cap", type=int, default=20_000, help="ToN training-pool rows per class")
    ap.add_argument("--test-cap", type=int, default=50_000, help="test rows per class (both datasets)")
    ap.add_argument("--val-cap", type=int, default=5_000, help="BoT-IoT validation rows per class")
    ap.add_argument("--test-files", nargs="*", type=int, default=[],
                    help="ToN file numbers used as test set (scenario split) instead of the hash split")
    ap.add_argument("--split-salt", type=int, default=2026)
    ap.add_argument("--steps-pretrain", type=int, default=4000)
    ap.add_argument("--steps-dann", type=int, default=4000)
    ap.add_argument("--steps-target", type=int, default=1500)
    ap.add_argument("--steps-lp", type=int, default=500)
    ap.add_argument("--steps-ft", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--rf-trees", type=int, default=200)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--quick", action="store_true", help="tiny settings for a smoke test")
    args = ap.parse_args(argv)
    if args.quick:
        args.shots, args.seeds = args.shots[:2], args.seeds[:1]
        args.steps_pretrain = args.steps_dann = 200
        args.steps_target, args.steps_lp, args.steps_ft = 150, 60, 100
        args.rf_trees = 25
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.out is None:
        args.out = Path("results") / time.strftime("run_%Y%m%d_%H%M%S")
    return args


class Encoded:
    """Features + labels of one split, ready for training."""

    def __init__(self, frame: pd.DataFrame, domain: str, groups):
        X, self.names = build_features(frame, domain, groups)
        self.raw = X
        self.X = X
        self.cls = frame["cls"].to_numpy(dtype=object)
        self.binary = frame["binary"].to_numpy(dtype=np.int64)

    def rescale(self, scaler: Standardizer) -> None:
        self.X = scaler.transform(self.raw)


def steps_for(base: int, n: int, batch: int) -> int:
    """Few-shot budgets use the base step count; big training sets get ~3 epochs (max 8000)."""
    return int(max(base, min(8000, 3 * n // max(batch, 1))))


def run(args: argparse.Namespace) -> pd.DataFrame:
    t_start = time.time()
    out: Path = args.out
    (out / "models").mkdir(parents=True, exist_ok=True)
    caps_src = {"train": args.source_cap, "val": args.val_cap, "test": args.test_cap}
    # The ToN validation split is deliberately not loaded: few-shot model selection may only
    # use the labelled budget itself.
    caps_tgt = {"train": args.target_pool_cap, "val": 0, "test": args.test_cap}
    print(f"loading {args.prepared} ...", flush=True)
    src: DomainData = load_domain(args.prepared, BOT, caps=caps_src, salt=args.split_salt)
    tgt: DomainData = load_domain(args.prepared, TON, caps=caps_tgt, salt=args.split_salt,
                                  test_files=tuple(args.test_files))
    groups = tuple(args.features)
    merge = {c: g for g in args.merge for c in g.lower().split("+")}
    for frame in (src.train, src.val, src.test, tgt.train, tgt.val, tgt.test):
        frame["cls"] = frame["cls"].map(lambda c: merge.get(c, c))

    s_tr, s_va, s_te = (Encoded(f, BOT, groups) for f in (src.train, src.val, src.test))
    t_tr, t_te = (Encoded(f, TON, groups) for f in (tgt.train, tgt.test))
    mask = continuous_mask(s_tr.names)
    if args.norm == "shared":
        src_scaler = tgt_scaler = Standardizer(mask).fit(np.concatenate([s_tr.raw, t_tr.raw]))
    else:
        src_scaler = Standardizer(mask).fit(s_tr.raw)
        tgt_scaler = Standardizer(mask).fit(t_tr.raw) if args.norm == "per_domain" else src_scaler
    for e in (s_tr, s_va, s_te):
        e.rescale(src_scaler)
    for e in (t_tr, t_te):
        e.rescale(tgt_scaler)
    (out / "models" / "scalers.json").write_text(json.dumps(
        {"features": s_tr.names, "source": src_scaler.state(), "target": tgt_scaler.state()}), encoding="utf-8")

    src_classes = sorted(set(s_tr.cls))
    src_index = {c: i for i, c in enumerate(src_classes)}
    ys_tr = np.array([src_index[c] for c in s_tr.cls])
    va_keep = np.isin(s_va.cls, src_classes)
    ys_va = np.array([src_index[c] for c in s_va.cls[va_keep]])
    unseen = tuple(c.lower() for c in args.unseen_classes)
    tgt_classes = sorted(c for c in set(t_tr.cls) | set(t_te.cls) if c not in unseen)
    tgt_index = {c: i for i, c in enumerate(tgt_classes)}
    print(f"source classes {src_classes} | target classes {tgt_classes} | unseen {list(unseen)}", flush=True)
    print(f"rows: source train {len(s_tr.cls):,} val {len(s_va.cls):,} test {len(s_te.cls):,} | "
          f"target pool {len(t_tr.cls):,} test {len(t_te.cls):,} | features {len(s_tr.names)}", flush=True)

    rows: list[dict] = []
    per_class: list[dict] = []

    def record(task, method, shots, seed, metrics, extra=None, cls_true=None, correct=None):
        row = {"task": task, "method": method, "shots": shots, "seed": seed, **metrics, **(extra or {})}
        rows.append(row)
        if cls_true is not None:
            for c, rate in per_class_rates(cls_true, correct).items():
                per_class.append({"task": task, "method": method, "shots": shots, "seed": seed, "cls": c,
                                  "rate": rate, "n": int((cls_true == c).sum())})

    def evaluate(task, method, shots, seed, proba=None, pred_labels=None, extra=None):
        """proba: (n_test, n_out) for the task's output space; binary uses column 1 = attack."""
        if task == "binary":
            p_attack = proba[:, 1]
            m = binary_metrics(t_te.binary, p_attack)
            correct = (p_attack >= 0.5).astype(int) == t_te.binary
            record(task, method, shots, seed, m, extra, t_te.cls, correct)
        else:
            keep = np.isin(t_te.cls, tgt_classes)
            y_true = np.array([tgt_index[c] for c in t_te.cls[keep]])
            y_pred = proba[keep].argmax(1) if pred_labels is None else pred_labels[keep]
            m = multiclass_metrics(y_true, y_pred)
            record(task, method, shots, seed, m, extra, t_te.cls[keep], y_true == y_pred)
        return rows[-1]

    def zero_shot(model, name, seed):
        proba_src = predict_proba(model, t_te.X, args.device)
        normal_col = src_index.get("normal")
        p_normal = proba_src[:, normal_col] if normal_col is not None else np.zeros(len(t_te.X))
        if "binary" in args.tasks:
            r = evaluate("binary", name, 0, seed, np.column_stack([p_normal, 1 - p_normal]))
            print(f"  [{name}] binary zero-shot macro-F1={r['macro_f1']:.3f} AUC={r['roc_auc']:.3f}", flush=True)
        # Multiclass only on the classes both taxonomies share (reconnaissance ~ scanning).
        shared_t = {merge.get(c, c): merge.get(v, v) for c, v in SHARED_CLASS_MAP[TON].items()}
        shared_s = {merge.get(c, c): merge.get(v, v) for c, v in SHARED_CLASS_MAP[BOT].items()}
        keep = np.isin(t_te.cls, list(shared_t))
        names = sorted(set(shared_t.values()))
        src_to_shared = {src_index[c]: names.index(s) for c, s in shared_s.items() if c in src_index}
        cols = list(src_to_shared)
        if keep.any() and cols:
            pred = np.array([src_to_shared[cols[j]] for j in proba_src[keep][:, cols].argmax(1)])
            y_true = np.array([names.index(shared_t[c]) for c in t_te.cls[keep]])
            m = multiclass_metrics(y_true, pred)
            record("multiclass_shared", name, 0, seed, m, {"classes": "|".join(names)},
                   t_te.cls[keep], y_true == pred)

    base = TrainConfig(batch_size=args.batch_size, device=args.device)
    for seed in args.seeds:
        print(f"\n=== seed {seed} ===", flush=True)
        src_model = dann_model = None
        if NEEDS_SOURCE & set(args.methods):
            t0 = time.time()
            cfg = scaled(base, steps=args.steps_pretrain, eval_every=max(args.steps_pretrain // 20, 1), patience=5)
            src_model, info = train_supervised(args.arch, s_tr.X, ys_tr, len(src_classes), cfg, seed,
                                               s_va.X[va_keep], ys_va)
            te_keep = np.isin(s_te.cls, src_classes)
            pred = predict_proba(src_model, s_te.X[te_keep], args.device).argmax(1)
            m = multiclass_metrics(np.array([src_index[c] for c in s_te.cls[te_keep]]), pred)
            record("source_in_domain", "source_pretrain", 0, seed, m,
                   {"params": count_params(src_model), "train_s": round(time.time() - t0, 1), **info})
            print(f"  source pre-training: in-domain macro-F1={m['macro_f1']:.3f} ({time.time() - t0:.0f}s)", flush=True)
            torch.save(src_model.encoder.state_dict(), out / "models" / f"source_encoder_seed{seed}.pt")
            if "source_only" in args.methods:
                zero_shot(src_model, "source_only", seed)
        if NEEDS_DANN & set(args.methods):
            t0 = time.time()
            cfg = scaled(base, steps=args.steps_dann, eval_every=max(args.steps_dann // 20, 1))
            dann_model, info = train_dann(args.arch, s_tr.X, ys_tr, len(src_classes), t_tr.X, cfg, seed)
            print(f"  DANN pre-training: final domain acc={info['final_domain_acc']:.3f} ({time.time() - t0:.0f}s)",
                  flush=True)
            torch.save(dann_model.encoder.state_dict(), out / "models" / f"dann_encoder_seed{seed}.pt")
            if "dann_zero" in args.methods:
                zero_shot(dann_model, "dann_zero", seed)

        budgets: list[int | None] = list(args.shots) + ([None] if args.include_full else [])
        for shots in budgets:
            label = "full" if shots is None else shots
            rng = np.random.default_rng(seed * 100_003 + (shots or 0))
            idx = sample_budget(t_tr.cls, shots, rng, exclude=unseen)
            fit_i, hold_i = holdout_split(t_tr.cls[idx], 0.2, rng)
            fit_i, hold_i = idx[fit_i], idx[hold_i]
            n_fit = len(fit_i)
            print(f"  budget {label}/class -> {len(idx):,} labelled flows ({n_fit} fit / {len(hold_i)} hold)", flush=True)
            for task in args.tasks:
                if task == "binary":
                    y_all, n_out = t_tr.binary, 2
                else:
                    y_all = np.array([tgt_index.get(c, -1) for c in t_tr.cls])
                    n_out = len(tgt_classes)
                Xf, yf, Xh, yh = t_tr.X[fit_i], y_all[fit_i], t_tr.X[hold_i], y_all[hold_i]
                cfg_t = scaled(base, steps=steps_for(args.steps_target, n_fit, args.batch_size), eval_every=50)
                cfg_lp = scaled(base, steps=steps_for(args.steps_lp, n_fit, args.batch_size), eval_every=50)
                cfg_ft = scaled(base, steps=steps_for(args.steps_ft, n_fit, args.batch_size), eval_every=50, lr=3e-4)
                common = {"n_labelled": int(len(idx))}
                for method in args.methods:
                    if method in ("source_only", "dann_zero"):
                        continue
                    t0 = time.time()
                    if method == "rf":
                        rf = RandomForestClassifier(n_estimators=args.rf_trees, class_weight="balanced_subsample",
                                                    n_jobs=-1, random_state=seed)
                        rf.fit(np.concatenate([Xf, Xh]), np.concatenate([yf, yh]))
                        proba = np.zeros((len(t_te.X), n_out))
                        proba[:, rf.classes_] = rf.predict_proba(t_te.X)
                        info = {}
                    elif method in ("target_only", "finetune_full", "dann_full"):
                        donor = {"finetune_full": src_model, "dann_full": dann_model}.get(method)
                        model, info = train_supervised(args.arch, Xf, yf, n_out, cfg_t, seed, Xh, yh,
                                                       encoder_state=None if donor is None else donor.encoder.state_dict())
                        proba = predict_proba(model, t_te.X, args.device)
                    elif method in ("frozen", "finetune", "dann"):
                        donor = dann_model if method == "dann" else src_model
                        head = "mlp" if method == "frozen" else "linear"
                        model, info = finetune(donor.encoder.state_dict(), args.arch, Xf, yf, n_out, seed,
                                               cfg_lp if method != "frozen" else cfg_t,
                                               None if method == "frozen" else cfg_ft, Xh, yh, head=head)
                        proba = predict_proba(model, t_te.X, args.device)
                    r = evaluate(task, method, label, seed, proba,
                                 extra={**common, **info, "train_s": round(time.time() - t0, 1)})
                    print(f"    {task:10s} {method:12s} macro-F1={r['macro_f1']:.3f}", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(out / "results.csv", index=False)
    pd.DataFrame(per_class).to_csv(out / "per_class.csv", index=False)
    summary = summarize(results)
    summary.to_csv(out / "summary.csv", index=False)
    text = render_summary(summary, args, src, tgt, time.time() - t_start) + render_gains(results)
    (out / "summary.txt").write_text(text, encoding="utf-8")
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update(source_classes=src_classes, target_classes=tgt_classes, features=s_tr.names,
                  source_split_counts=src.split_counts, target_split_counts=tgt.split_counts)
    (out / "config.json").write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")
    plot_curves(summary, out)
    print("\n" + text)
    print(f"results written to {out.resolve()}")
    return results


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [m for m in ("macro_f1", "balanced_accuracy", "accuracy", "f1", "recall_dr", "far_fpr", "roc_auc",
                           "weighted_f1") if m in results]
    g = results.groupby(["task", "method", "shots"], sort=False)[metrics]
    summary = g.agg(["mean", "std"])
    summary.columns = [f"{m}_{s}" for m, s in summary.columns]
    summary["n_seeds"] = g.size()
    return summary.reset_index()


def render_summary(summary: pd.DataFrame, args, src: DomainData, tgt: DomainData, seconds: float) -> str:
    lines = ["BoT-IoT -> ToN-IoT transfer learning | summary (mean +- std over seeds)", "=" * 78,
             f"arch={args.arch} features={args.features} norm={args.norm} seeds={args.seeds} "
             f"unseen={args.unseen_classes or 'none'} merge={args.merge or 'none'} "
             f"test_files={args.test_files or 'hash split'}",
             f"ToN test rows by class: {tgt.test['cls'].value_counts().to_dict()}", ""]
    order = lambda c: (int(c) if str(c).isdigit() else float("inf"))  # noqa: E731  0,10,...,full
    for task in summary["task"].unique():
        sub = summary[summary["task"] == task].assign(shots=lambda d: d["shots"].astype(str))
        metrics = ["macro_f1", "roc_auc"] if task == "binary" else ["macro_f1"]
        for metric in metrics:
            lines.append(f"[{task}]  {metric}   (columns = labelled ToN flows per class; 0 = no ToN labels)")
            table = sub.pivot_table(index="method", columns="shots", values=f"{metric}_mean", aggfunc="first")
            std = sub.pivot_table(index="method", columns="shots", values=f"{metric}_std", aggfunc="first")
            columns = sorted(table.columns, key=order)
            table, std = table[columns], std.reindex(index=table.index, columns=columns)
            cells = table.copy().astype(object)
            for c in table.columns:
                for i in table.index:
                    v, s = table.loc[i, c], std.loc[i, c]
                    cells.loc[i, c] = "" if pd.isna(v) else (f"{v:.3f}" if pd.isna(s) else f"{v:.3f}+-{s:.3f}")
            lines.append(cells.to_string())
            lines.append("")
    lines.append(f"total time {seconds / 60:.1f} min")
    return "\n".join(lines) + "\n"


def render_gains(results: pd.DataFrame) -> str:
    """Paired comparison per seed: transfer method minus each no-transfer baseline."""
    order = lambda c: (int(c) if str(c).isdigit() else float("inf"))  # noqa: E731
    transfer = [m for m in ("finetune_full", "dann_full", "finetune", "dann", "frozen") if m in set(results["method"])]
    lines = ["TRANSFER GAIN, paired by seed: mean macro-F1 difference (seeds where transfer wins / seeds)",
             "positive = transfer helps; a gain is convincing only if it wins on every seed", ""]
    for task in ("binary", "multiclass"):
        r = results[(results["task"] == task) & (results["shots"].astype(str) != "0")]
        if r.empty or not transfer:
            continue
        r = r.assign(shots=r["shots"].astype(str))
        piv = r.pivot_table(index=["shots", "seed"], columns="method", values="macro_f1", aggfunc="first")
        for base in ("target_only", "rf"):
            if base not in piv:
                continue
            table = {}
            for m in transfer:
                if m not in piv:
                    continue
                d = (piv[m] - piv[base]).dropna()
                g = d.groupby(level="shots")
                table[m] = {k: f"{v.mean():+.3f} ({int((v > 0).sum())}/{len(v)})" for k, v in g}
            frame = pd.DataFrame(table).T
            frame = frame[sorted(frame.columns, key=order)]
            lines += [f"[{task}] transfer minus {base}", frame.to_string(), ""]
    return "\n".join(lines) + "\n"


def plot_curves(summary: pd.DataFrame, out: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for task in ("binary", "multiclass"):
        sub = summary[(summary["task"] == task)]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        numeric = sub[sub["shots"].apply(lambda s: isinstance(s, (int, np.integer)) and s > 0)]
        for method, g in numeric.groupby("method", sort=False):
            g = g.sort_values("shots")
            ax.errorbar(g["shots"], g["macro_f1_mean"], yerr=g["macro_f1_std"].fillna(0), marker="o",
                        capsize=3, label=method)
        for method, g in sub[sub["shots"] == 0].groupby("method", sort=False):
            ax.axhline(g["macro_f1_mean"].iloc[0], linestyle="--", linewidth=1, color="grey")
            ax.annotate(f"{method} (0 labels)", (1, g["macro_f1_mean"].iloc[0]), xycoords=("axes fraction", "data"),
                        ha="right", va="bottom", fontsize=8, color="grey")
        ax.set_xscale("log")
        ticks = sorted(numeric["shots"].astype(int).unique())
        if ticks:
            ax.set_xticks(ticks)
            ax.set_xticklabels([str(t) for t in ticks])
            ax.minorticks_off()
        ax.set_xlabel("labelled ToN-IoT flows per class")
        ax.set_ylabel("macro-F1 on ToN-IoT test")
        ax.set_title(f"BoT-IoT -> ToN-IoT, {task}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"learning_curve_{task}.png", dpi=150)
        plt.close(fig)


def main(argv=None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
