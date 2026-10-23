"""Stage 2: how far apart are BoT-IoT and ToN-IoT in the shared feature space?

    python -m bot_ton_tl.diagnose --prepared prepared --out results/diagnose

Reports, on the training splits only:
  * per-feature mean/std per dataset and the two-sample KS statistic, overall and for the
    class pairs both taxonomies share (normal, dos, ddos, reconnaissance~scanning)
  * a domain classifier's ROC-AUC and the proxy A-distance (Ben-David et al. 2010):
    AUC ~0.5 -> datasets indistinguishable, ~1.0 -> trivially separable. Trees are
    invariant to any shared monotone rescaling, so this also holds for the scaled inputs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from .data import load_domain
from .features import ALL_GROUPS, DEFAULT_GROUPS, build_features
from .schema import BOT, SHARED_CLASS_MAP, TON


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    a, b = np.sort(a), np.sort(b)
    grid = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, grid, side="right") / len(a)
                               - np.searchsorted(b, grid, side="right") / len(b))))


def domain_auc(Xs: np.ndarray, Xt: np.ndarray, seed: int = 0, max_rows: int = 20_000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = min(len(Xs), len(Xt), max_rows)
    if n < 20:
        return float("nan"), float("nan")
    Xs = Xs[rng.choice(len(Xs), n, replace=False)]
    Xt = Xt[rng.choice(len(Xt), n, replace=False)]
    X = np.concatenate([Xs, Xt])
    y = np.r_[np.zeros(n), np.ones(n)]
    clf = HistGradientBoostingClassifier(max_iter=200, random_state=seed)
    p = cross_val_predict(clf, X, y, cv=StratifiedKFold(3, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    err = float(((p >= 0.5) != y).mean())
    return float(roc_auc_score(y, p)), 2.0 * (1.0 - 2.0 * min(err, 0.5))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prepared", type=Path, default=Path("prepared"))
    ap.add_argument("--out", type=Path, default=Path("results") / "diagnose")
    ap.add_argument("--features", nargs="+", default=list(DEFAULT_GROUPS), choices=list(ALL_GROUPS))
    ap.add_argument("--cap", type=int, default=20_000, help="training rows per class per dataset")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    caps = {"train": args.cap, "val": 0, "test": 0}
    src = load_domain(args.prepared, BOT, caps=caps, seed=args.seed).train
    tgt = load_domain(args.prepared, TON, caps=caps, seed=args.seed).train
    groups = tuple(args.features)
    Xs, names = build_features(src, BOT, groups)
    Xt, _ = build_features(tgt, TON, groups)
    cs, ct = src["cls"].to_numpy(dtype=object), tgt["cls"].to_numpy(dtype=object)

    pairs = [("all", np.ones(len(cs), bool), np.ones(len(ct), bool))]
    inv_t = {v: k for k, v in SHARED_CLASS_MAP[TON].items()}
    for s_cls, shared in SHARED_CLASS_MAP[BOT].items():
        t_cls = inv_t[shared]
        pairs.append((f"{s_cls}~{t_cls}", cs == s_cls, ct == t_cls))

    rows = []
    for j, name in enumerate(names):
        row = {"feature": name, "bot_mean": float(Xs[:, j].mean()), "ton_mean": float(Xt[:, j].mean()),
               "bot_std": float(Xs[:, j].std()), "ton_std": float(Xt[:, j].std())}
        for label, ms, mt in pairs:
            row[f"ks_{label}"] = ks_statistic(Xs[ms, j], Xt[mt, j])
        rows.append(row)
    shift = pd.DataFrame(rows).sort_values("ks_all", ascending=False)
    shift.to_csv(args.out / "feature_shift.csv", index=False)

    lines = ["BoT-IoT vs ToN-IoT | domain-shift diagnostics (training splits)", "=" * 72,
             f"rows: BoT {len(cs):,} {pd.Series(cs).value_counts().to_dict()}",
             f"      ToN {len(ct):,} {pd.Series(ct).value_counts().to_dict()}", "",
             "Domain classifier (3-fold CV, gradient boosting). AUC 0.5 = indistinguishable.",
             f"{'subset':28s} {'n_bot':>8s} {'n_ton':>8s} {'AUC':>7s} {'proxy A-dist':>13s}"]
    for label, ms, mt in pairs:
        auc, pad = domain_auc(Xs[ms], Xt[mt], args.seed)
        lines.append(f"{label:28s} {int(ms.sum()):8d} {int(mt.sum()):8d} {auc:7.3f} {pad:13.3f}")
    lines += ["", "Most shifted features (KS statistic; 0 = same distribution, 1 = disjoint):",
              shift.head(15).to_string(index=False, float_format=lambda v: f"{v:.3f}"), "",
              "Reading this: a near-1.0 AUC even within matched classes means the encoder must",
              "learn an invariance (DANN) or be re-fitted on target labels (fine-tuning); expect",
              "source_only to fail. Features with KS close to 1 inside matched classes are",
              "candidates to drop in an ablation (--features)."]
    text = "\n".join(lines) + "\n"
    (args.out / "diagnose_report.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
