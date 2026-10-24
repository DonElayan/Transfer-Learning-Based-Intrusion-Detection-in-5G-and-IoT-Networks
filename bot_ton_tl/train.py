"""Training loops: supervised (source pre-training / target-only), linear-probe-then-
fine-tune (LP-FT), and domain-adversarial pre-training (DANN, Ganin et al. 2016).

Training is step-based with mini-batches drawn with replacement, so the same loop
serves 30-sample few-shot budgets and 500k-row pre-training. No torch DataLoader is
used (no worker processes - safe on Windows).
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, replace

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from .models import DomainDiscriminator, Net, make_encoder


@dataclass
class TrainConfig:
    steps: int = 2000
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    eval_every: int = 100
    patience: int = 6          # evaluations without improvement before stopping
    balanced: bool = True      # class-balanced batches (essential for BoT-IoT: 477 normal rows)
    device: str = "cpu"


class BatchSampler:
    def __init__(self, y: np.ndarray, batch_size: int, balanced: bool, rng: np.random.Generator):
        self.y, self.batch_size, self.balanced, self.rng = y, batch_size, balanced, rng
        self.members = [np.flatnonzero(y == c) for c in np.unique(y)]

    def __call__(self) -> np.ndarray:
        n = len(self.y)
        if not self.balanced or len(self.members) < 2:
            return self.rng.integers(0, n, size=min(self.batch_size, max(n, 1)))
        which = self.rng.integers(0, len(self.members), size=self.batch_size)
        idx = np.empty(self.batch_size, dtype=np.int64)
        for k, members in enumerate(self.members):
            mask = which == k
            if mask.any():
                idx[mask] = members[self.rng.integers(0, len(members), size=int(mask.sum()))]
        return idx


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, device: str = "cpu", batch: int = 16384) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(device)
        out.append(torch.softmax(model(xb), dim=1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0), dtype=np.float32)


def macro_f1(model: nn.Module, X: np.ndarray, y: np.ndarray, device: str) -> float:
    pred = predict_proba(model, X, device).argmax(1)
    return float(f1_score(y, pred, average="macro", zero_division=0))


def fit(model: nn.Module, X: np.ndarray, y: np.ndarray, cfg: TrainConfig, rng: np.random.Generator,
        X_val: np.ndarray | None = None, y_val: np.ndarray | None = None,
        param_groups: list[dict] | None = None) -> dict:
    """Supervised training with early stopping on validation macro-F1 (if given)."""
    device = cfg.device
    model.to(device)
    params = param_groups if param_groups is not None else [{"params": [p for p in model.parameters() if p.requires_grad]}]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    sampler = BatchSampler(y, cfg.batch_size, cfg.balanced, rng)
    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y.astype(np.int64)).to(device)
    has_val = X_val is not None and len(X_val) > 0
    best_score, best_state, bad, step = -math.inf, None, 0, 0
    for step in range(1, cfg.steps + 1):
        model.train()
        idx = torch.from_numpy(sampler()).to(device)
        loss = loss_fn(model(Xt[idx]), yt[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if has_val and (step % cfg.eval_every == 0 or step == cfg.steps):
            score = macro_f1(model, X_val, y_val, device)
            if score > best_score + 1e-4:
                best_score, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    return {"steps": step, "val_macro_f1": best_score if has_val else float("nan")}


def train_supervised(arch: str, X: np.ndarray, y: np.ndarray, n_classes: int, cfg: TrainConfig, seed: int,
                     X_val=None, y_val=None, head: str = "linear",
                     encoder_state: dict | None = None) -> tuple[Net, dict]:
    """encoder_state=None trains from scratch; otherwise the encoder starts from the given
    pre-trained weights and is trained exactly like the from-scratch model."""
    torch.manual_seed(seed)
    model = Net(make_encoder(arch, X.shape[1]), n_classes, head=head)
    if encoder_state is not None:
        model.encoder.load_state_dict(encoder_state)
    info = fit(model, X, y, cfg, np.random.default_rng(seed), X_val, y_val)
    return model, info


def finetune(encoder_state: dict, arch: str, X: np.ndarray, y: np.ndarray, n_classes: int, seed: int,
             cfg_lp: TrainConfig, cfg_ft: TrainConfig | None, X_val=None, y_val=None,
             head: str = "linear", encoder_lr_mult: float = 0.1) -> tuple[Net, dict]:
    """LP-FT (Kumar et al., ICLR 2022): train a new head on the frozen pre-trained encoder,
    then (if cfg_ft) unfreeze everything with a smaller encoder learning rate. cfg_ft=None
    gives the frozen-feature-extractor variant the repository's CNN-TL notebook used."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Net(make_encoder(arch, X.shape[1]), n_classes, head=head)
    model.encoder.load_state_dict(encoder_state)
    for p in model.encoder.parameters():
        p.requires_grad = False
    info_lp = fit(model, X, y, cfg_lp, rng, X_val, y_val)
    info = {"lp_steps": info_lp["steps"], "val_macro_f1": info_lp["val_macro_f1"]}
    if cfg_ft is not None:
        for p in model.encoder.parameters():
            p.requires_grad = True
        groups = [{"params": model.head.parameters(), "lr": cfg_ft.lr},
                  {"params": model.encoder.parameters(), "lr": cfg_ft.lr * encoder_lr_mult}]
        info_ft = fit(model, X, y, cfg_ft, rng, X_val, y_val, param_groups=groups)
        info.update(ft_steps=info_ft["steps"], val_macro_f1=info_ft["val_macro_f1"])
    return model, info


def train_dann(arch: str, Xs: np.ndarray, ys: np.ndarray, n_classes: int, Xt: np.ndarray, cfg: TrainConfig,
               seed: int, lambda_max: float = 1.0) -> tuple[Net, dict]:
    """Source-supervised training with a gradient-reversed domain discriminator on
    unlabeled target flows. Labels of the target domain are never touched. The final
    model is kept (no target labels exist for model selection in this stage)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = cfg.device
    model = Net(make_encoder(arch, Xs.shape[1]), n_classes).to(device)
    disc = DomainDiscriminator(model.encoder.out_dim).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(disc.parameters()), lr=cfg.lr,
                            weight_decay=cfg.weight_decay)
    ce, bce = nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()
    src_sampler = BatchSampler(ys, cfg.batch_size, cfg.balanced, rng)
    Xs_t = torch.from_numpy(Xs).to(device)
    ys_t = torch.from_numpy(ys.astype(np.int64)).to(device)
    Xt_t = torch.from_numpy(Xt).to(device)
    last_dom_acc = float("nan")
    for step in range(1, cfg.steps + 1):
        model.train()
        disc.train()
        p = step / cfg.steps
        alpha = lambda_max * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)
        si = torch.from_numpy(src_sampler()).to(device)
        ti = torch.from_numpy(rng.integers(0, len(Xt), size=cfg.batch_size)).to(device)
        zs = model.encoder(Xs_t[si])
        zt = model.encoder(Xt_t[ti])
        cls_loss = ce(model.head(zs), ys_t[si])
        dom_labels = torch.cat([torch.zeros(len(si)), torch.ones(len(ti))]).to(device)
        dom_logits = disc(torch.cat([zs, zt]), alpha)
        dom_loss = bce(dom_logits, dom_labels)
        loss = cls_loss + dom_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % cfg.eval_every == 0:
            last_dom_acc = float(((dom_logits > 0).float() == dom_labels).float().mean())
    # domain accuracy near 0.5 means the encoder no longer separates the datasets
    return model, {"steps": cfg.steps, "final_domain_acc": last_dom_acc}


def scaled(cfg: TrainConfig, **changes) -> TrainConfig:
    return replace(cfg, **changes)
