"""Evaluation metrics. Macro scores are the headline numbers: accuracy is dominated by
the majority attack classes in both datasets and hides failures on normal traffic."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, roc_auc_score)


def binary_metrics(y_true: np.ndarray, p_attack: np.ndarray, threshold: float = 0.5) -> dict:
    y_pred = (p_attack >= threshold).astype(int)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall_dr": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    normal = y_true == 0
    out["far_fpr"] = float((y_pred[normal] == 1).mean()) if normal.any() else float("nan")
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = roc_auc_score(y_true, p_attack)
        out["pr_auc"] = average_precision_score(y_true, p_attack)
    else:
        out["roc_auc"] = out["pr_auc"] = float("nan")
    return {k: float(v) for k, v in out.items()}


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def per_class_rates(cls_true: np.ndarray, correct: np.ndarray) -> dict[str, float]:
    """Share of each original class handled correctly (detection rate for attack
    classes in the binary task, recall in the multiclass task)."""
    return {str(c): float(correct[cls_true == c].mean()) for c in np.unique(cls_true)}
