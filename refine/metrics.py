from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_auc_score


def classification_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_score > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    recall = recall_score(y_true, y_pred, zero_division=0)
    return {
        "acc": accuracy_score(y_true, y_pred),
        "rec": recall,
        "pre": precision_score(y_true, y_pred, zero_division=0),
        "spe": tn / (tn + fp) if (tn + fp) else 0.0,
        "sen": recall,
        "auc": roc_auc_score(y_true, y_score),
    }

