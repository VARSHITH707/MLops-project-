"""Evaluation metrics for the failure classifier.

Accuracy alone is misleading here: with ~13% failures, always predicting "no failure"
scores ~87% accuracy while catching zero failures. A missed failure (false negative)
means an unplanned breakdown, so recall and F1 matter more.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.data.generate_data import TARGET


def compute_metrics(y_true, proba, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y_true)) == 2
    return {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, proba)) if both_classes else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "class_distribution": {"0": int((y_true == 0).sum()), "1": int((y_true == 1).sum())},
    }


def evaluate_model(model, df: pd.DataFrame, threshold: float) -> dict:
    proba = model.predict_proba(df)[:, 1]
    return compute_metrics(df[TARGET], proba, threshold)
