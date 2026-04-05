"""
src/utils/metrics.py
Evaluation helpers: accuracy, F1, MAE, R².
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    mean_absolute_error, r2_score,
)


def cls_metrics(y_true, y_pred, average="macro") -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1":       f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def reg_metrics(y_true, y_pred) -> dict:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "r2":  r2_score(y_true, y_pred),
    }


def print_report(y_true, y_pred, task_name: str):
    print(f"\n{'='*50}")
    print(f"  {task_name}")
    print(f"{'='*50}")
    print(classification_report(y_true, y_pred, zero_division=0))
