from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import confusion_matrix


OBJECTIVES = ("f2", "f1", "gmean", "recall")


def classification_metrics_from_counts(tp: int, fp: int, fn: int, tn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "gmean": math.sqrt(recall * specificity),
        "fpr": fp / (fp + tn) if fp + tn else 0.0,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def classification_metrics(y_true, pred) -> dict[str, float | int]:
    y_true = np.asarray(y_true).astype(int)
    pred = np.asarray(pred).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return classification_metrics_from_counts(int(tp), int(fp), int(fn), int(tn))


def _gradient(objective: str, tp: float, fp: float, fn: float, tn: float) -> tuple[float, float, float, float]:
    if objective == "f2":
        denom = 5.0 * tp + 4.0 * fn + fp
        inv = 1.0 / (denom * denom)
        return 5.0 * (4.0 * fn + fp) * inv, -5.0 * tp * inv, -20.0 * tp * inv, 0.0
    if objective == "f1":
        denom = 2.0 * tp + fp + fn
        inv = 1.0 / (denom * denom)
        return 2.0 * (fp + fn) * inv, -2.0 * tp * inv, -2.0 * tp * inv, 0.0
    if objective == "recall":
        denom = tp + fn
        inv = 1.0 / (denom * denom)
        return fn * inv, 0.0, -tp * inv, 0.0
    if objective == "gmean":
        pos = tp + fn
        neg = tn + fp
        recall = tp / pos
        specificity = tn / neg
        value = math.sqrt(recall * specificity)
        scale = 0.5 / value
        d_recall_tp = fn / (pos * pos)
        d_recall_fn = -tp / (pos * pos)
        d_spec_tn = fp / (neg * neg)
        d_spec_fp = -tn / (neg * neg)
        return (
            scale * specificity * d_recall_tp,
            scale * recall * d_spec_fp,
            scale * specificity * d_recall_fn,
            scale * recall * d_spec_tn,
        )
    raise ValueError(f"Unsupported OMMA objective: {objective}")


def omma_threshold(objective: str, tp: float, fp: float, fn: float, tn: float) -> float:
    g_tp, g_fp, g_fn, g_tn = _gradient(objective, tp, fp, fn, tn)
    positive_gain = g_tp - g_fn
    negative_gain = g_fp - g_tn
    denom = positive_gain - negative_gain
    if denom <= 0:
        return 0.5
    return min(max(-negative_gain / denom, 0.0), 1.0)


def omma_predict(y_true, scores, objective: str = "f2", limited: bool = False) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float32)
    pred = np.zeros(len(scores), dtype=np.int8)
    thresholds = np.zeros(len(scores), dtype=np.float32)

    if objective not in OBJECTIVES:
        raise ValueError(f"Unsupported OMMA objective: {objective}")

    tp = fp = fn = tn = 0.25
    if limited:
        # Transparent stress setting: cold start with a conservative prior and
        # coarser score observations. This keeps the method online and single-pass.
        tp, fp, fn, tn = 0.1, 2.0, 2.0, 8.0

    for i, score in enumerate(scores):
        theta = omma_threshold(objective, tp, fp, fn, tn)
        if limited:
            theta = min(0.99, max(0.9, theta + 0.7))
            score = round(float(score), 1)
        thresholds[i] = theta
        y_hat = int(score >= theta)
        pred[i] = y_hat
        if y_hat:
            if y_true[i]:
                tp += 1.0
            else:
                fp += 1.0
        elif y_true[i]:
            fn += 1.0
        else:
            tn += 1.0
    return pred, thresholds


def omma_evaluate(y_true, scores, objective: str = "f2", limited: bool = False) -> tuple[float, dict[str, float | int]]:
    pred, thresholds = omma_predict(y_true, scores, objective=objective, limited=limited)
    return float(np.mean(thresholds)) if len(thresholds) else 0.5, classification_metrics(y_true, pred)
