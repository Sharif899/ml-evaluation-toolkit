"""Classification metrics over saved predictions.

This module evaluates predictions a model has already made; it does not
train models. Counts-based metrics (confusion matrix, per-class
precision/recall/F1, accuracy, balanced accuracy, Matthews correlation
coefficient) are computed directly from the confusion matrix, so the
definitions are visible in the code. Ranking metrics (ROC-AUC, average
precision) are delegated to scikit-learn, which is the reference
implementation for them — reimplementing them would only add risk.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _as_labels(y_true, y_pred, labels=None):
    """Return (true_idx, pred_idx, labels): integer codes into `labels`."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if len(y_true) == 0:
        raise ValueError("y_true and y_pred must not be empty")
    if labels is None:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()),
                        key=lambda v: (str(type(v)), str(v)))
    labels = list(labels)
    index = {label: i for i, label in enumerate(labels)}
    unknown = [v for v in list(y_true) + list(y_pred) if v not in index]
    if unknown:
        raise ValueError(f"labels contain values not in `labels`: {sorted(set(map(str, unknown)))}")
    true_idx = np.array([index[v] for v in y_true], dtype=int)
    pred_idx = np.array([index[v] for v in y_pred], dtype=int)
    return true_idx, pred_idx, labels


def confusion_matrix_counts(y_true, y_pred, labels=None):
    """Confusion matrix as counts. Rows are true labels, columns predicted.

    Returns (matrix, labels) where matrix[i][j] counts true label
    ``labels[i]`` predicted as ``labels[j]``.
    """
    true_idx, pred_idx, labels = _as_labels(y_true, y_pred, labels)
    n = len(labels)
    matrix = np.zeros((n, n), dtype=int)
    np.add.at(matrix, (true_idx, pred_idx), 1)
    return matrix, labels


def confusion_matrix_normalized(y_true, y_pred, labels=None):
    """Confusion matrix normalised by true-label row (rows sum to 1.0).

    A row of all zeros (a true label with no examples) stays all zeros.
    """
    matrix, labels = confusion_matrix_counts(y_true, y_pred, labels)
    row_sums = matrix.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalised = np.where(row_sums > 0, matrix / np.where(row_sums == 0, 1, row_sums), 0.0)
    return normalised, labels


def per_class_report(y_true, y_pred, labels=None):
    """Per-class precision, recall, F1 and support, plus averages.

    Returns a dict with ``per_class`` (a list of dicts, one per label, in
    label order), ``macro`` and ``weighted`` average dicts, ``accuracy``
    and ``balanced_accuracy``. A class with no predicted examples has
    precision 0.0; a class with no true examples has recall 0.0. These
    are the standard zero-division conventions, documented here rather
    than hidden.
    """
    matrix, labels = confusion_matrix_counts(y_true, y_pred, labels)
    total = int(matrix.sum())
    tp = np.diag(matrix).astype(float)
    predicted = matrix.sum(axis=0).astype(float)  # column sums
    actual = matrix.sum(axis=1).astype(float)     # row sums
    precision = np.where(predicted > 0, tp / np.where(predicted == 0, 1, predicted), 0.0)
    recall = np.where(actual > 0, tp / np.where(actual == 0, 1, actual), 0.0)
    f1 = np.where(
        (precision + recall) > 0,
        2 * precision * recall / np.where((precision + recall) == 0, 1, precision + recall),
        0.0,
    )
    per_class = [
        {
            "label": label,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(actual[i]),
        }
        for i, label in enumerate(labels)
    ]
    macro = {
        "precision": float(np.mean(precision)),
        "recall": float(np.mean(recall)),
        "f1": float(np.mean(f1)),
    }
    weights = actual / total if total else np.zeros(len(labels))
    weighted = {
        "precision": float(np.sum(weights * precision)),
        "recall": float(np.sum(weights * recall)),
        "f1": float(np.sum(weights * f1)),
    }
    accuracy = float(tp.sum() / total) if total else 0.0
    return {
        "per_class": per_class,
        "macro": macro,
        "weighted": weighted,
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recall)),
        "labels": labels,
    }


def matthews_corrcoef(y_true, y_pred, labels=None):
    """Matthews correlation coefficient, binary and multiclass.

    Multiclass MCC uses the standard generalisation:
    (c*s - sum_k p_k*t_k) / sqrt((s^2 - sum p_k^2) (s^2 - sum t_k^2)),
    where c is the trace, s the total, p_k predicted counts and t_k true
    counts per class. Returns 0.0 when the denominator is zero (a
    degenerate prediction, e.g. a single class predicted for everything
    on a single-class truth).
    """
    matrix, _ = confusion_matrix_counts(y_true, y_pred, labels)
    matrix = matrix.astype(float)
    c = float(np.trace(matrix))
    s = float(matrix.sum())
    p = matrix.sum(axis=0)  # predicted counts per class
    t = matrix.sum(axis=1)  # true counts per class
    numerator = c * s - float(np.dot(p, t))
    denominator = np.sqrt((s * s - float(np.dot(p, p))) * (s * s - float(np.dot(t, t))))
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def roc_auc(y_true, y_proba, labels=None):
    """ROC-AUC. Binary: probability of the positive class.

    Multiclass: macro average of one-vs-rest AUCs. A class whose
    one-vs-rest problem has only one outcome present is skipped and
    reported in ``per_class`` as None. Returns a dict with ``roc_auc``
    (the macro value, or the binary value) and ``per_class``.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(y_proba, dtype=float)
    if proba.ndim == 1:
        proba = proba.reshape(-1, 1)
    if len(y_true) != len(proba):
        raise ValueError("y_true and y_proba must have the same length")
    if labels is None:
        labels = sorted(set(y_true.tolist()), key=lambda v: (str(type(v)), str(v)))
    labels = list(labels)
    if len(labels) == 2 and proba.shape[1] == 1:
        # Binary: single column is the probability of labels[1].
        positive = labels[1]
        y_bin = (y_true == positive).astype(int)
        if len(np.unique(y_bin)) < 2:
            return {"roc_auc": None, "per_class": {positive: None}}
        return {"roc_auc": float(roc_auc_score(y_bin, proba[:, 0])),
                "per_class": {positive: float(roc_auc_score(y_bin, proba[:, 0]))}}
    if proba.shape[1] != len(labels):
        raise ValueError(
            f"y_proba has {proba.shape[1]} columns but {len(labels)} labels were given"
        )
    per_class = {}
    for i, label in enumerate(labels):
        y_bin = (y_true == label).astype(int)
        if len(np.unique(y_bin)) < 2:
            per_class[label] = None
        else:
            per_class[label] = float(roc_auc_score(y_bin, proba[:, i]))
    values = [v for v in per_class.values() if v is not None]
    return {
        "roc_auc": float(np.mean(values)) if values else None,
        "per_class": per_class,
    }


def average_precision(y_true, y_proba, labels=None):
    """Average precision (area under the precision-recall curve).

    Same conventions as :func:`roc_auc`: binary takes a single column
    for the positive class; multiclass is the macro average of
    one-vs-rest average precisions.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(y_proba, dtype=float)
    if proba.ndim == 1:
        proba = proba.reshape(-1, 1)
    if len(y_true) != len(proba):
        raise ValueError("y_true and y_proba must have the same length")
    if labels is None:
        labels = sorted(set(y_true.tolist()), key=lambda v: (str(type(v)), str(v)))
    labels = list(labels)
    if len(labels) == 2 and proba.shape[1] == 1:
        positive = labels[1]
        y_bin = (y_true == positive).astype(int)
        if len(np.unique(y_bin)) < 2:
            return {"average_precision": None, "per_class": {positive: None}}
        value = float(average_precision_score(y_bin, proba[:, 0]))
        return {"average_precision": value, "per_class": {positive: value}}
    if proba.shape[1] != len(labels):
        raise ValueError(
            f"y_proba has {proba.shape[1]} columns but {len(labels)} labels were given"
        )
    per_class = {}
    for i, label in enumerate(labels):
        y_bin = (y_true == label).astype(int)
        if len(np.unique(y_bin)) < 2:
            per_class[label] = None
        else:
            per_class[label] = float(average_precision_score(y_bin, proba[:, i]))
    values = [v for v in per_class.values() if v is not None]
    return {
        "average_precision": float(np.mean(values)) if values else None,
        "per_class": per_class,
    }


def classification_summary(y_true, y_pred, y_proba=None, labels=None):
    """One-call summary: per-class report, MCC, and ranking metrics.

    ``y_proba`` is optional. For binary problems it may be a 1-D array
    of positive-class probabilities or a 2-column matrix; for multiclass
    it must be an (n, n_classes) matrix aligned with ``labels``.
    """
    report = per_class_report(y_true, y_pred, labels)
    report["mcc"] = matthews_corrcoef(y_true, y_pred, labels)
    matrix, labels = confusion_matrix_counts(y_true, y_pred, labels)
    report["confusion_matrix"] = matrix.tolist()
    report["labels"] = labels
    if y_proba is not None:
        proba = np.asarray(y_proba, dtype=float)
        if proba.ndim == 2 and proba.shape[1] == 2 and len(labels) == 2:
            proba = proba[:, 1]
        report["roc_auc"] = roc_auc(y_true, proba, labels)
        report["average_precision"] = average_precision(y_true, proba, labels)
    return report
