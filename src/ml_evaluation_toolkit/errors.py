"""Error analysis over saved predictions.

Finds *where* a classifier fails: which classes it confuses, which
subgroups (a categorical feature column) carry the errors, and which
individual examples it gets wrong with the most confidence. Everything
here is descriptive — it reports the errors in the predictions it is
given; it does not retrain or repair the model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .metrics import _as_labels, confusion_matrix_counts


def confusion_pairs(y_true, y_pred, labels=None, max_examples=5):
    """Off-diagonal confusion pairs, most frequent first.

    Each entry is ``{"true": ..., "predicted": ..., "count": ...,
    "example_indices": [...]}`` — the indices are row positions in the
    input, so the examples can be looked up in the original data.
    """
    matrix, labels = confusion_matrix_counts(y_true, y_pred, labels)
    true_idx, pred_idx, _ = _as_labels(y_true, y_pred, labels)
    pairs = []
    n = len(labels)
    for i in range(n):
        for j in range(n):
            if i == j or matrix[i, j] == 0:
                continue
            mask = (true_idx == i) & (pred_idx == j)
            indices = np.flatnonzero(mask).tolist()[:max_examples]
            pairs.append({
                "true": labels[i],
                "predicted": labels[j],
                "count": int(matrix[i, j]),
                "example_indices": indices,
            })
    pairs.sort(key=lambda p: (-p["count"], str(p["true"]), str(p["predicted"])))
    return pairs


def error_rate_by_true_class(y_true, y_pred, labels=None):
    """Error rate for each true class: share of that class predicted wrong."""
    matrix, labels = confusion_matrix_counts(y_true, y_pred, labels)
    out = []
    for i, label in enumerate(labels):
        support = int(matrix[i].sum())
        errors = support - int(matrix[i, i])
        out.append({
            "label": label,
            "support": support,
            "errors": errors,
            "error_rate": (errors / support) if support else 0.0,
        })
    return out


def error_rate_by_predicted_class(y_true, y_pred, labels=None):
    """Error rate for each predicted class: share of predictions of that
    class that were wrong (1 - precision, in other words)."""
    matrix, labels = confusion_matrix_counts(y_true, y_pred, labels)
    out = []
    for j, label in enumerate(labels):
        predicted = int(matrix[:, j].sum())
        errors = predicted - int(matrix[j, j])
        out.append({
            "label": label,
            "predicted": predicted,
            "errors": errors,
            "error_rate": (errors / predicted) if predicted else 0.0,
        })
    return out


def error_rate_by_slice(df, slice_col, y_true_col="y_true", y_pred_col="y_pred"):
    """Error rate broken down by a categorical feature column.

    ``df`` is a DataFrame with the true labels, predicted labels and a
    column to slice on (for example a subgroup, a data source or a
    time period). Returns one row per slice value, sorted by error
    rate descending, then by count descending. A slice is only as
    meaningful as the column supplied: a column with one value for
    every row says nothing, and a column that leaks the label will
    flatter the model.
    """
    if slice_col not in df.columns:
        raise ValueError(f"column {slice_col!r} not found in the data")
    if y_true_col not in df.columns or y_pred_col not in df.columns:
        raise ValueError(f"columns {y_true_col!r} and {y_pred_col!r} must be present")
    rows = []
    for value, group in df.groupby(df[slice_col], sort=True, dropna=False):
        n = len(group)
        errors = int((group[y_true_col] != group[y_pred_col]).sum())
        rows.append({
            "slice": value,
            "count": n,
            "errors": errors,
            "error_rate": errors / n if n else 0.0,
        })
    rows.sort(key=lambda r: (-r["error_rate"], -r["count"], str(r["slice"])))
    return rows


def hard_examples(y_true, y_pred, y_proba, labels=None, top_n=10):
    """The errors the model was most confident about.

    For each misclassified row, the loss is ``-log P(true class)`` and
    the confidence is ``P(predicted class)``. The list is sorted by
    loss descending — the most confidently wrong predictions first.
    ``y_proba`` may be a 1-D array of positive-class probabilities for
    a binary problem, or an (n, n_classes) matrix aligned with
    ``labels``.
    """
    true_idx, pred_idx, labels = _as_labels(y_true, y_pred, labels)
    proba = np.asarray(y_proba, dtype=float)
    if proba.ndim == 1:
        if len(labels) != 2:
            raise ValueError("1-D y_proba needs exactly 2 labels")
        proba = np.column_stack([1.0 - proba, proba])
    if proba.shape[1] != len(labels):
        raise ValueError(
            f"y_proba has {proba.shape[1]} columns but {len(labels)} labels were given"
        )
    if len(true_idx) != len(proba):
        raise ValueError("y_true and y_proba must have the same length")
    clipped = np.clip(proba, 1e-15, 1.0)
    wrong = np.flatnonzero(true_idx != pred_idx)
    out = []
    for i in wrong:
        prob_true = float(clipped[i, true_idx[i]])
        prob_pred = float(clipped[i, pred_idx[i]])
        out.append({
            "index": int(i),
            "true": labels[true_idx[i]],
            "predicted": labels[pred_idx[i]],
            "prob_true": prob_true,
            "prob_predicted": prob_pred,
            "loss": float(-np.log(prob_true)),
        })
    out.sort(key=lambda r: (-r["loss"], -r["prob_predicted"], r["index"]))
    return out[:top_n]


def error_summary(y_true, y_pred, y_proba=None, labels=None, top_n_pairs=5, top_n_hard=10):
    """One-call error summary: pairs, per-class rates, hard examples."""
    summary = {
        "total": len(np.asarray(y_true)),
        "errors": int((np.asarray(y_true) != np.asarray(y_pred)).sum()),
        "confusion_pairs": confusion_pairs(y_true, y_pred, labels)[:top_n_pairs],
        "error_rate_by_true_class": error_rate_by_true_class(y_true, y_pred, labels),
        "error_rate_by_predicted_class": error_rate_by_predicted_class(y_true, y_pred, labels),
    }
    summary["error_rate"] = (
        summary["errors"] / summary["total"] if summary["total"] else 0.0
    )
    if y_proba is not None:
        summary["hard_examples"] = hard_examples(
            y_true, y_pred, y_proba, labels, top_n=top_n_hard
        )
    return summary


def slice_error_report(df, slice_col, y_true_col="y_true", y_pred_col="y_pred"):
    """Error rates by slice plus the overall rate, for a DataFrame."""
    rows = error_rate_by_slice(df, slice_col, y_true_col, y_pred_col)
    total = len(df)
    errors = int((df[y_true_col] != df[y_pred_col]).sum())
    return {
        "slice_column": slice_col,
        "overall_error_rate": errors / total if total else 0.0,
        "overall_errors": errors,
        "overall_count": total,
        "slices": rows,
    }
