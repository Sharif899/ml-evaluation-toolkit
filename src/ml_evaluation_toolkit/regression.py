"""Regression evaluation over saved predictions.

MAE, RMSE, R² and a zero-safe MAPE, plus residual analysis by a
categorical feature column. Like the rest of the toolkit this evaluates
predictions already made; it does not fit models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def mean_absolute_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if len(y_true) == 0:
        raise ValueError("y_true and y_pred must not be empty")
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if len(y_true) == 0:
        raise ValueError("y_true and y_pred must not be empty")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def r2_score(y_true, y_pred):
    """Coefficient of determination. 1.0 is perfect; 0.0 means the model
    does no better than predicting the mean; negative means worse."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if len(y_true) == 0:
        raise ValueError("y_true and y_pred must not be empty")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def mean_absolute_percentage_error(y_true, y_pred):
    """MAPE with a zero-safe definition.

    Rows where the true value is exactly 0 are excluded from the mean
    — a percentage error against a zero denominator is undefined, not
    infinite — and the count of excluded rows is reported alongside
    the value, so the exclusion is visible rather than silent.
    Returns ``{"mape": ..., "excluded_zero_true": ...}``.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if len(y_true) == 0:
        raise ValueError("y_true and y_pred must not be empty")
    nonzero = y_true != 0
    excluded = int((~nonzero).sum())
    if nonzero.sum() == 0:
        return {"mape": None, "excluded_zero_true": excluded}
    ape = np.abs(y_true[nonzero] - y_pred[nonzero]) / np.abs(y_true[nonzero])
    return {"mape": float(np.mean(ape)), "excluded_zero_true": excluded}


def residual_by_slice(df, slice_col, y_true_col="y_true", y_pred_col="y_pred"):
    """Residual statistics per value of a categorical feature column.

    Each row reports count, MAE, RMSE and mean residual (signed: a
    positive mean residual means the model under-predicts that slice,
    a negative one means it over-predicts). Rows are sorted by MAE
    descending, so the worst slice is first.
    """
    if slice_col not in df.columns:
        raise ValueError(f"column {slice_col!r} not found in the data")
    if y_true_col not in df.columns or y_pred_col not in df.columns:
        raise ValueError(f"columns {y_true_col!r} and {y_pred_col!r} must be present")
    rows = []
    for value, group in df.groupby(df[slice_col], sort=True, dropna=False):
        y_true = group[y_true_col].to_numpy(dtype=float)
        y_pred = group[y_pred_col].to_numpy(dtype=float)
        residuals = y_true - y_pred
        rows.append({
            "slice": value,
            "count": len(group),
            "mae": float(np.mean(np.abs(residuals))),
            "rmse": float(np.sqrt(np.mean(residuals ** 2))),
            "mean_residual": float(np.mean(residuals)),
        })
    rows.sort(key=lambda r: (-r["mae"], -r["count"], str(r["slice"])))
    return rows


def regression_summary(y_true, y_pred):
    """One-call regression summary."""
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {
        "count": len(np.asarray(y_true)),
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mape": mape["mape"],
        "mape_excluded_zero_true": mape["excluded_zero_true"],
    }
