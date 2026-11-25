"""Loading prediction files (CSV / JSONL) into a common shape.

Expected columns: ``y_true`` and ``y_pred``. Probabilities are optional
and may be given as a single ``y_proba`` column (binary positive-class
probability) or as ``proba_0``, ``proba_1``, ... columns aligned with
the sorted unique labels. Any extra columns are kept — they are what
error and residual slicing runs on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_predictions(path):
    """Load a prediction file and return (df, y_true, y_pred, y_proba, labels).

    ``y_proba`` is None when no probability column is present.
    ``labels`` is the sorted unique labels from y_true and y_pred.
    """
    path = str(path)
    if path.endswith(".jsonl") or path.endswith(".json"):
        df = pd.read_json(path, lines=True)
    elif path.endswith(".csv"):
        df = pd.read_csv(path)
    else:
        raise ValueError(f"unsupported file type (need .csv or .jsonl): {path}")
    if "y_true" not in df.columns or "y_pred" not in df.columns:
        raise ValueError("prediction file must have y_true and y_pred columns")
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()),
                    key=lambda v: (str(type(v)), str(v)))
    y_proba = None
    if "y_proba" in df.columns:
        y_proba = df["y_proba"].to_numpy(dtype=float)
    else:
        proba_cols = [c for c in df.columns if str(c).startswith("proba_")]
        if proba_cols:
            ordered = sorted(proba_cols, key=lambda c: int(str(c).split("_", 1)[1]))
            y_proba = df[ordered].to_numpy(dtype=float)
    if y_proba is not None and len(y_proba) != len(df):
        raise ValueError("probability columns must have one value per row")
    return df, y_true, y_pred, y_proba, labels


def labels_from_proba_columns(df):
    """Label order implied by proba_0..proba_n columns, if present."""
    proba_cols = [c for c in df.columns if str(c).startswith("proba_")]
    if not proba_cols:
        return None
    ordered = sorted(proba_cols, key=lambda c: int(str(c).split("_", 1)[1]))
    return [int(str(c).split("_", 1)[1]) for c in ordered]
