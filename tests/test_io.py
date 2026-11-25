"""Tests for prediction-file loading (CSV / JSONL)."""

import json

import numpy as np
import pandas as pd
import pytest

from ml_evaluation_toolkit.io import labels_from_proba_columns, load_predictions


def _write_csv(tmp_path, name, df):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def test_load_csv_with_single_proba_column(tmp_path):
    path = _write_csv(tmp_path, "p.csv", pd.DataFrame({
        "y_true": [0, 1, 1], "y_pred": [0, 0, 1], "y_proba": [0.2, 0.4, 0.9],
    }))
    df, y_true, y_pred, y_proba, labels = load_predictions(path)
    assert y_true.tolist() == [0, 1, 1]
    assert y_proba.tolist() == [0.2, 0.4, 0.9]
    assert labels == [0, 1]


def test_load_csv_with_proba_columns(tmp_path):
    path = _write_csv(tmp_path, "p.csv", pd.DataFrame({
        "y_true": [0, 1, 2], "y_pred": [0, 2, 2],
        "proba_0": [0.8, 0.1, 0.1], "proba_1": [0.1, 0.4, 0.2],
        "proba_2": [0.1, 0.5, 0.7],
    }))
    _, _, _, y_proba, labels = load_predictions(path)
    assert y_proba.shape == (3, 3)
    assert labels == [0, 1, 2]
    assert y_proba[0].tolist() == [0.8, 0.1, 0.1]


def test_load_jsonl(tmp_path):
    path = tmp_path / "p.jsonl"
    rows = [{"y_true": 0, "y_pred": 0, "y_proba": 0.3},
            {"y_true": 1, "y_pred": 1, "y_proba": 0.8}]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    _, y_true, y_pred, y_proba, _ = load_predictions(path)
    assert y_true.tolist() == [0, 1]
    assert y_proba.tolist() == pytest.approx([0.3, 0.8])


def test_load_without_proba_returns_none(tmp_path):
    path = _write_csv(tmp_path, "p.csv", pd.DataFrame({
        "y_true": [0, 1], "y_pred": [0, 1],
    }))
    _, _, _, y_proba, _ = load_predictions(path)
    assert y_proba is None


def test_load_missing_label_column_raises(tmp_path):
    path = _write_csv(tmp_path, "p.csv", pd.DataFrame({"y_true": [0, 1]}))
    with pytest.raises(ValueError, match="y_true and y_pred"):
        load_predictions(path)


def test_load_unsupported_extension_raises(tmp_path):
    path = tmp_path / "p.txt"
    path.write_text("hello")
    with pytest.raises(ValueError, match="unsupported file type"):
        load_predictions(path)


def test_labels_sorted_with_mixed_types():
    df = pd.DataFrame({"y_true": ["b", "a"], "y_pred": ["a", "b"]})
    # Labels are sorted by (type name, string value) — strings sort alphabetically.
    _, _, _, _, labels = _load_from_df(df)
    assert labels == ["a", "b"]


def _load_from_df(df):
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        df.to_csv(f, index=False)
        name = f.name
    try:
        return load_predictions(name)
    finally:
        os.unlink(name)


def test_labels_from_proba_columns():
    df = pd.DataFrame(columns=["proba_2", "proba_0", "proba_1"])
    assert labels_from_proba_columns(df) == [0, 1, 2]
    assert labels_from_proba_columns(pd.DataFrame(columns=["y_true"])) is None
