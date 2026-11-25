"""Tests for error analysis: confusion pairs, slices, hard examples."""

import numpy as np
import pandas as pd
import pytest

from ml_evaluation_toolkit import errors

Y_TRUE = [0, 0, 0, 1, 1, 1, 2, 2]
Y_PRED = [0, 1, 0, 1, 0, 1, 2, 1]
PROBA = [
    [0.9, 0.1, 0.0],
    [0.4, 0.6, 0.0],
    [0.8, 0.2, 0.0],
    [0.2, 0.8, 0.0],
    [0.7, 0.3, 0.0],
    [0.1, 0.9, 0.0],
    [0.1, 0.1, 0.8],
    [0.2, 0.7, 0.1],
]


def test_confusion_pairs_ordered_by_count():
    pairs = errors.confusion_pairs(Y_TRUE, Y_PRED, labels=[0, 1, 2])
    # Three off-diagonal pairs, each with count 1, sorted by label.
    assert len(pairs) == 3
    assert pairs[0]["true"] == 0 and pairs[0]["predicted"] == 1
    assert pairs[0]["count"] == 1
    assert all(p["count"] == 1 for p in pairs)


def test_confusion_pairs_example_indices():
    pairs = errors.confusion_pairs(Y_TRUE, Y_PRED, labels=[0, 1, 2])
    pair = next(p for p in pairs if p["true"] == 0 and p["predicted"] == 1)
    assert pair["example_indices"] == [1]
    pair = next(p for p in pairs if p["true"] == 2 and p["predicted"] == 1)
    assert pair["example_indices"] == [7]


def test_confusion_pairs_empty_when_perfect():
    assert errors.confusion_pairs([0, 1], [0, 1]) == []


def test_error_rate_by_true_class():
    rows = errors.error_rate_by_true_class(Y_TRUE, Y_PRED, labels=[0, 1, 2])
    by_label = {r["label"]: r for r in rows}
    assert by_label[0]["error_rate"] == pytest.approx(1 / 3)
    assert by_label[0]["errors"] == 1
    assert by_label[1]["error_rate"] == pytest.approx(1 / 3)
    assert by_label[2]["error_rate"] == pytest.approx(0.5)


def test_error_rate_by_predicted_class():
    rows = errors.error_rate_by_predicted_class(Y_TRUE, Y_PRED, labels=[0, 1, 2])
    by_label = {r["label"]: r for r in rows}
    # Predicted 1 four times, two of them wrong.
    assert by_label[1]["predicted"] == 4
    assert by_label[1]["error_rate"] == pytest.approx(0.5)
    # Predicted 2 once, correct.
    assert by_label[2]["error_rate"] == pytest.approx(0.0)


def test_error_rate_by_slice_known_values():
    df = pd.DataFrame({
        "y_true": [0, 0, 1, 1, 0, 1],
        "y_pred": [0, 1, 1, 0, 1, 1],
        "group": ["a", "a", "a", "b", "b", "b"],
    })
    rows = errors.error_rate_by_slice(df, "group")
    assert rows[0]["slice"] == "b"  # 2/3 errors, worst first
    assert rows[0]["error_rate"] == pytest.approx(2 / 3)
    assert rows[1]["slice"] == "a"  # 1/3 errors
    assert rows[1]["error_rate"] == pytest.approx(1 / 3)


def test_error_rate_by_slice_sorted_by_rate_then_count():
    df = pd.DataFrame({
        "y_true": [0, 0, 0, 0, 1, 1],
        "y_pred": [1, 0, 0, 0, 0, 1],
        "group": ["x", "x", "x", "x", "y", "y"],
    })
    rows = errors.error_rate_by_slice(df, "group")
    assert rows[0]["slice"] == "y"  # 0.5 error rate
    assert rows[1]["slice"] == "x"  # 0.25 error rate


def test_error_rate_by_slice_missing_column_raises():
    df = pd.DataFrame({"y_true": [0], "y_pred": [0]})
    with pytest.raises(ValueError):
        errors.error_rate_by_slice(df, "nope")


def test_hard_examples_only_errors_sorted_by_loss():
    hard = errors.hard_examples(Y_TRUE, Y_PRED, PROBA, labels=[0, 1, 2], top_n=10)
    # Errors are rows 1, 4, 7. Losses: row 7 -log(0.1)=2.303, row 4
    # -log(0.3)=1.204, row 1 -log(0.4)=0.916 — highest loss first.
    assert [h["index"] for h in hard] == [7, 4, 1]
    # Row 1: true 0, P(true)=0.4 -> loss -log(0.4) = 0.916.
    row1 = next(h for h in hard if h["index"] == 1)
    assert row1["loss"] == pytest.approx(0.916, abs=0.001)
    assert row1["prob_predicted"] == pytest.approx(0.6)


def test_hard_examples_top_n_limits():
    hard = errors.hard_examples(Y_TRUE, Y_PRED, PROBA, labels=[0, 1, 2], top_n=2)
    assert len(hard) == 2


def test_hard_examples_binary_1d_proba():
    hard = errors.hard_examples([0, 1, 1], [0, 0, 1], [0.1, 0.2, 0.9], labels=[0, 1])
    assert len(hard) == 1
    assert hard[0]["index"] == 1
    assert hard[0]["prob_true"] == pytest.approx(0.2)


def test_hard_examples_none_when_perfect():
    assert errors.hard_examples([0, 1], [0, 1], [0.9, 0.8], labels=[0, 1]) == []


def test_error_summary_structure():
    summary = errors.error_summary(Y_TRUE, Y_PRED, PROBA, labels=[0, 1, 2])
    assert summary["total"] == 8
    assert summary["errors"] == 3
    assert summary["error_rate"] == pytest.approx(3 / 8)
    assert len(summary["confusion_pairs"]) == 3
    assert len(summary["hard_examples"]) == 3


def test_error_summary_without_proba_has_no_hard_examples():
    summary = errors.error_summary(Y_TRUE, Y_PRED, labels=[0, 1, 2])
    assert "hard_examples" not in summary


def test_slice_error_report_overall_rate():
    df = pd.DataFrame({
        "y_true": [0, 0, 1, 1],
        "y_pred": [0, 1, 1, 1],
        "group": ["a", "a", "b", "b"],
    })
    report = errors.slice_error_report(df, "group")
    assert report["overall_error_rate"] == pytest.approx(0.25)
    assert report["overall_count"] == 4
    assert report["slice_column"] == "group"
