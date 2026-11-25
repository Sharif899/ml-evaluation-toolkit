"""Tests for classification metrics, cross-checked against scikit-learn."""

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    matthews_corrcoef,
    mean_absolute_error,
    precision_recall_fscore_support,
    roc_auc_score,
)

from ml_evaluation_toolkit import metrics

rng = np.random.default_rng(0)
Y_TRUE_BIN = rng.integers(0, 2, size=120)
Y_PRED_BIN = rng.integers(0, 2, size=120)
PROBA_BIN = rng.random(120)
Y_TRUE_MC = rng.integers(0, 3, size=150)
Y_PRED_MC = rng.integers(0, 3, size=150)
PROBA_MC = rng.random((150, 3))
PROBA_MC = PROBA_MC / PROBA_MC.sum(axis=1, keepdims=True)


def test_confusion_matrix_counts_matches_sklearn():
    matrix, labels = metrics.confusion_matrix_counts(Y_TRUE_MC, Y_PRED_MC)
    assert labels == [0, 1, 2]
    assert matrix.tolist() == confusion_matrix(Y_TRUE_MC, Y_PRED_MC).tolist()


def test_confusion_matrix_counts_with_string_labels():
    y_true = ["cat", "dog", "cat", "bird"]
    y_pred = ["cat", "cat", "dog", "bird"]
    matrix, labels = metrics.confusion_matrix_counts(y_true, y_pred)
    assert labels == ["bird", "cat", "dog"]
    assert matrix[1, 1] == 1  # cat -> cat
    assert matrix[1, 2] == 1  # cat -> dog


def test_confusion_matrix_normalized_rows_sum_to_one():
    norm, _ = metrics.confusion_matrix_normalized(Y_TRUE_MC, Y_PRED_MC)
    assert np.allclose(norm.sum(axis=1), 1.0)
    assert np.all((norm >= 0) & (norm <= 1))


def test_confusion_matrix_normalized_known_values():
    norm, labels = metrics.confusion_matrix_normalized([0, 0, 1], [0, 1, 1])
    assert labels == [0, 1]
    assert norm[0].tolist() == [0.5, 0.5]
    assert norm[1].tolist() == [0.0, 1.0]


def test_per_class_matches_sklearn_binary():
    report = metrics.per_class_report(Y_TRUE_BIN, Y_PRED_BIN)
    precision, recall, f1, support = precision_recall_fscore_support(
        Y_TRUE_BIN, Y_PRED_BIN, labels=[0, 1], zero_division=0
    )
    for i, row in enumerate(report["per_class"]):
        assert row["precision"] == pytest.approx(precision[i])
        assert row["recall"] == pytest.approx(recall[i])
        assert row["f1"] == pytest.approx(f1[i])
        assert row["support"] == support[i]


def test_per_class_matches_sklearn_multiclass():
    report = metrics.per_class_report(Y_TRUE_MC, Y_PRED_MC)
    precision, recall, f1, _ = precision_recall_fscore_support(
        Y_TRUE_MC, Y_PRED_MC, labels=[0, 1, 2], zero_division=0
    )
    assert report["macro"]["f1"] == pytest.approx(float(np.mean(f1)))
    for i, row in enumerate(report["per_class"]):
        assert row["f1"] == pytest.approx(f1[i])


def test_accuracy_and_balanced_accuracy_match_sklearn():
    report = metrics.per_class_report(Y_TRUE_MC, Y_PRED_MC)
    assert report["accuracy"] == pytest.approx(accuracy_score(Y_TRUE_MC, Y_PRED_MC))
    assert report["balanced_accuracy"] == pytest.approx(
        balanced_accuracy_score(Y_TRUE_MC, Y_PRED_MC)
    )


def test_weighted_f1_matches_sklearn():
    report = metrics.per_class_report(Y_TRUE_MC, Y_PRED_MC)
    _, _, f1, support = precision_recall_fscore_support(
        Y_TRUE_MC, Y_PRED_MC, labels=[0, 1, 2], zero_division=0
    )
    expected = float(np.average(f1, weights=support))
    assert report["weighted"]["f1"] == pytest.approx(expected)


def test_perfect_predictions_score_one():
    report = metrics.per_class_report([0, 1, 2, 0], [0, 1, 2, 0])
    assert report["accuracy"] == 1.0
    assert report["macro"]["f1"] == 1.0
    assert all(r["f1"] == 1.0 for r in report["per_class"])


def test_zero_division_conventions():
    # Class 1 is never predicted: precision 0. Class 2 has no true examples.
    report = metrics.per_class_report([0, 0, 1, 1], [0, 0, 0, 0], labels=[0, 1, 2])
    by_label = {r["label"]: r for r in report["per_class"]}
    assert by_label[1]["precision"] == 0.0
    assert by_label[1]["recall"] == 0.0
    assert by_label[2]["recall"] == 0.0
    assert by_label[2]["support"] == 0


def test_mcc_matches_sklearn_binary():
    assert metrics.matthews_corrcoef(Y_TRUE_BIN, Y_PRED_BIN) == pytest.approx(
        matthews_corrcoef(Y_TRUE_BIN, Y_PRED_BIN)
    )


def test_mcc_matches_sklearn_multiclass():
    assert metrics.matthews_corrcoef(Y_TRUE_MC, Y_PRED_MC) == pytest.approx(
        matthews_corrcoef(Y_TRUE_MC, Y_PRED_MC)
    )


def test_mcc_perfect_and_degenerate():
    assert metrics.matthews_corrcoef([0, 1, 0, 1], [0, 1, 0, 1]) == pytest.approx(1.0)
    # Single class predicted for everything on single-class truth.
    assert metrics.matthews_corrcoef([1, 1, 1], [1, 1, 1]) == 0.0


def test_roc_auc_binary_matches_sklearn():
    result = metrics.roc_auc(Y_TRUE_BIN, PROBA_BIN)
    assert result["roc_auc"] == pytest.approx(roc_auc_score(Y_TRUE_BIN, PROBA_BIN))


def test_roc_auc_multiclass_macro_matches_sklearn():
    result = metrics.roc_auc(Y_TRUE_MC, PROBA_MC, labels=[0, 1, 2])
    expected = roc_auc_score(Y_TRUE_MC, PROBA_MC, multi_class="ovr", average="macro")
    assert result["roc_auc"] == pytest.approx(expected)
    assert set(result["per_class"]) == {0, 1, 2}


def test_roc_auc_single_class_present_returns_none():
    result = metrics.roc_auc([0, 0, 0], [0.1, 0.2, 0.3])
    assert result["roc_auc"] is None


def test_average_precision_binary_matches_sklearn():
    result = metrics.average_precision(Y_TRUE_BIN, PROBA_BIN)
    assert result["average_precision"] == pytest.approx(
        average_precision_score(Y_TRUE_BIN, PROBA_BIN)
    )


def test_average_precision_multiclass_matches_sklearn():
    result = metrics.average_precision(Y_TRUE_MC, PROBA_MC, labels=[0, 1, 2])
    expected = average_precision_score(Y_TRUE_MC, PROBA_MC, average="macro")
    assert result["average_precision"] == pytest.approx(expected)


def test_classification_summary_includes_everything():
    summary = metrics.classification_summary(Y_TRUE_MC, Y_PRED_MC, PROBA_MC, labels=[0, 1, 2])
    assert summary["accuracy"] == pytest.approx(accuracy_score(Y_TRUE_MC, Y_PRED_MC))
    assert "mcc" in summary
    assert "confusion_matrix" in summary
    assert summary["roc_auc"]["roc_auc"] is not None
    assert summary["average_precision"]["average_precision"] is not None


def test_classification_summary_without_proba():
    summary = metrics.classification_summary(Y_TRUE_BIN, Y_PRED_BIN)
    assert "roc_auc" not in summary
    assert summary["accuracy"] == pytest.approx(accuracy_score(Y_TRUE_BIN, Y_PRED_BIN))


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        metrics.per_class_report([0, 1], [0])
    with pytest.raises(ValueError):
        metrics.roc_auc([0, 1], [0.2])


def test_empty_inputs_raise():
    with pytest.raises(ValueError):
        metrics.per_class_report([], [])


def test_unknown_label_with_explicit_labels_raises():
    with pytest.raises(ValueError):
        metrics.per_class_report([0, 5], [0, 0], labels=[0, 1])


def test_proba_column_mismatch_raises():
    with pytest.raises(ValueError):
        metrics.roc_auc(Y_TRUE_MC, PROBA_BIN, labels=[0, 1, 2])
