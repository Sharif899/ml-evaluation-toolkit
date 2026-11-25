"""Tests for regression metrics, cross-checked against scikit-learn."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    mean_absolute_error as sk_mae,
    mean_absolute_percentage_error as sk_mape,
    mean_squared_error as sk_mse,
    r2_score as sk_r2,
)

from ml_evaluation_toolkit import regression

rng = np.random.default_rng(2)
Y_TRUE = rng.normal(50, 15, size=120)
Y_PRED = Y_TRUE + rng.normal(0, 5, size=120)


def test_mae_matches_sklearn():
    assert regression.mean_absolute_error(Y_TRUE, Y_PRED) == pytest.approx(
        sk_mae(Y_TRUE, Y_PRED)
    )


def test_rmse_matches_sklearn():
    assert regression.root_mean_squared_error(Y_TRUE, Y_PRED) == pytest.approx(
        np.sqrt(sk_mse(Y_TRUE, Y_PRED))
    )


def test_r2_matches_sklearn():
    assert regression.r2_score(Y_TRUE, Y_PRED) == pytest.approx(sk_r2(Y_TRUE, Y_PRED))


def test_r2_perfect_is_one():
    assert regression.r2_score([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_r2_mean_prediction_is_zero():
    assert regression.r2_score([1.0, 2.0, 3.0], [2.0, 2.0, 2.0]) == pytest.approx(0.0)


def test_r2_constant_truth_is_zero():
    assert regression.r2_score([5.0, 5.0, 5.0], [4.0, 5.0, 6.0]) == 0.0


def test_mape_matches_sklearn_without_zeros():
    assert regression.mean_absolute_percentage_error(Y_TRUE, Y_PRED)["mape"] == pytest.approx(
        sk_mape(Y_TRUE, Y_PRED)
    )


def test_mape_excludes_zero_true_values():
    result = regression.mean_absolute_percentage_error([0.0, 100.0, 200.0], [0.0, 90.0, 220.0])
    # Only the two nonzero rows count: (0.1 + 0.1) / 2 = 0.1.
    assert result["mape"] == pytest.approx(0.1)
    assert result["excluded_zero_true"] == 1


def test_mape_all_zero_true_returns_none():
    result = regression.mean_absolute_percentage_error([0.0, 0.0], [1.0, 2.0])
    assert result["mape"] is None
    assert result["excluded_zero_true"] == 2


def test_regression_summary_matches_sklearn():
    summary = regression.regression_summary(Y_TRUE, Y_PRED)
    assert summary["count"] == 120
    assert summary["mae"] == pytest.approx(sk_mae(Y_TRUE, Y_PRED))
    assert summary["rmse"] == pytest.approx(np.sqrt(sk_mse(Y_TRUE, Y_PRED)))
    assert summary["r2"] == pytest.approx(sk_r2(Y_TRUE, Y_PRED))
    assert summary["mape_excluded_zero_true"] == 0


def test_residual_by_slice_known_values():
    df = pd.DataFrame({
        "y_true": [10.0, 12.0, 20.0, 22.0],
        "y_pred": [9.0, 13.0, 25.0, 21.0],
        "group": ["a", "a", "b", "b"],
    })
    rows = regression.residual_by_slice(df, "group")
    # Group b: residuals -5, +1 -> MAE 3. Group a: residuals +1, -1 -> MAE 1.
    assert rows[0]["slice"] == "b"
    assert rows[0]["mae"] == pytest.approx(3.0)
    assert rows[0]["mean_residual"] == pytest.approx(-2.0)
    assert rows[1]["slice"] == "a"
    assert rows[1]["mae"] == pytest.approx(1.0)
    assert rows[1]["mean_residual"] == pytest.approx(0.0)


def test_residual_by_slice_missing_column_raises():
    df = pd.DataFrame({"y_true": [1.0], "y_pred": [1.0]})
    with pytest.raises(ValueError):
        regression.residual_by_slice(df, "nope")


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        regression.mean_absolute_error([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        regression.r2_score([1.0], [1.0, 2.0])


def test_empty_inputs_raise():
    with pytest.raises(ValueError):
        regression.mean_absolute_error([], [])
    with pytest.raises(ValueError):
        regression.root_mean_squared_error([], [])
