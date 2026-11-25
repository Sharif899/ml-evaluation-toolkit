"""Tests for calibration: ECE, Brier, log loss, reliability bins, temperature scaling."""

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, log_loss as sklearn_log_loss

from ml_evaluation_toolkit import calibration

rng = np.random.default_rng(1)


def test_ece_perfect_predictions_is_zero():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    assert calibration.expected_calibration_error(y, p) == pytest.approx(0.0)


def test_ece_known_value():
    # 10 rows, all p=0.5, 2 positives -> bin [0.5,0.6): gap |0.5-0.2|=0.3.
    y = np.array([1, 1] + [0] * 8)
    p = np.full(10, 0.5)
    assert calibration.expected_calibration_error(y, p, n_bins=10) == pytest.approx(0.3)


def test_brier_binary_matches_sklearn():
    y = rng.integers(0, 2, size=100)
    p = rng.random(100)
    assert calibration.brier_score(y, p) == pytest.approx(brier_score_loss(y, p))


def test_brier_perfect_is_zero():
    assert calibration.brier_score([0, 1], [0.0, 1.0]) == pytest.approx(0.0)


def test_brier_multiclass_manual():
    # Row 0: true class 0, p=[1,0,0] -> 0. Row 1: true class 1, p=[0.5,0.5,0]
    # -> (0.5^2 + 0.5^2 + 0) = 0.5. Mean = 0.25.
    y = [0, 1]
    p = [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]]
    assert calibration.brier_score(y, p, labels=[0, 1, 2]) == pytest.approx(0.25)


def test_log_loss_matches_sklearn_binary():
    y = rng.integers(0, 2, size=100)
    p = rng.random(100)
    assert calibration.log_loss(y, p) == pytest.approx(sklearn_log_loss(y, p))


def test_log_loss_matches_sklearn_multiclass():
    y = rng.integers(0, 3, size=100)
    p = rng.random((100, 3))
    p = p / p.sum(axis=1, keepdims=True)
    assert calibration.log_loss(y, p, labels=[0, 1, 2]) == pytest.approx(
        sklearn_log_loss(y, p, labels=[0, 1, 2])
    )


def test_reliability_bins_omit_empty_bins():
    result = calibration.reliability_bins([0, 1], [0.05, 0.95], n_bins=10)
    assert len(result["bins"]) == 2
    assert result["mode"] == "binary"


def test_reliability_bin_verdicts():
    # Over-confident: mean predicted 0.9, observed 0.0.
    result = calibration.reliability_bins([0] * 10, [0.9] * 10, n_bins=10)
    assert result["bins"][0]["verdict"] == "over-confident"
    # Under-confident: mean predicted 0.1, observed 1.0.
    result = calibration.reliability_bins([1] * 10, [0.1] * 10, n_bins=10)
    assert result["bins"][0]["verdict"] == "under-confident"
    # Well calibrated: mean predicted 0.5, observed 0.5.
    result = calibration.reliability_bins(
        [0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5], n_bins=10
    )
    assert result["bins"][0]["verdict"] == "well calibrated"


def test_reliability_bins_multiclass_confidence_mode():
    y = [0, 1, 2, 0]
    p = [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7], [0.4, 0.4, 0.2]]
    result = calibration.reliability_bins(y, p, n_bins=5, labels=[0, 1, 2])
    assert result["mode"] == "confidence"
    total = sum(b["count"] for b in result["bins"])
    assert total == 4


def test_temperature_overconfident_data_gets_t_above_one():
    # Sharpened probabilities: badly over-confident by construction.
    y = rng.integers(0, 2, size=400)
    raw = rng.random(400)
    sharp = raw ** 2 / (raw ** 2 + (1 - raw) ** 2)
    temperature, calibrated = calibration.fit_temperature(y, sharp)
    assert temperature > 1.0
    assert calibrated.shape == (400,)
    assert np.all((calibrated > 0) & (calibrated < 1))
    assert calibration.log_loss(y, calibrated) < calibration.log_loss(y, sharp)


def test_temperature_well_calibrated_data_stays_near_one():
    # Genuinely calibrated data: draw p, then draw y ~ Bernoulli(p).
    p = rng.random(2000)
    y = (rng.random(2000) < p).astype(int)
    temperature, _ = calibration.fit_temperature(y, p)
    assert 0.5 < temperature < 2.0


def test_temperature_multiclass_rows_sum_to_one():
    y = rng.integers(0, 3, size=200)
    p = rng.random((200, 3))
    p = p / p.sum(axis=1, keepdims=True)
    temperature, calibrated = calibration.fit_temperature(y, p, labels=[0, 1, 2])
    assert calibrated.shape == (200, 3)
    assert np.allclose(calibrated.sum(axis=1), 1.0)
    assert temperature > 0


def test_calibration_summary_has_all_keys():
    y = rng.integers(0, 2, size=80)
    p = rng.random(80)
    summary = calibration.calibration_summary(y, p)
    for key in ("ece", "brier_score", "log_loss", "reliability",
                "temperature", "calibrated_ece", "calibrated_brier_score",
                "calibrated_log_loss"):
        assert key in summary


def test_calibration_summary_without_temperature():
    y = rng.integers(0, 2, size=50)
    p = rng.random(50)
    summary = calibration.calibration_summary(y, p, fit_temp=False)
    assert "temperature" not in summary


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        calibration.expected_calibration_error([0, 1], [0.5])
    with pytest.raises(ValueError):
        calibration.brier_score([0, 1], [0.5, 0.6, 0.7])
