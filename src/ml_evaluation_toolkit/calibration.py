"""Calibration: do the predicted probabilities mean what they say?

A model that predicts 0.8 should be right about 80% of the time. This
module measures that, per probability bin, and scores it with Expected
Calibration Error (ECE) and the Brier score. It also fits temperature
scaling — a single-parameter recalibration — honestly: the temperature
is fitted by minimising negative log-likelihood on the data supplied,
and the function returns both the fitted temperature and the
recalibrated probabilities so the improvement can be checked on
held-out data. Fitting and evaluating on the same data will flatter
the result; the README says so.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-15
# A bin is "well calibrated" when |mean predicted - observed| is within
# this tolerance; otherwise it is over- or under-confident.
_BIN_TOLERANCE = 0.05


def _softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _nll_for_temperature(logits, true_idx, temperature):
    scaled = _softmax(logits / temperature)
    probs = np.clip(scaled[np.arange(len(true_idx)), true_idx], _EPS, 1.0)
    return float(-np.log(probs).mean())


def _golden_section_minimise(func, low, high, iterations=100):
    """Minimise a unimodal 1-D function on [low, high]."""
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = low, high
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
    fc, fd = func(c), func(d)
    for _ in range(iterations):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = func(c)
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = func(d)
    return (a + b) / 2.0


def reliability_bins(y_true, y_proba, n_bins=10, labels=None):
    """Reliability-diagram data, per probability bin.

    Binary input: ``y_true`` is 0/1 and ``y_proba`` a 1-D array of
    positive-class probabilities. Multiclass input: ``y_true`` holds
    labels and ``y_proba`` is an (n, n_classes) matrix; each row
    contributes its maximum probability (the model's confidence) and
    whether the argmax prediction was correct — this is confidence
    calibration, the standard multiclass analogue.

    Each bin reports its range, count, mean predicted probability,
    observed frequency, the gap (predicted - observed) and a verdict:
    "well calibrated" when |gap| <= 0.05, "over-confident" when the
    model claims more than it delivers, "under-confident" otherwise.
    Empty bins are omitted.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(y_proba, dtype=float)
    if proba.ndim == 1:
        proba = proba.reshape(-1, 1)
    if len(y_true) != len(proba):
        raise ValueError("y_true and y_proba must have the same length")
    if proba.shape[1] == 1:
        if set(np.unique(y_true).tolist()) - {0, 1}:
            raise ValueError("1-D y_proba needs binary 0/1 y_true")
        confidence = proba[:, 0]
        correct = (y_true == 1).astype(float)
        mode = "binary"
    else:
        if labels is None:
            labels = sorted(set(y_true.tolist()), key=lambda v: (str(type(v)), str(v)))
        if proba.shape[1] != len(labels):
            raise ValueError(
                f"y_proba has {proba.shape[1]} columns but {len(labels)} labels were given"
            )
        index = {label: i for i, label in enumerate(labels)}
        true_idx = np.array([index[v] for v in y_true], dtype=int)
        pred_idx = proba.argmax(axis=1)
        confidence = proba.max(axis=1)
        correct = (pred_idx == true_idx).astype(float)
        mode = "confidence"
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        mean_pred = float(confidence[mask].mean())
        observed = float(correct[mask].mean())
        gap = mean_pred - observed
        if abs(gap) <= _BIN_TOLERANCE:
            verdict = "well calibrated"
        elif gap > 0:
            verdict = "over-confident"
        else:
            verdict = "under-confident"
        bins.append({
            "bin": [float(lo), float(hi)],
            "count": count,
            "mean_predicted": mean_pred,
            "observed_frequency": observed,
            "gap": float(gap),
            "verdict": verdict,
        })
    return {"mode": mode, "n_bins": n_bins, "bins": bins}


def expected_calibration_error(y_true, y_proba, n_bins=10, labels=None):
    """ECE: count-weighted mean |mean predicted - observed| across bins."""
    result = reliability_bins(y_true, y_proba, n_bins=n_bins, labels=labels)
    total = sum(b["count"] for b in result["bins"])
    if total == 0:
        return 0.0
    return float(sum(b["count"] / total * abs(b["gap"]) for b in result["bins"]))


def brier_score(y_true, y_proba, labels=None):
    """Brier score: mean squared error of the probabilities.

    Binary: mean (p - y)^2 on the positive class. Multiclass: mean over
    rows of sum over classes (p_c - 1[y = c])^2. Lower is better; 0 is
    perfect.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(y_proba, dtype=float)
    if proba.ndim == 1:
        proba = proba.reshape(-1, 1)
    if len(y_true) != len(proba):
        raise ValueError("y_true and y_proba must have the same length")
    if proba.shape[1] == 1:
        if set(np.unique(y_true).tolist()) - {0, 1}:
            raise ValueError("1-D y_proba needs binary 0/1 y_true")
        return float(np.mean((proba[:, 0] - y_true.astype(float)) ** 2))
    if labels is None:
        labels = sorted(set(y_true.tolist()), key=lambda v: (str(type(v)), str(v)))
    if proba.shape[1] != len(labels):
        raise ValueError(
            f"y_proba has {proba.shape[1]} columns but {len(labels)} labels were given"
        )
    index = {label: i for i, label in enumerate(labels)}
    one_hot = np.zeros_like(proba)
    for r, v in enumerate(y_true):
        one_hot[r, index[v]] = 1.0
    return float(np.mean(np.sum((proba - one_hot) ** 2, axis=1)))


def log_loss(y_true, y_proba, labels=None):
    """Mean negative log-likelihood of the true class (natural log)."""
    y_true = np.asarray(y_true)
    proba = np.asarray(y_proba, dtype=float)
    if proba.ndim == 1:
        proba = proba.reshape(-1, 1)
    if len(y_true) != len(proba):
        raise ValueError("y_true and y_proba must have the same length")
    if proba.shape[1] == 1:
        if set(np.unique(y_true).tolist()) - {0, 1}:
            raise ValueError("1-D y_proba needs binary 0/1 y_true")
        p = np.clip(proba[:, 0], _EPS, 1.0 - _EPS)
        y = y_true.astype(float)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    if labels is None:
        labels = sorted(set(y_true.tolist()), key=lambda v: (str(type(v)), str(v)))
    if proba.shape[1] != len(labels):
        raise ValueError(
            f"y_proba has {proba.shape[1]} columns but {len(labels)} labels were given"
        )
    index = {label: i for i, label in enumerate(labels)}
    true_idx = np.array([index[v] for v in y_true], dtype=int)
    p = np.clip(proba[np.arange(len(true_idx)), true_idx], _EPS, 1.0)
    return float(-np.log(p).mean())


def fit_temperature(y_true, y_proba, labels=None):
    """Fit temperature scaling and return (temperature, calibrated_proba).

    Probabilities are turned into logits with log(p), divided by a
    single temperature T, and passed back through softmax. T is chosen
    by golden-section search on negative log-likelihood over
    [0.05, 10]. T > 1 softens over-confident probabilities; T < 1
    sharpens under-confident ones; T = 1 leaves them unchanged.

    Honest limits: this is a one-parameter fix. It can repair
    miscalibration of the probability *scale*, it cannot repair a
    model that ranks examples badly, and fitting it on the same data
    used to measure ECE will make the calibrated ECE look better than
    it is on new data. Fit on a calibration split, evaluate on held-out
    data.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(y_proba, dtype=float)
    if proba.ndim == 1:
        proba = proba.reshape(-1, 1)
    if len(y_true) != len(proba):
        raise ValueError("y_true and y_proba must have the same length")
    if proba.shape[1] == 1:
        if set(np.unique(y_true).tolist()) - {0, 1}:
            raise ValueError("1-D y_proba needs binary 0/1 y_true")
        p = np.clip(proba[:, 0], _EPS, 1.0 - _EPS)
        two_class = np.column_stack([1.0 - p, p])
        true_idx = y_true.astype(int)
        logits = np.log(np.clip(two_class, _EPS, 1.0))
        temperature = _golden_section_minimise(
            lambda t: _nll_for_temperature(logits, true_idx, t), 0.05, 10.0
        )
        calibrated = _softmax(logits / temperature)
        return float(temperature), calibrated[:, 1]
    if labels is None:
        labels = sorted(set(y_true.tolist()), key=lambda v: (str(type(v)), str(v)))
    if proba.shape[1] != len(labels):
        raise ValueError(
            f"y_proba has {proba.shape[1]} columns but {len(labels)} labels were given"
        )
    index = {label: i for i, label in enumerate(labels)}
    true_idx = np.array([index[v] for v in y_true], dtype=int)
    logits = np.log(np.clip(proba, _EPS, 1.0))
    temperature = _golden_section_minimise(
        lambda t: _nll_for_temperature(logits, true_idx, t), 0.05, 10.0
    )
    calibrated = _softmax(logits / temperature)
    return float(temperature), calibrated


def calibration_summary(y_true, y_proba, n_bins=10, labels=None, fit_temp=True):
    """One-call calibration summary, with optional temperature scaling."""
    summary = {
        "ece": expected_calibration_error(y_true, y_proba, n_bins=n_bins, labels=labels),
        "brier_score": brier_score(y_true, y_proba, labels=labels),
        "log_loss": log_loss(y_true, y_proba, labels=labels),
        "reliability": reliability_bins(y_true, y_proba, n_bins=n_bins, labels=labels),
    }
    if fit_temp:
        temperature, calibrated = fit_temperature(y_true, y_proba, labels=labels)
        summary["temperature"] = temperature
        summary["calibrated_ece"] = expected_calibration_error(
            y_true, calibrated, n_bins=n_bins, labels=labels
        )
        summary["calibrated_brier_score"] = brier_score(y_true, calibrated, labels=labels)
        summary["calibrated_log_loss"] = log_loss(y_true, calibrated, labels=labels)
    return summary
