"""Unit tests for fraud_calibrated.calibration."""

from __future__ import annotations

import numpy as np
import pytest

from fraud_calibrated.calibration import (
    apply_isotonic,
    brier_score,
    expected_calibration_error,
    fit_isotonic,
    report,
)


def test_brier_score_zero_for_perfect_predictions() -> None:
    y = np.array([1, 0, 1, 0])
    prob = np.array([1.0, 0.0, 1.0, 0.0])
    assert brier_score(y, prob) == pytest.approx(0.0)


def test_brier_score_matches_hand_computation() -> None:
    y = np.array([1, 0])
    prob = np.array([0.7, 0.3])
    assert brier_score(y, prob) == pytest.approx(((0.3) ** 2 + (0.3) ** 2) / 2)


def test_ece_zero_when_predictions_match_bin_true_rate() -> None:
    rng = np.random.default_rng(0)
    prob = np.full(2000, 0.3)
    y = (rng.random(2000) < 0.3).astype(int)
    ece, _, _, _ = expected_calibration_error(y, prob, n_bins=5)
    assert ece < 0.03  # sampling noise only, no systematic miscalibration


def test_ece_large_for_badly_miscalibrated_scores() -> None:
    y = np.zeros(500, dtype=int)
    prob = np.full(500, 0.9)  # confidently wrong every time
    ece, _, _, _ = expected_calibration_error(y, prob, n_bins=5)
    assert ece == pytest.approx(0.9, abs=1e-6)


def test_report_bundles_brier_and_ece() -> None:
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    prob = np.array([0.6, 0.4, 0.6, 0.4, 0.6, 0.4, 0.6, 0.4])
    result = report(y, prob, n_bins=4)
    assert result.brier > 0
    assert result.bin_count.sum() == len(y)


def test_isotonic_calibration_improves_or_matches_brier_on_held_out_fold() -> None:
    rng = np.random.default_rng(1)
    n = 3000
    true_prob = rng.uniform(0, 1, n)
    y = (rng.random(n) < true_prob).astype(int)
    # Miscalibrated scores: systematically overconfident (pushed toward extremes).
    miscalibrated = np.clip(true_prob + 0.3 * np.sign(true_prob - 0.5) * true_prob, 0, 1)

    split = n // 2
    calibrator = fit_isotonic(y[:split], miscalibrated[:split])
    calibrated_test = apply_isotonic(calibrator, miscalibrated[split:])

    before = brier_score(y[split:], miscalibrated[split:])
    after = brier_score(y[split:], calibrated_test)
    assert after <= before


def test_isotonic_output_stays_in_unit_interval() -> None:
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 200)
    prob = rng.uniform(0, 1, 200)
    calibrator = fit_isotonic(y, prob)
    out = apply_isotonic(calibrator, np.array([-1.0, 0.5, 2.0]))
    assert np.all(out >= 0.0) and np.all(out <= 1.0)
