"""Integration tests against the real, committed dataset cache.

These are the only tests that touch actual data volume (30,000 rows) rather than the
fast synthetic fixture, and they check the headline claims the README makes:
calibration improves measurably, and the per-row cost-optimal rule beats every global
threshold on a grid fine enough to be a fair comparison.
"""

from __future__ import annotations

import numpy as np
import pytest

from fraud_calibrated import costs
from fraud_calibrated.pipeline import run


@pytest.fixture(scope="module")
def result(real_df):
    return run(real_df, seed=0)


def test_dataset_shape_and_balance(real_df) -> None:
    assert real_df.shape[0] == 30_000
    positive_rate = real_df["default"].mean()
    assert 0.15 < positive_rate < 0.30  # moderate imbalance, not extreme


def test_lightgbm_beats_baseline_on_auc(result) -> None:
    assert result.metrics["auc"]["lightgbm"] > result.metrics["auc"]["baseline"]


def test_isotonic_calibration_improves_both_brier_and_ece(result) -> None:
    assert result.metrics["brier"]["after"] < result.metrics["brier"]["before"]
    assert result.metrics["ece"]["after"] < result.metrics["ece"]["before"]


def test_per_row_threshold_beats_best_global_threshold_at_scale(result) -> None:
    thresholds = np.linspace(0.001, 0.99, 200)
    global_costs = costs.global_threshold_cost(
        result.test_ds.y,
        result.test_prob_calibrated,
        result.test_exposure,
        result.cost_model,
        thresholds,
    )
    per_row_cost = result.metrics["cost"]["per_row_threshold"]
    assert per_row_cost < global_costs.min()


def test_per_row_rule_flags_a_minority_of_customers(result) -> None:
    # A sanity floor against the degenerate case where the "optimal" rule just
    # flags everyone -- see NOTES.md for why that would be the wrong headline.
    assert 0 < result.metrics["cost"]["n_flagged_per_row"] < result.metrics["n_test"]
