"""Unit tests for fraud_calibrated.costs: the expected-cost decision rule."""

from __future__ import annotations

import numpy as np
import pytest

from fraud_calibrated.costs import (
    CostModel,
    decide,
    global_threshold_cost,
    optimal_threshold_per_row,
    realized_cost,
    total_cost,
)


def test_cost_model_rejects_out_of_range_rates() -> None:
    with pytest.raises(ValueError):
        CostModel(lgd=1.5)
    with pytest.raises(ValueError):
        CostModel(margin_rate=-0.1)
    with pytest.raises(ValueError):
        CostModel(intervention_cost=-1.0)


def test_cost_matrix_scales_with_exposure() -> None:
    model = CostModel(lgd=0.5, intervention_cost=100.0, margin_rate=0.1)
    exposure = np.array([1000.0, 2000.0])
    fn_cost, fp_cost = model.cost_matrix(exposure)
    np.testing.assert_allclose(fn_cost, [500.0, 1000.0])
    np.testing.assert_allclose(fp_cost, [200.0, 300.0])


def test_optimal_threshold_higher_for_low_exposure_rows() -> None:
    model = CostModel()
    exposure = np.array([100.0, 1_000_000.0])
    thresholds = optimal_threshold_per_row(np.array([0.5, 0.5]), exposure, model)
    # Low exposure -> intervention cost dominates -> higher bar to flag.
    assert thresholds[0] > thresholds[1]


def test_optimal_threshold_clipped_to_unit_interval() -> None:
    model = CostModel(intervention_cost=1e9)
    exposure = np.array([1.0, 100.0])
    thresholds = optimal_threshold_per_row(np.array([0.9, 0.9]), exposure, model)
    assert np.all(thresholds <= 1.0)
    assert np.all(thresholds >= 0.0)


def test_decide_flags_high_probability_high_exposure_rows() -> None:
    model = CostModel()
    prob = np.array([0.99, 0.01])
    exposure = np.array([100_000.0, 100_000.0])
    flagged = decide(prob, exposure, model)
    assert flagged[0] and not flagged[1]


def test_realized_cost_is_zero_for_correct_decisions() -> None:
    model = CostModel()
    y_true = np.array([1, 0])
    flagged = np.array([True, False])
    exposure = np.array([5000.0, 5000.0])
    cost = realized_cost(y_true, flagged, exposure, model)
    np.testing.assert_allclose(cost, [0.0, 0.0])


def test_realized_cost_charges_the_right_side_for_mistakes() -> None:
    model = CostModel(lgd=0.75, intervention_cost=500.0, margin_rate=0.05)
    y_true = np.array([1, 0])
    flagged = np.array([False, True])  # missed defaulter, wrongly flagged good customer
    exposure = np.array([10_000.0, 10_000.0])
    cost = realized_cost(y_true, flagged, exposure, model)
    assert cost[0] == pytest.approx(0.75 * 10_000.0)
    assert cost[1] == pytest.approx(0.05 * 10_000.0 + 500.0)


def test_total_cost_sums_realized_cost() -> None:
    model = CostModel()
    y_true = np.array([1, 0, 1])
    flagged = np.array([True, True, False])
    exposure = np.array([1000.0, 1000.0, 1000.0])
    assert total_cost(y_true, flagged, exposure, model) == pytest.approx(
        realized_cost(y_true, flagged, exposure, model).sum()
    )


def test_global_threshold_cost_matches_manual_computation() -> None:
    model = CostModel()
    y_true = np.array([1, 0, 1, 0])
    prob = np.array([0.9, 0.8, 0.3, 0.1])
    exposure = np.array([1000.0, 2000.0, 3000.0, 4000.0])
    thresholds = np.array([0.5])
    costs = global_threshold_cost(y_true, prob, exposure, model, thresholds)
    expected = total_cost(y_true, prob >= 0.5, exposure, model)
    assert costs[0] == pytest.approx(expected)
