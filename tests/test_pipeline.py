"""End-to-end pipeline test on the synthetic fixture (fast, no network)."""

from __future__ import annotations

from fraud_calibrated.costs import CostModel
from fraud_calibrated.pipeline import run


def test_pipeline_runs_and_produces_bounded_metrics(synthetic_frame) -> None:
    result = run(synthetic_frame, seed=0)
    m = result.metrics
    assert m["n_train"] + m["n_valid"] + m["n_test"] == len(synthetic_frame)
    assert 0.0 <= m["auc"]["lightgbm"] <= 1.0
    assert 0.0 <= m["auc"]["baseline"] <= 1.0
    assert m["brier"]["before"] >= 0.0
    assert m["brier"]["after"] >= 0.0
    assert m["cost"]["per_row_threshold"] >= 0.0
    assert 0 <= m["cost"]["n_flagged_per_row"] <= m["n_test"]


def test_pipeline_is_deterministic_given_seed(synthetic_frame) -> None:
    a = run(synthetic_frame, seed=3)
    b = run(synthetic_frame, seed=3)
    assert a.metrics == b.metrics


def test_pipeline_respects_custom_cost_model(synthetic_frame) -> None:
    cheap_intervention = run(synthetic_frame, seed=0, cost_model=CostModel(intervention_cost=1.0))
    expensive_intervention = run(
        synthetic_frame, seed=0, cost_model=CostModel(intervention_cost=100_000.0)
    )
    # A near-free intervention should never be flagged less often than a very
    # expensive one, for the same scores and exposures.
    assert (
        cheap_intervention.metrics["cost"]["n_flagged_per_row"]
        >= expensive_intervention.metrics["cost"]["n_flagged_per_row"]
    )


def test_global_threshold_cost_is_computed_over_the_full_grid(synthetic_frame) -> None:
    # Whether the per-row rule beats every global cutoff is an empirical, large-sample
    # result (checked in test_pipeline_integration.py against the real dataset, where
    # ~6,000 test rows make the comparison meaningful) rather than a guarantee that
    # holds on an 80-row synthetic test fold. Here we only check the grid is sane.
    import numpy as np

    from fraud_calibrated import costs

    result = run(synthetic_frame, seed=0)
    thresholds = np.linspace(0.01, 0.99, 50)
    global_costs = costs.global_threshold_cost(
        result.test_ds.y,
        result.test_prob_calibrated,
        result.test_exposure,
        result.cost_model,
        thresholds,
    )
    assert global_costs.shape == thresholds.shape
    assert np.all(global_costs >= 0.0)
