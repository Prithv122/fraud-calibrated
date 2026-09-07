"""Unit tests for fraud_calibrated.drift."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_calibrated.drift import (
    feature_psi_table,
    performance_drop,
    population_stability_index,
)


def test_psi_near_zero_for_identical_distributions() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    cur = rng.normal(0, 1, 5000)
    assert population_stability_index(ref, cur) < 0.05


def test_psi_large_for_shifted_distributions() -> None:
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    cur = rng.normal(3, 1, 5000)  # a real, large mean shift
    assert population_stability_index(ref, cur) > 0.25


def test_psi_nan_for_degenerate_reference() -> None:
    ref = np.zeros(100)
    cur = np.ones(100)
    assert np.isnan(population_stability_index(ref, cur))


def test_feature_psi_table_sorted_worst_first() -> None:
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"stable": rng.normal(0, 1, 2000), "shifted": rng.normal(0, 1, 2000)})
    cur = pd.DataFrame({"stable": rng.normal(0, 1, 2000), "shifted": rng.normal(4, 1, 2000)})
    table = feature_psi_table(ref, cur, ["stable", "shifted"])
    assert table.iloc[0]["feature"] == "shifted"
    assert table.iloc[0]["psi"] > table.iloc[1]["psi"]


def test_performance_drop_computes_deltas() -> None:
    rng = np.random.default_rng(0)
    y_ref = rng.integers(0, 2, 500)
    prob_ref = np.clip(y_ref + rng.normal(0, 0.2, 500), 0, 1)
    y_cur = rng.integers(0, 2, 500)
    prob_cur = rng.uniform(0, 1, 500)  # uninformative on the shifted slice
    result = performance_drop(y_ref, prob_ref, y_cur, prob_cur)
    assert result.reference_auc > result.current_auc
    assert result.auc_delta < 0
