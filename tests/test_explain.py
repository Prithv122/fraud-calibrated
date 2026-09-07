"""Unit tests for fraud_calibrated.explain (SHAP)."""

from __future__ import annotations

import numpy as np

from fraud_calibrated.explain import explain
from fraud_calibrated.model import fit_lightgbm, make_dataset


def test_shap_values_shape_matches_data(synthetic_frame) -> None:
    ds = make_dataset(synthetic_frame)
    model = fit_lightgbm(ds, seed=0, n_estimators=30)
    result = explain(model, ds.x)
    assert result.values.shape == (len(ds.x), ds.x.shape[1])
    assert len(result.feature_names) == ds.x.shape[1]


def test_global_importance_sorted_descending(synthetic_frame) -> None:
    ds = make_dataset(synthetic_frame)
    model = fit_lightgbm(ds, seed=0, n_estimators=30)
    result = explain(model, ds.x)
    importance = result.global_importance()
    values = importance["mean_abs_shap"].to_numpy()
    assert np.all(values[:-1] >= values[1:])


def test_shap_values_plus_base_approximate_raw_margin(synthetic_frame) -> None:
    ds = make_dataset(synthetic_frame)
    model = fit_lightgbm(ds, seed=0, n_estimators=30)
    result = explain(model, ds.x)
    raw_margin = model.predict(ds.x, raw_score=True)
    reconstructed = result.values.sum(axis=1) + result.base_value
    # SHAP's additivity guarantee: contributions + base value reconstruct the
    # model's raw margin score for every row.
    np.testing.assert_allclose(reconstructed, raw_margin, atol=1e-4)


def test_local_explanation_returns_one_row_per_feature(synthetic_frame) -> None:
    ds = make_dataset(synthetic_frame)
    model = fit_lightgbm(ds, seed=0, n_estimators=30)
    result = explain(model, ds.x)
    local = result.local_explanation(0)
    assert len(local) == ds.x.shape[1]
    assert set(local["feature"]) == set(ds.x.columns)
