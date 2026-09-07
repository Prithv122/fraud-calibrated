"""Unit tests for fraud_calibrated.model, on the synthetic fixture."""

from __future__ import annotations

import numpy as np
import pytest

from fraud_calibrated.model import (
    fit_baseline,
    fit_lightgbm,
    make_dataset,
    predict_baseline,
    predict_proba,
    scale_pos_weight,
)


def test_scale_pos_weight_matches_manual_ratio() -> None:
    y = np.array([1, 0, 0, 0, 1, 0, 0, 0])
    assert scale_pos_weight(y) == pytest.approx(6 / 2)


def test_scale_pos_weight_rejects_no_positives() -> None:
    with pytest.raises(ValueError):
        scale_pos_weight(np.zeros(10))


def test_make_dataset_sets_categorical_dtype(synthetic_frame) -> None:
    ds = make_dataset(synthetic_frame)
    assert str(ds.x["SEX"].dtype) == "category"
    assert len(ds.y) == len(synthetic_frame)


def test_lightgbm_predictions_are_probabilities(synthetic_frame) -> None:
    ds = make_dataset(synthetic_frame)
    model = fit_lightgbm(ds, seed=0, n_estimators=20)
    prob = predict_proba(model, ds)
    assert prob.shape == ds.y.shape
    assert np.all((prob >= 0) & (prob <= 1))


def test_lightgbm_beats_chance_on_a_learnable_signal(synthetic_frame) -> None:
    from sklearn.metrics import roc_auc_score

    n = len(synthetic_frame)
    train, test = synthetic_frame.iloc[: n // 2], synthetic_frame.iloc[n // 2 :]
    train_ds, test_ds = make_dataset(train), make_dataset(test)
    model = fit_lightgbm(train_ds, seed=0, n_estimators=50)
    prob = predict_proba(model, test_ds)
    assert roc_auc_score(test_ds.y, prob) > 0.6


def test_baseline_predictions_are_probabilities(synthetic_frame) -> None:
    ds = make_dataset(synthetic_frame)
    model = fit_baseline(ds, seed=0)
    prob = predict_baseline(model, ds)
    assert prob.shape == ds.y.shape
    assert np.all((prob >= 0) & (prob <= 1))


def test_baseline_handles_unseen_categories_at_inference(synthetic_frame) -> None:
    train_ds = make_dataset(synthetic_frame)
    model = fit_baseline(train_ds, seed=0)
    novel = synthetic_frame.copy()
    novel["EDUCATION"] = 99  # a category absent from training
    novel_ds = make_dataset(novel)
    prob = predict_baseline(model, novel_ds)
    assert np.all(np.isfinite(prob))
