"""End-to-end orchestration: load -> split -> train -> calibrate -> threshold -> score.

Kept separate from ``cli.py`` so tests can call :func:`run` directly on a tiny
synthetic frame without going through argument parsing or stdout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud_calibrated import calibration, costs, model
from fraud_calibrated.data import Split, exposure
from fraud_calibrated.data import split as split_fn


@dataclass
class PipelineResult:
    split: Split
    lgbm: object
    baseline: object
    train_ds: model.Dataset
    valid_ds: model.Dataset
    test_ds: model.Dataset
    valid_prob_raw: np.ndarray
    test_prob_raw: np.ndarray
    calibrator: object
    valid_prob_calibrated: np.ndarray
    test_prob_calibrated: np.ndarray
    baseline_test_prob: np.ndarray
    calibration_before: calibration.CalibrationReport
    calibration_after: calibration.CalibrationReport
    cost_model: costs.CostModel
    test_exposure: np.ndarray
    metrics: dict = field(default_factory=dict)


def run(
    df: pd.DataFrame,
    *,
    cost_model: costs.CostModel | None = None,
    seed: int = 0,
) -> PipelineResult:
    cost_model = cost_model or costs.CostModel()
    split = split_fn(df, seed=seed)
    train_ds = model.make_dataset(split.train)
    valid_ds = model.make_dataset(split.valid)
    test_ds = model.make_dataset(split.test)

    lgbm = model.fit_lightgbm(train_ds, seed=seed)
    baseline = model.fit_baseline(train_ds, seed=seed)

    valid_prob_raw = model.predict_proba(lgbm, valid_ds)
    test_prob_raw = model.predict_proba(lgbm, test_ds)
    baseline_test_prob = model.predict_baseline(baseline, test_ds)

    calibration_before = calibration.report(test_ds.y, test_prob_raw)

    calibrator = calibration.fit_isotonic(valid_ds.y, valid_prob_raw)
    valid_prob_calibrated = calibration.apply_isotonic(calibrator, valid_prob_raw)
    test_prob_calibrated = calibration.apply_isotonic(calibrator, test_prob_raw)

    calibration_after = calibration.report(test_ds.y, test_prob_calibrated)

    test_exposure = exposure(split.test)
    flagged = costs.decide(test_prob_calibrated, test_exposure, cost_model)
    cost_calibrated = costs.total_cost(test_ds.y, flagged, test_exposure, cost_model)

    flagged_raw = costs.decide(test_prob_raw, test_exposure, cost_model)
    cost_raw = costs.total_cost(test_ds.y, flagged_raw, test_exposure, cost_model)

    thresholds = np.linspace(0.01, 0.99, 99)
    global_costs = costs.global_threshold_cost(
        test_ds.y, test_prob_calibrated, test_exposure, cost_model, thresholds
    )
    best_global_idx = int(np.argmin(global_costs))

    metrics = {
        "n_train": len(train_ds.y),
        "n_valid": len(valid_ds.y),
        "n_test": len(test_ds.y),
        "positive_rate": {
            "train": float(train_ds.y.mean()),
            "valid": float(valid_ds.y.mean()),
            "test": float(test_ds.y.mean()),
        },
        "auc": {
            "lightgbm": float(roc_auc_score(test_ds.y, test_prob_calibrated)),
            "baseline": float(roc_auc_score(test_ds.y, baseline_test_prob)),
        },
        "pr_auc": {
            "lightgbm": float(average_precision_score(test_ds.y, test_prob_calibrated)),
            "baseline": float(average_precision_score(test_ds.y, baseline_test_prob)),
        },
        "brier": {"before": calibration_before.brier, "after": calibration_after.brier},
        "ece": {"before": calibration_before.ece, "after": calibration_after.ece},
        "cost": {
            "per_row_threshold": cost_calibrated,
            "raw_uncalibrated_per_row_threshold": cost_raw,
            "best_global_threshold": float(global_costs[best_global_idx]),
            "best_global_threshold_value": float(thresholds[best_global_idx]),
            "n_flagged_per_row": int(flagged.sum()),
        },
    }

    return PipelineResult(
        split=split,
        lgbm=lgbm,
        baseline=baseline,
        train_ds=train_ds,
        valid_ds=valid_ds,
        test_ds=test_ds,
        valid_prob_raw=valid_prob_raw,
        test_prob_raw=test_prob_raw,
        calibrator=calibrator,
        valid_prob_calibrated=valid_prob_calibrated,
        test_prob_calibrated=test_prob_calibrated,
        baseline_test_prob=baseline_test_prob,
        calibration_before=calibration_before,
        calibration_after=calibration_after,
        cost_model=cost_model,
        test_exposure=test_exposure,
        metrics=metrics,
    )
