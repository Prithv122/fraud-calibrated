"""Shared fixtures.

``synthetic_frame`` builds a small but schema-correct frame entirely in memory --
no network, no file I/O -- for the majority of unit tests. ``real_df`` loads the
dataset from the cache committed at ``data/interim/credit_default.csv.gz`` (see
:mod:`fraud_calibrated.data`) and is used only by the handful of tests in
``test_pipeline_integration.py`` that check the pipeline against real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fraud_calibrated.data import CATEGORICAL_COLS, NUMERIC_COLS, TARGET


@pytest.fixture
def synthetic_frame() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 400
    data = {
        "LIMIT_BAL": rng.integers(10_000, 500_000, n),
        "AGE": rng.integers(21, 70, n),
        "SEX": rng.integers(1, 3, n),
        "EDUCATION": rng.integers(1, 5, n),
        "MARRIAGE": rng.integers(1, 4, n),
    }
    for col in NUMERIC_COLS:
        if col in data:
            continue
        if col.startswith("PAY_") and not col.startswith("PAY_AMT"):
            data[col] = rng.integers(-2, 5, n)
        elif col.startswith("BILL_AMT"):
            data[col] = rng.integers(0, 200_000, n)
        elif col.startswith("PAY_AMT"):
            data[col] = rng.integers(0, 50_000, n)
    df = pd.DataFrame(data)
    # A logistic signal in LIMIT_BAL and PAY_1 so the model has something real to
    # learn, rather than pure noise that would make AUC checks meaningless.
    logit = -1.5 - df["LIMIT_BAL"] / 200_000 + df["PAY_1"].clip(lower=0) * 0.6
    prob = 1 / (1 + np.exp(-logit))
    df[TARGET] = (rng.random(n) < prob).astype(int)
    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype(int)
    return df


@pytest.fixture(scope="session")
def real_df() -> pd.DataFrame:
    from fraud_calibrated.data import load

    return load(cache_dir="data")
