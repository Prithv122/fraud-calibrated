"""SHAP explanations for the LightGBM model.

Wraps ``shap.TreeExplainer`` rather than the model-agnostic ``KernelExplainer``: tree
SHAP is exact and fast for gradient-boosted trees, so there is no approximation
tradeoff to disclose, and it comfortably handles the categorical columns LightGBM
consumes natively.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier


@dataclass(frozen=True)
class ShapResult:
    values: np.ndarray  # (n_rows, n_features)
    base_value: float
    feature_names: list[str]

    def global_importance(self) -> pd.DataFrame:
        """Mean |SHAP value| per feature, sorted descending -- the global-importance
        table for the README."""
        mean_abs = np.abs(self.values).mean(axis=0)
        return (
            pd.DataFrame({"feature": self.feature_names, "mean_abs_shap": mean_abs})
            .sort_values("mean_abs_shap", ascending=False)
            .reset_index(drop=True)
        )

    def local_explanation(self, row_idx: int) -> pd.DataFrame:
        """Per-feature contribution for one customer, sorted by |contribution|."""
        contributions = self.values[row_idx]
        return (
            pd.DataFrame({"feature": self.feature_names, "shap_value": contributions})
            .assign(abs_shap=lambda d: d["shap_value"].abs())
            .sort_values("abs_shap", ascending=False)
            .drop(columns="abs_shap")
            .reset_index(drop=True)
        )


def explain(model: LGBMClassifier, x: pd.DataFrame) -> ShapResult:
    """Compute SHAP values for every row of ``x`` against the positive class."""
    explainer = shap.TreeExplainer(model)
    raw = explainer(x)
    values = raw.values
    base = raw.base_values
    # Binary LightGBM via the sklearn API returns shape (n, features, 2) classes on
    # some shap/lightgbm version combinations, and (n, features) on others -- pin to
    # the positive-class slice either way rather than assuming one shape.
    if values.ndim == 3:
        values = values[:, :, 1]
        base = base[:, 1] if np.ndim(base) > 0 and np.shape(base)[-1] == 2 else base
    base_value = float(np.mean(base))
    return ShapResult(values=values, base_value=base_value, feature_names=list(x.columns))
