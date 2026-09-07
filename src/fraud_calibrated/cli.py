"""Command-line entry point: `fraud-calibrated <subcommand>`.

Every number in the README traces back to one of these subcommands, run with a fixed
seed, so the results are reproducible from a clean clone (see README section 6).
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from fraud_calibrated import data, drift
from fraud_calibrated.costs import CostModel
from fraud_calibrated.explain import explain
from fraud_calibrated.pipeline import run


def _load(args: argparse.Namespace) -> pd.DataFrame:
    return data.load(cache_dir=args.data_dir, force_rebuild=args.force_rebuild)


def cmd_report(args: argparse.Namespace) -> int:
    df = _load(args)
    cost_model = CostModel(
        lgd=args.lgd, intervention_cost=args.intervention_cost, margin_rate=args.margin_rate
    )
    result = run(df, cost_model=cost_model, seed=args.seed)
    print(json.dumps(result.metrics, indent=2))
    return 0


def cmd_sensitivity(args: argparse.Namespace) -> int:
    """How the per-row-optimal-threshold total cost moves as each cost-model
    assumption varies, holding the other two at their default. This is the table
    that justifies calling LGD/intervention-cost/margin-rate "assumptions" rather
    than measurements in the README.
    """
    df = _load(args)
    base = CostModel()
    rows = []
    grids = {
        "lgd": np.round(np.linspace(0.4, 0.95, 6), 2),
        "intervention_cost": np.array([100, 250, 500, 1000, 2000], dtype=float),
        "margin_rate": np.round(np.linspace(0.01, 0.15, 6), 3),
    }
    for param, grid in grids.items():
        for value in grid:
            kwargs = {
                "lgd": base.lgd,
                "intervention_cost": base.intervention_cost,
                "margin_rate": base.margin_rate,
            }
            kwargs[param] = float(value)
            result = run(df, cost_model=CostModel(**kwargs), seed=args.seed)
            rows.append(
                {
                    "varied": param,
                    "value": float(value),
                    "total_cost": result.metrics["cost"]["per_row_threshold"],
                    "n_flagged": result.metrics["cost"]["n_flagged_per_row"],
                }
            )
    print(json.dumps(rows, indent=2))
    return 0


def cmd_explain(args: argparse.Namespace) -> int:
    df = _load(args)
    result = run(df, seed=args.seed)
    shap_result = explain(result.lgbm, result.test_ds.x)
    print("Global importance (mean |SHAP|):")
    print(shap_result.global_importance().to_string(index=False))
    if args.row is not None:
        print(f"\nLocal explanation for test row {args.row}:")
        print(shap_result.local_explanation(args.row).to_string(index=False))
    return 0


def cmd_drift(args: argparse.Namespace) -> int:
    """Measured (not synthetic) covariate shift: young vs. older customers.

    AGE <= 30 is a real subpopulation the source data already contains, not an
    injected perturbation -- see fraud_calibrated.drift's module docstring.
    """
    df = _load(args)
    result = run(df, seed=args.seed)
    in_domain, out_of_domain = data.stratum_split(result.split.test, "AGE", list(range(18, 31)))
    if len(in_domain) == 0 or len(out_of_domain) == 0:
        print("stratum split produced an empty slice; adjust the AGE cutoff", file=sys.stderr)
        return 1

    psi_table = drift.feature_psi_table(in_domain, out_of_domain, data.NUMERIC_COLS)

    from fraud_calibrated.model import make_dataset, predict_proba

    in_ds, out_ds = make_dataset(in_domain), make_dataset(out_of_domain)
    in_prob = predict_proba(result.lgbm, in_ds)
    out_prob = predict_proba(result.lgbm, out_ds)
    perf = drift.performance_drop(in_ds.y, in_prob, out_ds.y, out_prob)

    print("Feature PSI, young (<=30) as reference vs. older customers:")
    print(psi_table.to_string(index=False))
    print("\nPerformance on the two slices (same fixed model):")
    print(json.dumps(perf.__dict__, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fraud-calibrated")
    parser.add_argument("--data-dir", default="data", help="cache directory for the dataset")
    parser.add_argument("--force-rebuild", action="store_true", help="ignore any cached CSV")
    parser.add_argument("--seed", type=int, default=0)
    sub = parser.add_subparsers(dest="command", required=True)

    p_report = sub.add_parser("report", help="train, calibrate, threshold, and print metrics")
    p_report.add_argument("--lgd", type=float, default=0.75)
    p_report.add_argument("--intervention-cost", type=float, default=500.0)
    p_report.add_argument("--margin-rate", type=float, default=0.05)
    p_report.set_defaults(func=cmd_report)

    p_sens = sub.add_parser("sensitivity", help="cost-model sensitivity table")
    p_sens.set_defaults(func=cmd_sensitivity)

    p_explain = sub.add_parser("explain", help="SHAP global importance and one local example")
    p_explain.add_argument("--row", type=int, default=None, help="test-set row index to explain")
    p_explain.set_defaults(func=cmd_explain)

    p_drift = sub.add_parser("drift", help="PSI and performance drop, young vs. older customers")
    p_drift.set_defaults(func=cmd_drift)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    sys.exit(args.func(args))
