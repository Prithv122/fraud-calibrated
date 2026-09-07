"""CLI smoke tests: each subcommand runs and prints something parseable."""

from __future__ import annotations

import json

import pytest

from fraud_calibrated.cli import build_parser, main


def test_report_subcommand_prints_valid_json(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--seed", "0", "report"])
    assert exc.value.code == 0
    payload = json.loads(capsys.readouterr().out)
    assert "auc" in payload
    assert "cost" in payload


def test_drift_subcommand_runs(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--seed", "0", "drift"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Feature PSI" in out


def test_explain_subcommand_runs_with_and_without_row(capsys) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--seed", "0", "explain", "--row", "0"])
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "Global importance" in out
    assert "Local explanation" in out


def test_sensitivity_subcommand_prints_a_json_list(monkeypatch, capsys, synthetic_frame) -> None:
    # Runs the pipeline 17 times over a cost-parameter grid; swap in the tiny
    # synthetic fixture so this stays a fast unit test rather than a 17x repeat
    # of the full 30,000-row pipeline.
    monkeypatch.setattr("fraud_calibrated.cli._load", lambda args: synthetic_frame)
    with pytest.raises(SystemExit) as exc:
        main(["--seed", "0", "sensitivity"])
    assert exc.value.code == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) > 0
    assert {"varied", "value", "total_cost", "n_flagged"} <= rows[0].keys()


def test_parser_requires_a_subcommand() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
