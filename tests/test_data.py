"""Unit tests for fraud_calibrated.data: cleaning, splitting, exposure."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fraud_calibrated.data import (
    RAW_URL,
    DatasetError,
    checksum_matches,
    clean,
    download,
    exposure,
    load,
    read_archive,
    split,
    stratum_split,
)


def _raw_frame(n: int = 20) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "ID": np.arange(1, n + 1),
            "LIMIT_BAL": rng.integers(10_000, 200_000, n),
            "SEX": rng.integers(1, 3, n),
            "EDUCATION": [0, 1, 2, 3, 4, 5, 6] * (n // 7) + [1] * (n % 7),
            "MARRIAGE": [0, 1, 2, 3] * (n // 4) + [1] * (n % 4),
            "AGE": rng.integers(21, 70, n),
            "PAY_0": rng.integers(-2, 5, n),
            "PAY_2": rng.integers(-2, 5, n),
            "PAY_3": rng.integers(-2, 5, n),
            "PAY_4": rng.integers(-2, 5, n),
            "PAY_5": rng.integers(-2, 5, n),
            "PAY_6": rng.integers(-2, 5, n),
            "BILL_AMT1": rng.integers(0, 100_000, n),
            "BILL_AMT2": rng.integers(0, 100_000, n),
            "BILL_AMT3": rng.integers(0, 100_000, n),
            "BILL_AMT4": rng.integers(0, 100_000, n),
            "BILL_AMT5": rng.integers(0, 100_000, n),
            "BILL_AMT6": rng.integers(0, 100_000, n),
            "PAY_AMT1": rng.integers(0, 20_000, n),
            "PAY_AMT2": rng.integers(0, 20_000, n),
            "PAY_AMT3": rng.integers(0, 20_000, n),
            "PAY_AMT4": rng.integers(0, 20_000, n),
            "PAY_AMT5": rng.integers(0, 20_000, n),
            "PAY_AMT6": rng.integers(0, 20_000, n),
            "default payment next month": rng.integers(0, 2, n),
        }
    )


def test_clean_renames_pay_0_and_target() -> None:
    df = clean(_raw_frame())
    assert "PAY_1" in df.columns
    assert "PAY_0" not in df.columns
    assert "default" in df.columns
    assert "default payment next month" not in df.columns


def test_clean_drops_id() -> None:
    df = clean(_raw_frame())
    assert "ID" not in df.columns


def test_clean_folds_undocumented_education_and_marriage_codes() -> None:
    df = clean(_raw_frame())
    assert set(df["EDUCATION"].unique()) <= {1, 2, 3, 4}
    assert set(df["MARRIAGE"].unique()) <= {1, 2, 3}


def test_clean_raises_on_unexpected_schema() -> None:
    bad = _raw_frame().drop(columns=["LIMIT_BAL"])
    with pytest.raises(DatasetError):
        clean(bad)


def test_checksum_matches_detects_mismatch(tmp_path) -> None:
    fake = tmp_path / "not_the_dataset.zip"
    fake.write_bytes(b"definitely not the real archive")
    assert checksum_matches(fake) is False


def test_exposure_is_nonnegative_and_capped_at_limit() -> None:
    df = pd.DataFrame(
        {
            "BILL_AMT1": [1000.0, -500.0, 50_000.0],
            "PAY_AMT1": [200.0, 0.0, 0.0],
            "LIMIT_BAL": [5000.0, 5000.0, 10_000.0],
        }
    )
    result = exposure(df)
    assert (result >= 0).all()
    assert result[2] == 10_000.0  # capped: 50,000 unpaid but only 10,000 extended
    assert result[1] == 0.0  # negative unpaid balance floors at zero


def test_split_is_stratified_and_covers_every_row() -> None:
    df = clean(_raw_frame(200))
    result = split(df, seed=1, valid_frac=0.2, test_frac=0.2)
    total = sum(result.sizes.values())
    assert total == len(df)
    for part in (result.train, result.valid, result.test):
        assert len(part) > 0


def test_split_is_deterministic_given_seed() -> None:
    df = clean(_raw_frame(200))
    a = split(df, seed=7)
    b = split(df, seed=7)
    pd.testing.assert_frame_equal(a.train.sort_index(axis=1), b.train.sort_index(axis=1))


def test_split_rejects_invalid_fractions() -> None:
    df = clean(_raw_frame(50))
    with pytest.raises(ValueError):
        split(df, valid_frac=0.6, test_frac=0.6)


def test_stratum_split_partitions_disjointly() -> None:
    df = clean(_raw_frame(100))
    in_domain, out_of_domain = stratum_split(df, "AGE", list(range(18, 31)))
    assert len(in_domain) + len(out_of_domain) == len(df)
    assert set(in_domain["AGE"]).issubset(set(range(18, 31)))
    assert not set(out_of_domain["AGE"]) & set(range(18, 31))


def test_stratum_split_rejects_unknown_column() -> None:
    df = clean(_raw_frame(10))
    with pytest.raises(ValueError):
        stratum_split(df, "NOT_A_COLUMN", [1])


def test_download_skips_network_when_file_present(tmp_path) -> None:
    existing = tmp_path / "uci350.zip"
    existing.write_bytes(b"already here")
    path = download(tmp_path)
    assert path == existing
    assert path.read_bytes() == b"already here"


def test_download_writes_response_body(tmp_path, monkeypatch) -> None:
    calls = {}

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b"fake archive bytes"

    def _fake_urlopen(url, timeout):
        calls["url"] = url
        calls["timeout"] = timeout
        return _FakeResponse()

    monkeypatch.setattr("fraud_calibrated.data.urlopen", _fake_urlopen)
    path = download(tmp_path / "sub")
    assert calls["url"] == RAW_URL
    assert path.read_bytes() == b"fake archive bytes"


def test_read_archive_rejects_a_non_zip_file(tmp_path) -> None:
    bogus = tmp_path / "not_a_zip.zip"
    bogus.write_bytes(b"plain text, not a zip")
    with pytest.raises(DatasetError):
        read_archive(bogus)


def test_load_rebuilds_from_the_committed_raw_archive(tmp_path) -> None:
    import shutil

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    repo_root = Path(__file__).resolve().parent.parent
    shutil.copy(repo_root / "data" / "raw" / "uci350.zip", raw_dir / "uci350.zip")
    df = load(cache_dir=tmp_path, force_rebuild=True)
    assert df.shape[0] == 30_000
    assert (tmp_path / "interim" / "credit_default.csv.gz").exists()


def test_load_uses_cache_on_second_call(tmp_path) -> None:
    import shutil

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    repo_root = Path(__file__).resolve().parent.parent
    shutil.copy(repo_root / "data" / "raw" / "uci350.zip", raw_dir / "uci350.zip")
    first = load(cache_dir=tmp_path)
    cache_path = tmp_path / "interim" / "credit_default.csv.gz"
    mtime_before = cache_path.stat().st_mtime
    second = load(cache_dir=tmp_path)
    assert cache_path.stat().st_mtime == mtime_before
    pd.testing.assert_frame_equal(first, second)
