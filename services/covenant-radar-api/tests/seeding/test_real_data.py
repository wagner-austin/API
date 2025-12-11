"""Tests for real bankruptcy data loaders and converters."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from covenant_radar_api.seeding.real_data import (
    RawDataset,
    RealDataSample,
    _parse_arff_file,
    _safe_float,
    get_dataset_stats,
    load_all_real_data,
    load_polish_arff,
    load_polish_raw,
    load_taiwan_data,
    load_taiwan_raw,
    load_us_data,
    load_us_raw,
    samples_to_arrays,
)


def _write_taiwan_dataset(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / "data.csv"
    content = (
        "Bankrupt?,Debt ratio %,Interest Coverage Ratio (Interest expense to EBIT),"
        "Current Ratio,Quick Ratio,ROA(A) before interest and % after tax,Net worth/Assets\n"
        "0,0.4,1.5,1.2,0.8,0.05,0.3\n"
        "1,0.8,0.2,0.5,0.4,0.01,0.1\n"
        "0,0.0,0.0,0.0,0.0,0.0,0.0\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


def _write_us_dataset(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / "american_bankruptcy.csv"
    content = (
        "company_name,status_label,year,X1,X2,X3,X4,X5\n"
        "acme,failed,2020,50,75,20,30,40\n"
        "betacorp,alive,2021,0,0,0,0,0\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


def _write_polish_arff(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / "1year.arff"
    numeric_attrs = "\n".join([f"@attribute Attr{i} numeric" for i in range(1, 65)])
    values = ",".join([str(0.1 * i) for i in range(1, 65)] + ["1"])
    content = "\n".join(
        [
            "@relation bankruptcy",
            numeric_attrs,
            "% comment line",
            "@data",
            "1,2,3",
            values,
        ]
    )
    path.write_text(content, encoding="utf-8")
    return path


def test_safe_float_handles_defaults_and_clipping() -> None:
    """_safe_float returns defaults and respects bounds."""
    assert _safe_float("nan", default=1.5) == 1.5
    assert _safe_float("not_a_number", default=2.0, min_val=0.5, max_val=3.0) == 2.0
    assert _safe_float("10", default=0.0, max_val=5.0) == 5.0
    assert _safe_float("-5", default=0.0, min_val=-2.0) == -2.0
    assert _safe_float("1e309", default=4.0) == 4.0


def test_load_taiwan_data_parses_samples(tmp_path: Path) -> None:
    """load_taiwan_data converts Taiwan CSV into RealDataSample list."""
    data_path = _write_taiwan_dataset(tmp_path / "taiwan_data")
    samples = load_taiwan_data(data_path)

    assert len(samples) == 3
    first = samples[0]
    assert first["company_id"] == "taiwan_0"
    assert first["debt_ratio"] == 0.4
    assert first["interest_coverage"] == 1.5
    assert first["is_bankrupt"] == 0
    assert samples[-1]["debt_to_ebitda"] == 0.0


def test_read_csv_handles_missing_columns(tmp_path: Path) -> None:
    """_read_csv_rows should skip missing values without raising."""
    path = tmp_path / "taiwan_data" / "data.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "A,B,C\n"  # headers
        "1,2\n",  # missing last column
        encoding="utf-8",
    )

    samples = load_taiwan_data(path)
    assert len(samples) == 1
    sample = samples[0]
    assert sample["debt_ratio"] == 0.0
    assert sample["is_bankrupt"] == 0


def test_load_us_data_parses_samples(tmp_path: Path) -> None:
    """load_us_data maps US dataset columns to sample fields."""
    data_path = _write_us_dataset(tmp_path / "us_data")
    samples = load_us_data(data_path)

    assert len(samples) == 2
    failed = samples[0]
    assert failed["company_id"] == "acme"
    assert failed["is_bankrupt"] == 1
    assert failed["interest_coverage"] > 0
    healthy = samples[1]
    assert healthy["is_bankrupt"] == 0


def test_load_polish_arff_parses_samples(tmp_path: Path) -> None:
    """load_polish_arff parses ARFF lines and builds samples."""
    data_path = _write_polish_arff(tmp_path / "polish_data")
    samples = load_polish_arff(data_path)

    assert len(samples) == 1
    sample = samples[0]
    assert sample["company_id"] == "polish_0"
    assert sample["is_bankrupt"] == 1
    assert sample["debt_ratio"] >= 0.0


def test_load_all_real_data_handles_missing_and_present(tmp_path: Path) -> None:
    """load_all_real_data skips missing files and aggregates when present."""
    empty_result = load_all_real_data(tmp_path)
    assert empty_result == []

    taiwan_dir = tmp_path / "taiwan_data"
    us_dir = tmp_path / "us_data"
    polish_dir = tmp_path / "polish_data"
    _write_taiwan_dataset(taiwan_dir)
    _write_us_dataset(us_dir)
    _write_polish_arff(polish_dir)

    combined = load_all_real_data(tmp_path)
    assert len(combined) == 6
    ids = {s["company_id"] for s in combined}
    assert "taiwan_0" in ids and "acme" in ids and "polish_0" in ids


def test_samples_to_arrays_and_stats() -> None:
    """samples_to_arrays returns model-ready arrays and stats handle empty data."""
    samples: list[RealDataSample] = [
        RealDataSample(
            company_id="s1",
            debt_to_ebitda=3.0,
            interest_coverage=2.5,
            current_ratio=1.5,
            debt_ratio=0.6,
            quick_ratio=1.0,
            roa=0.05,
            net_worth_ratio=0.4,
            is_bankrupt=1,
        ),
        RealDataSample(
            company_id="s2",
            debt_to_ebitda=1.0,
            interest_coverage=4.0,
            current_ratio=2.0,
            debt_ratio=0.3,
            quick_ratio=1.5,
            roa=0.1,
            net_worth_ratio=0.6,
            is_bankrupt=0,
        ),
    ]
    x_array, y_array = samples_to_arrays(samples)
    assert x_array.shape == (2, 8)
    assert x_array.dtype == np.float64
    assert y_array.shape == (2,)
    assert y_array.dtype == np.int64
    expected_y: NDArray[np.int64] = np.zeros(2, dtype=np.int64)
    expected_y[0] = np.int64(1)
    expected_y[1] = np.int64(0)
    assert bool(np.array_equal(y_array, expected_y))

    stats = get_dataset_stats(samples)
    assert stats["n_total"] == 2
    assert stats["n_bankrupt"] == 1
    assert stats["bankruptcy_rate"] == 0.5

    empty_stats = get_dataset_stats([])
    assert empty_stats["bankruptcy_rate"] == 0.0


def _write_taiwan_raw_dataset(base_dir: Path) -> Path:
    """Write a Taiwan raw dataset CSV for testing load_taiwan_raw."""
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / "data.csv"
    # Taiwan format: first column is label "Bankrupt?", rest are feature columns
    content = " Bankrupt?, Feat1, Feat2, Feat3\n0,1.5,2.5,3.5\n1,4.5,5.5,6.5\n0,0.0,0.0,0.0\n"
    path.write_text(content, encoding="utf-8")
    return path


def _write_us_raw_dataset(base_dir: Path) -> Path:
    """Write a US raw dataset CSV for testing load_us_raw."""
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / "american_bankruptcy.csv"
    # US format: company_name, status_label, year, X1, X2, ..., X18
    content = (
        "company_name,status_label,year,X1,X2,X3,X4,X5,X6,X7,X8,X9,"
        "X10,X11,X12,X13,X14,X15,X16,X17,X18\n"
        "acme,failed,2020,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,"
        "10.0,11.0,12.0,13.0,14.0,15.0,16.0,17.0,18.0\n"
        "beta,alive,2021,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,"
        "9.5,10.5,11.5,12.5,13.5,14.5,15.5,16.5,17.5\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


def _write_polish_raw_arff(base_dir: Path) -> Path:
    """Write a Polish raw ARFF dataset for testing load_polish_raw."""
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / "1year.arff"
    # Build 64 attribute declarations
    attrs = "\n".join([f"@attribute Attr{i} numeric" for i in range(1, 65)])
    # Build data rows with 64 features + class
    row1_features = ",".join([f"{0.1 * i:.1f}" for i in range(1, 65)])
    row2_features = ",".join([f"{0.2 * i:.1f}" for i in range(1, 65)])
    content = "\n".join(
        [
            "@relation polish_bankruptcy",
            attrs,
            "@attribute class {0,1}",
            "",
            "% This is a comment",
            "@data",
            f"{row1_features},1",  # bankrupt
            f"{row2_features},0",  # healthy
        ]
    )
    path.write_text(content, encoding="utf-8")
    return path


def test_load_taiwan_raw_parses_all_columns(tmp_path: Path) -> None:
    """load_taiwan_raw loads all feature columns from Taiwan CSV."""
    data_path = _write_taiwan_raw_dataset(tmp_path / "taiwan_data")
    result: RawDataset = load_taiwan_raw(data_path)

    assert result["n_samples"] == 3
    assert result["n_features"] == 3
    assert result["feature_names"] == ["Feat1", "Feat2", "Feat3"]
    assert result["n_bankrupt"] == 1
    assert result["n_healthy"] == 2

    # Verify X array shape and values
    x = result["x"]
    assert x.shape == (3, 3)
    assert x.dtype == np.float64
    # Build expected array avoiding Any type issues
    expected_x: NDArray[np.float64] = np.zeros((3, 3), dtype=np.float64)
    expected_x[0, :] = (1.5, 2.5, 3.5)
    expected_x[1, :] = (4.5, 5.5, 6.5)
    expected_x[2, :] = (0.0, 0.0, 0.0)
    assert bool(np.allclose(x, expected_x))

    # Verify y array
    y = result["y"]
    assert y.shape == (3,)
    assert y.dtype == np.int64
    expected_y: NDArray[np.int64] = np.zeros(3, dtype=np.int64)
    expected_y[0] = 0
    expected_y[1] = 1
    expected_y[2] = 0
    assert bool(np.array_equal(y, expected_y))


def test_load_taiwan_raw_raises_on_empty_file(tmp_path: Path) -> None:
    """load_taiwan_raw raises ValueError for empty CSV."""
    path = tmp_path / "empty.csv"
    path.write_text("Bankrupt?,Feat1\n", encoding="utf-8")  # Headers only

    with pytest.raises(ValueError, match="No data rows found"):
        load_taiwan_raw(path)


def test_load_us_raw_parses_all_columns(tmp_path: Path) -> None:
    """load_us_raw loads all X1-X18 feature columns from US CSV."""
    data_path = _write_us_raw_dataset(tmp_path / "us_data")
    result: RawDataset = load_us_raw(data_path)

    assert result["n_samples"] == 2
    assert result["n_features"] == 18
    assert result["feature_names"][0] == "X1"
    assert result["feature_names"][-1] == "X18"
    assert result["n_bankrupt"] == 1
    assert result["n_healthy"] == 1

    # Verify X array
    x = result["x"]
    assert x.shape == (2, 18)
    # Build expected first row: X1=1.0 to X18=18.0
    expected_x: NDArray[np.float64] = np.zeros((2, 18), dtype=np.float64)
    for i in range(18):
        expected_x[0, i] = float(i + 1)
        expected_x[1, i] = float(i) + 0.5
    assert bool(np.allclose(x, expected_x))

    # Verify y array (failed=1, alive=0)
    y = result["y"]
    expected_y: NDArray[np.int64] = np.zeros(2, dtype=np.int64)
    expected_y[0] = 1
    expected_y[1] = 0
    assert bool(np.array_equal(y, expected_y))


def test_load_us_raw_raises_on_empty_file(tmp_path: Path) -> None:
    """load_us_raw raises ValueError for empty CSV."""
    path = tmp_path / "empty.csv"
    path.write_text("company_name,status_label,year,X1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="No data rows found"):
        load_us_raw(path)


def test_parse_arff_file_extracts_features_and_data(tmp_path: Path) -> None:
    """_parse_arff_file correctly parses ARFF format."""
    data_path = _write_polish_raw_arff(tmp_path / "polish_data")
    result = _parse_arff_file(data_path)

    assert len(result["feature_names"]) == 64
    assert result["feature_names"][0] == "Attr1"
    assert result["feature_names"][-1] == "Attr64"
    assert len(result["data_rows"]) == 2


def test_parse_arff_file_raises_on_empty_data(tmp_path: Path) -> None:
    """_parse_arff_file raises ValueError when no data rows present."""
    path = tmp_path / "empty.arff"
    content = "\n".join(
        [
            "@relation test",
            "@attribute Attr1 numeric",
            "@attribute class {0,1}",
            "@data",
            # No data rows
        ]
    )
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="No data rows found"):
        _parse_arff_file(path)


def test_parse_arff_file_skips_malformed_attribute_lines(tmp_path: Path) -> None:
    """_parse_arff_file skips @attribute lines with less than 2 parts."""
    path = tmp_path / "malformed.arff"
    content = "\n".join(
        [
            "@relation test",
            "@attribute",  # Malformed - no name
            "@attribute Attr1 numeric",  # Valid
            "@attribute    ",  # Malformed - just whitespace after keyword
            "@attribute Attr2 numeric",  # Valid
            "@attribute class {0,1}",
            "@data",
            "1.0,2.0,1",  # Valid row
        ]
    )
    path.write_text(content, encoding="utf-8")

    result = _parse_arff_file(path)
    # Only valid attributes should be captured (Attr1 and Attr2)
    assert result["feature_names"] == ["Attr1", "Attr2"]
    assert len(result["data_rows"]) == 1


def test_parse_arff_file_skips_short_data_rows(tmp_path: Path) -> None:
    """_parse_arff_file skips data rows with insufficient values."""
    path = tmp_path / "short_rows.arff"
    content = "\n".join(
        [
            "@relation test",
            "@attribute Attr1 numeric",
            "@attribute Attr2 numeric",
            "@attribute class {0,1}",
            "@data",
            "1.0,2.0,1",  # Valid: 3 values (2 features + class)
            "1.0",  # Invalid: only 1 value (needs 3)
            "1.0,2.0",  # Invalid: only 2 values (needs 3)
            "3.0,4.0,0",  # Valid
        ]
    )
    path.write_text(content, encoding="utf-8")

    result = _parse_arff_file(path)
    assert result["feature_names"] == ["Attr1", "Attr2"]
    # Only the two valid rows should be captured
    assert len(result["data_rows"]) == 2
    assert result["data_rows"][0] == ["1.0", "2.0", "1"]
    assert result["data_rows"][1] == ["3.0", "4.0", "0"]


def test_load_polish_raw_parses_all_columns(tmp_path: Path) -> None:
    """load_polish_raw loads all 64 attributes from Polish ARFF."""
    data_path = _write_polish_raw_arff(tmp_path / "polish_data")
    result: RawDataset = load_polish_raw(data_path)

    assert result["n_samples"] == 2
    assert result["n_features"] == 64
    assert result["feature_names"][0] == "Attr1"
    assert result["feature_names"][-1] == "Attr64"
    assert result["n_bankrupt"] == 1
    assert result["n_healthy"] == 1

    # Verify X array
    x = result["x"]
    assert x.shape == (2, 64)
    # Build expected array - Row 1: 0.1*i for i=1..64, Row 2: 0.2*i for i=1..64
    expected_x: NDArray[np.float64] = np.zeros((2, 64), dtype=np.float64)
    for i in range(64):
        expected_x[0, i] = 0.1 * (i + 1)
        expected_x[1, i] = 0.2 * (i + 1)
    assert bool(np.allclose(x, expected_x))

    # Verify y array
    y = result["y"]
    expected_y: NDArray[np.int64] = np.zeros(2, dtype=np.int64)
    expected_y[0] = 1
    expected_y[1] = 0
    assert bool(np.array_equal(y, expected_y))
