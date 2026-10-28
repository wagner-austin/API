"""Real extract bytes, timing boundaries and refused provenance changes."""

from __future__ import annotations

import hashlib
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest
from platform_core.json_utils import JSONObject, JSONValue, dump_json_str

from covenant_ml.evaluation.wrds import (
    decode_wrds_extract,
    encode_wrds_extract,
    load_wrds_evaluation,
)


def write_extract(directory: Path, csv_text: str | None = None) -> tuple[Path, JSONObject]:
    """Write synthetic CRSP-shaped evidence, never labeled empirical.

    Args:
        directory: Test-only directory.
        csv_text: Explicit malformed data or generated dated returns.

    Returns:
        Manifest path and its JSON fields.
    """
    if csv_text is None:
        start = date(2020, 1, 1)
        csv_text = "permno,date,ret\n" + "".join(
            f"10001,{start + timedelta(days=index)},{(index % 7 - 3) / 100}\n"
            for index in range(160)
        )
    csv_bytes = csv_text.encode()
    query = b"select permno, date, ret from crsp.dsf where permno=10001 order by date;"
    (directory / "synthetic.csv").write_bytes(csv_bytes)
    (directory / "extract.sql").write_bytes(query)
    document: JSONObject = {
        "source_table": "crsp.dsf",
        "security_id": 10001,
        "extracted_at": "2021-01-01T00:00:00+00:00",
        "csv_file": "synthetic.csv",
        "csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "query_file": "extract.sql",
        "query_sha256": hashlib.sha256(query).hexdigest(),
        "train_fraction": 0.6,
        "validation_fraction": 0.2,
        "embargo": 1,
    }
    manifest = directory / "synthetic.json"
    manifest.write_text(dump_json_str(document), encoding="utf-8")
    return manifest, document


def test_round_trip_and_exact_chronological_rows(tmp_path: Path) -> None:
    path, document = write_extract(tmp_path)
    assert encode_wrds_extract(decode_wrds_extract(document)) == document
    loaded = load_wrds_evaluation(path)
    grouped = loaded.splits(99)
    # 159 labels; training keys [0,95), validation [96,127), test [128,159).
    assert grouped.split.x_train.shape == (95, 1)
    assert grouped.split.x_val.shape == (31, 1)
    assert grouped.split.x_test.shape == (31, 1)
    np.testing.assert_array_equal(
        grouped.split.x_train[:, 0], [(i % 7 - 3) / 100 for i in range(95)]
    )
    np.testing.assert_array_equal(grouped.split.y_train, [(i % 7 - 3) > 0 for i in range(1, 96)])
    np.testing.assert_array_equal(
        grouped.test_groups,
        np.arange(date(2020, 5, 8).toordinal(), date(2020, 6, 8).toordinal(), dtype=np.int64),
    )
    assert loaded.info == {
        "sha256": hashlib.sha256(dump_json_str(document).encode()).hexdigest(),
        "n_rows": 159,
        "n_features": 1,
    }
    np.testing.assert_array_equal(loaded.splits(1).split.x_test, grouped.split.x_test)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_table", "comp.funda"),
        ("security_id", 0),
        ("extracted_at", "2021-01-01T00:00:00"),
        ("train_fraction", 0.0),
        ("train_fraction", 1.0),
        ("validation_fraction", 0.0),
        ("validation_fraction", 1.0),
        ("validation_fraction", 0.5),
        ("embargo", 0),
        ("csv_sha256", "x"),
        ("query_sha256", "g" * 64),
        ("csv_file", ""),
        ("csv_file", ".."),
        ("csv_file", "../elsewhere.csv"),
        ("query_file", "a\\b"),
    ],
)
def test_invalid_manifest_is_refused(tmp_path: Path, field: str, value: JSONValue) -> None:
    _, document = write_extract(tmp_path)
    document[field] = value
    with pytest.raises(ValueError, match="WRDS_DATA_"):
        decode_wrds_extract(document)


def test_unknown_manifest_field_is_refused(tmp_path: Path) -> None:
    _, document = write_extract(tmp_path)
    document["invented"] = True
    with pytest.raises(ValueError, match="WRDS_DATA_MANIFEST"):
        decode_wrds_extract(document)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "date,permno,ret\n",
        "permno,date,ret\n10001,2020-01-01\n",
        "permno,date,ret\n10002,2020-01-01,0.1\n",
        "permno,date,ret\n10001,2020-01-01,nan\n",
        "permno,date,ret\n10001,2020-01-01,-1.1\n",
        "permno,date,ret\n10001,2021-01-01,0.1\n",
        "permno,date,ret\n10001,2020-01-01,0.1\n10001,2020-01-01,0.2\n",
        "permno,date,ret\n10001,2020-01-02,0.1\n10001,2020-01-01,0.2\n",
        "permno,date,ret\n10001,2020-01-01,0.1\n",
    ],
)
def test_bad_rows_never_become_a_training_set(tmp_path: Path, text: str) -> None:
    path, _ = write_extract(tmp_path, text)
    with pytest.raises(ValueError, match="WRDS_DATA_ROWS"):
        load_wrds_evaluation(path)


@pytest.mark.parametrize("value", ["", "B", "not-a-return"])
def test_missing_and_special_crsp_returns_are_not_imputed(tmp_path: Path, value: str) -> None:
    path, _ = write_extract(tmp_path, f"permno,date,ret\n10001,2020-01-01,{value}\n")
    with pytest.raises(ValueError, match="could not convert string to float"):
        load_wrds_evaluation(path)


@pytest.mark.parametrize("name", ["synthetic.csv", "extract.sql"])
def test_modified_evidence_is_refused(tmp_path: Path, name: str) -> None:
    path, _ = write_extract(tmp_path)
    (tmp_path / name).write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="WRDS_DATA_DRIFT"):
        load_wrds_evaluation(path)


def test_missing_query_is_not_replaced(tmp_path: Path) -> None:
    path, _ = write_extract(tmp_path)
    (tmp_path / "extract.sql").unlink()
    with pytest.raises(FileNotFoundError):
        load_wrds_evaluation(path)


def test_empty_but_hash_matched_query_is_refused(tmp_path: Path) -> None:
    path, document = write_extract(tmp_path)
    (tmp_path / "extract.sql").write_bytes(b"")
    document["query_sha256"] = hashlib.sha256(b"").hexdigest()
    path.write_text(dump_json_str(document), encoding="utf-8")
    with pytest.raises(ValueError, match="WRDS_DATA_ROWS"):
        load_wrds_evaluation(path)


def test_evidence_cannot_escape_through_a_symlink(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    path, _ = write_extract(nested)
    source = nested / "synthetic.csv"
    outside = tmp_path / "outside.csv"
    outside.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(outside)
    with pytest.raises(ValueError, match="WRDS_DATA_DRIFT"):
        load_wrds_evaluation(path)


def test_changed_extraction_or_split_has_a_different_identity(tmp_path: Path) -> None:
    path, document = write_extract(tmp_path)
    first = load_wrds_evaluation(path).info["sha256"]
    document["extracted_at"] = "2021-01-02T00:00:00+00:00"
    path.write_text(dump_json_str(document), encoding="utf-8")
    second = load_wrds_evaluation(path).info["sha256"]
    assert first != second
    document["embargo"] = 2
    path.write_text(dump_json_str(document), encoding="utf-8")
    assert load_wrds_evaluation(path).info["sha256"] != second


__all__ = ["write_extract"]
