"""Exact-byte, schema, missingness and natural-key checks on real fixture files."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.json_utils import JSONValue, dump_json_str, narrow_json_to_dict

from covenant_ml.markets.crsp_extract import (
    decode_crsp_extract,
    encode_crsp_extract,
    load_crsp_sources,
)
from covenant_ml.markets.crsp_rows import decode_crsp_rows

from ._crsp_fixtures import save_file, source_rows, write_sources


def test_manifest_roundtrip_and_missing_terminal_values_remain_null(tmp_path: Path) -> None:
    rows = source_rows()
    rows["delists"] = [
        {
            "permno": 9,
            "delistingdt": "2019-06-01",
            "deldlydt": "2019-06-03",
            "delamtdt": None,
            "delnextdt": None,
            "delret": None,
            "delretmisstype": "DM",
        }
    ]
    rows["returns"][0]["dlyprevdt"] = None
    path, document = write_sources(tmp_path, rows)
    assert encode_crsp_extract(decode_crsp_extract(document)) == document
    loaded = load_crsp_sources(path)
    assert loaded.membership == tuple(rows["membership"])
    assert loaded.calendar == tuple(rows["calendar"])
    assert loaded.returns == tuple(rows["returns"])
    assert loaded.delists == tuple(rows["delists"])


@pytest.mark.parametrize(
    "name",
    [
        "protocol.json",
        "membership.csv",
        "membership.sql",
        "returns.csv",
        "returns.sql",
        "calendar.csv",
        "calendar.sql",
        "delists.csv",
        "delists.sql",
    ],
)
def test_changed_artifact_is_refused_before_consumption(tmp_path: Path, name: str) -> None:
    path, _ = write_sources(tmp_path, source_rows())
    with (tmp_path / name).open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="WRDS_DATA_DRIFT"):
        load_crsp_sources(path)


@pytest.mark.parametrize(
    "mutation", ["extra", "time", "relation", "repeated", "size", "empty_query", "newline"]
)
def test_ambiguous_manifest_is_refused(tmp_path: Path, mutation: str) -> None:
    path, document = write_sources(tmp_path, source_rows())
    membership = narrow_json_to_dict(document["membership"])
    if mutation == "extra":
        document["unexpected"] = 1
    elif mutation == "time":
        document["extracted_at"] = "2026-10-29T00:00:00"
    elif mutation == "relation":
        membership["unexpected"] = 1
    elif mutation == "repeated":
        membership["query"] = membership["csv"]
    elif mutation == "size":
        narrow_json_to_dict(membership["csv"])["size_bytes"] = 1
    elif mutation == "empty_query":
        membership["query"] = save_file(tmp_path, "membership.sql", b" \n")
    else:
        narrow_json_to_dict(membership["csv"])["name"] = "bad\nname"
    path.write_text(dump_json_str(document), encoding="utf-8")
    with pytest.raises(ValueError, match=r"CRSP-006|WRDS_DATA_PATH"):
        load_crsp_sources(path)


@pytest.mark.parametrize("relation", ["membership", "returns", "calendar", "delists"])
def test_duplicate_natural_keys_fail(tmp_path: Path, relation: str) -> None:
    rows = source_rows()
    rows["delists"] = [
        {
            "permno": 1,
            "delistingdt": "2019-06-03",
            "deldlydt": "2019-06-04",
            "delamtdt": "2019-06-05",
            "delnextdt": "2019-06-06",
            "delret": -0.5,
            "delretmisstype": "NA",
        }
    ]
    rows[relation].append(rows[relation][0].copy())
    path, _ = write_sources(tmp_path, rows)
    with pytest.raises(ValueError, match=r"CRSP-007.*duplicate"):
        load_crsp_sources(path)


@pytest.mark.parametrize(
    ("relation", "field", "value"),
    [
        ("membership", "permno", 0),
        ("membership", "permno", "01"),
        ("membership", "nameendt", "2017-01-01"),
        ("returns", "dlyret", "NaN"),
        ("returns", "dlyret", -1.1),
        ("returns", "dlyvol", -1.0),
        ("returns", "dlyprevdt", "2020-01-01"),
        ("returns", "dlyretmissflg", ""),
        ("returns", "dlycaldt", "2026-10-29"),
        ("calendar", "exchangegroup", "OTHER"),
        ("calendar", "tradingflg", "X"),
    ],
)
def test_malformed_source_cells_fail(
    tmp_path: Path, relation: str, field: str, value: JSONValue
) -> None:
    rows = source_rows()
    rows[relation][0][field] = value
    path, _ = write_sources(tmp_path, rows)
    with pytest.raises(ValueError, match="CRSP-007"):
        load_crsp_sources(path)


@pytest.mark.parametrize("payload", [b"", b"wrong\n", b"permno,namedt\n1\n"])
def test_projection_and_width_fail(payload: bytes) -> None:
    with pytest.raises(ValueError, match="CRSP-007"):
        decode_crsp_rows(payload, "membership", ("permno", "namedt"), "2026-10-29T00:00:00Z")
    with pytest.raises(ValueError, match=r"CRSP-007.*unknown"):
        decode_crsp_rows(payload, "unknown", (), "2026-10-29T00:00:00Z")
