"""Evaluator verification of collected output with real keys and the existing reader."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    narrow_json_to_list,
)

from covenant_ml.markets.crsp_verify import AGE_HEADER, verify_crsp_extraction
from covenant_ml.markets.crsp_writer import extract_crsp
from covenant_ml.markets.research_keys import create_evaluator_keys, encode_research_access

from ._crsp_extraction_fixtures import (
    Extraction,
    FakeDatabase,
    database_rows,
    extraction,
    installed,
    limits,
    run_vendor,
)
from ._encrypted_fixtures import evaluator_keys


@pytest.fixture()
def database() -> Generator[FakeDatabase, None, None]:
    with installed(FakeDatabase(database_rows())) as fake:
        yield fake


def _scratch(case: Extraction) -> Path:
    path = case.root / "evaluator-scratch"
    path.mkdir(mode=0o700, exist_ok=True)
    return path


def _verify(case: Extraction, *, open_final: bool) -> JSONObject:
    return verify_crsp_extraction(
        case.root, evaluator_keys(case.root), scratch=_scratch(case), open_final=open_final
    )


VALIDATION: JSONObject = {
    "partition": "train-validation",
    "opened": True,
    "observation_end": "2019-12-31",
    "latest_return": "2019-12-31",
    "latest_membership_end": "2019-12-31",
    "latest_delisting": "2019-03-14",
    "latest_calendar": "2020-02-29",
}


def test_validation_opens_and_final_stays_sealed_for_licensed_evidence(
    tmp_path: Path, database: FakeDatabase
) -> None:
    case = extraction(tmp_path)
    receipt = run_vendor(case, limits())
    result = _verify(case, open_final=False)
    item = receipt["train_validation"]
    assert result == {
        "schema": "crsp_a_stock",
        "snapshot": "750:750:",
        "ciphertext_files": 20,
        "partitions": [
            VALIDATION
            | {
                "rows": [
                    item["membership_rows"],
                    item["returns_rows"],
                    item["calendar_rows"],
                    item["delists_rows"],
                ],
                "plaintext_removed": True,
            },
            {"partition": "final", "opened": False},
        ],
    }
    assert list(_scratch(case).iterdir()) == []
    with pytest.raises(ValueError, match=r"\[CRSP-013\] licensed final evidence opens only"):
        _verify(case, open_final=True)


def test_synthetic_final_partition_opens_with_its_own_key(
    tmp_path: Path, database: FakeDatabase
) -> None:
    case = extraction(tmp_path)
    setup = tmp_path / "setup.sql"
    setup.write_text("SELECT 1;\n", encoding="utf-8")
    extract_crsp(
        case.protocol,
        case.access,
        schema="pg_temp",
        setup=setup,
        scratch=case.scratch,
        output=case.output,
        limits=limits(),
    )
    partitions = narrow_json_to_list(_verify(case, open_final=True)["partitions"])
    final = narrow_json_to_dict(partitions[1])
    assert {key: final[key] for key in VALIDATION} == {
        "partition": "final",
        "opened": True,
        "observation_end": "2020-08-31",
        "latest_return": "2020-08-31",
        "latest_membership_end": "2020-08-31",
        "latest_delisting": "2019-12-20",
        "latest_calendar": "2020-08-31",
    }
    assert final["plaintext_removed"] is True


@pytest.mark.parametrize(
    ("name", "payload", "message"),
    [
        ("train-validation/returns.csv", b"permno\n", "is not age ciphertext"),
        ("final/extra.age", b"plain", "is not age ciphertext"),
        ("notes.txt", b"x", "unexpected collected file notes.txt"),
        ("final/nested/x.age", AGE_HEADER, "unexpected collected file final/nested/x.age"),
        (".pending-final/bundle.json", b"{}", "unexpected collected file .pending-final"),
    ],
)
def test_any_file_beyond_ciphertext_and_metadata_is_refused(
    tmp_path: Path, database: FakeDatabase, name: str, payload: bytes, message: str
) -> None:
    case = extraction(tmp_path)
    run_vendor(case, limits())
    target = case.output / name
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(payload)
    with pytest.raises(ValueError, match=rf"\[CRSP-013\] .*{message}"):
        _verify(case, open_final=False)


def test_a_bundle_sealed_to_other_recipients_is_refused(
    tmp_path: Path, database: FakeDatabase
) -> None:
    case = extraction(tmp_path)
    run_vendor(case, limits())
    other = create_evaluator_keys(tmp_path / "other-keys", 10_000_000)
    (tmp_path / "access.json").write_text(
        dump_json_str(encode_research_access(other)), encoding="utf-8"
    )
    with pytest.raises(ValueError, match=r"\[CRSP-013\] a bundle is not sealed"):
        _verify(case, open_final=False)


def _rewrite(case: Extraction, field: str, value: JSONValue) -> None:
    """Change one field of the published train-validation receipt."""
    path = case.output / "extraction.json"
    receipt = narrow_json_to_dict(load_json_str(path.read_text("utf-8")))
    partition = narrow_json_to_dict(receipt["train_validation"])
    partition[field] = value
    receipt["train_validation"] = partition
    path.write_text(dump_json_str(receipt), encoding="utf-8")


def test_receipt_counts_disagreeing_with_the_opened_bundle_are_refused(
    tmp_path: Path, database: FakeDatabase
) -> None:
    case = extraction(tmp_path)
    run_vendor(case, limits())
    _rewrite(case, "returns_rows", 1)
    with pytest.raises(ValueError, match=r"\[CRSP-013\] decrypted row counts differ"):
        _verify(case, open_final=False)


def test_receipt_manifest_disagreeing_with_the_opened_bundle_is_refused(
    tmp_path: Path, database: FakeDatabase
) -> None:
    case = extraction(tmp_path)
    run_vendor(case, limits())
    _rewrite(case, "manifest", {"name": "manifest.json", "sha256": "0" * 64, "size_bytes": 1})
    with pytest.raises(ValueError, match=r"\[CRSP-013\] decrypted manifest differs"):
        _verify(case, open_final=False)
