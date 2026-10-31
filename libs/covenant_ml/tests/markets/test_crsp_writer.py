"""The writer from client output to sealed bundles, opened again with real keys."""

from __future__ import annotations

from collections.abc import Generator
from hashlib import sha256
from pathlib import Path

import pytest
from platform_core.json_utils import (
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_int,
    require_str,
)

from covenant_ml.markets.crsp_extract import CrspSources, load_crsp_sources
from covenant_ml.markets.crsp_extraction_records import (
    decode_crsp_extraction_receipt,
    encode_crsp_extraction_receipt,
)
from covenant_ml.markets.crsp_queries import (
    crsp_relation_sql,
    crsp_scopes,
    synthetic_source,
    vendor_source,
)
from covenant_ml.markets.crsp_writer import extract_crsp
from covenant_ml.markets.encrypted_storage import decrypted_bundle

from ._crsp_database import (
    INELIGIBLE,
    LATE,
    STRADDLE,
    copy_csv,
    database_protocol,
    partition_rows,
    relation_columns,
)
from ._crsp_extraction_fixtures import (
    NOW,
    Extraction,
    FakeDatabase,
    database_rows,
    extraction,
    installed,
    limits,
    run_vendor,
)
from ._encrypted_fixtures import evaluator_keys

FILES = {
    "bundle.json",
    "manifest.json.age",
    "protocol.json.age",
    *(
        f"{name}.{kind}.age"
        for name in ("membership", "returns", "calendar", "delists")
        for kind in ("csv", "sql")
    ),
}


@pytest.fixture()
def database() -> Generator[FakeDatabase, None, None]:
    with installed(FakeDatabase(database_rows())) as fake:
        yield fake


def _open(case: Extraction, partition: str, digest: str) -> CrspSources:
    keys = evaluator_keys(case.root)
    opened = case.root / ("opened-" + partition)
    opened.mkdir(mode=0o700)
    with decrypted_bundle(
        case.output / partition / "bundle.json",
        digest,
        partition=partition,
        identity=keys.acquire(partition),
        scratch=opened,
        max_plaintext_bytes=10_000_000,
    ) as manifest:
        return load_crsp_sources(manifest)


def _permnos(sources: CrspSources) -> set[int]:
    return {require_int(row, "permno") for row in (*sources.membership, *sources.delists)}


def test_one_snapshot_yields_two_separate_sealed_partitions(
    tmp_path: Path, database: FakeDatabase
) -> None:
    case = extraction(tmp_path)
    receipt = run_vendor(case, limits())
    assert database.calls == [(120, 2_000_000)]
    assert sorted(path.name for path in case.output.iterdir()) == [
        "extraction.json",
        "final",
        "metrics.json",
        "train-validation",
    ]
    for partition in ("train-validation", "final"):
        assert {path.name for path in (case.output / partition).iterdir()} == FILES
    assert list(case.scratch.iterdir()) == []
    published = load_json_str((case.output / "extraction.json").read_text("utf-8"))
    assert decode_crsp_extraction_receipt(published) == receipt
    assert encode_crsp_extraction_receipt(receipt) == published
    assert (receipt["started_at"], receipt["completed_at"]) == (NOW, NOW)
    assert (receipt["snapshot"], receipt["transaction_started_at"]) == (
        "750:750:",
        "2026-10-31 09:00:00.5+00",
    )
    assert receipt["driver_sha256"] == sha256(database.driver.encode()).hexdigest()
    assert receipt["protocol"]["sha256"] == sha256(case.protocol.read_bytes()).hexdigest()
    assert (receipt["source"], receipt["setup_sha256"], receipt["plaintext_removed"]) == (
        "crsp_a_stock",
        None,
        True,
    )
    protocol = database_protocol()
    validation, final = crsp_scopes(protocol)
    opened = _open(case, "train-validation", receipt["train_validation"]["bundle"]["sha256"])
    expected = partition_rows(database_rows(), protocol, validation)
    assert opened.manifest["extracted_at"] == NOW
    assert opened.protocol == protocol
    assert list(opened.membership) == expected["membership"]
    assert list(opened.returns) == expected["returns"]
    assert list(opened.delists) == expected["delists"]
    assert len(opened.returns) == receipt["train_validation"]["returns_rows"]
    assert {require_str(row, "nameendt") for row in opened.membership} == {"2019-12-31"}
    assert max(require_str(row, "dlycaldt") for row in opened.returns) == "2019-12-31"
    assert require_str(opened.calendar[-1], "caldt") == "2020-02-29"
    assert _permnos(opened) == set(range(1, 9))
    sealed = _open(case, "final", receipt["final"]["bundle"]["sha256"])
    assert list(sealed.returns) == partition_rows(database_rows(), protocol, final)["returns"]
    assert _permnos(sealed) == set(range(1, 9)) | {LATE}
    assert STRADDLE in {require_int(row, "permno") for row in sealed.delists}
    assert STRADDLE not in {require_int(row, "permno") for row in opened.delists}
    assert INELIGIBLE not in _permnos(sealed)


def test_manifest_preserves_the_exact_queries_and_protocol_bytes(
    tmp_path: Path, database: FakeDatabase
) -> None:
    case = extraction(tmp_path)
    receipt = run_vendor(case, limits(max_rows=15000))
    keys = evaluator_keys(case.root)
    opened = case.root / "opened"
    opened.mkdir(mode=0o700)
    protocol = database_protocol()
    queries = crsp_relation_sql(protocol, crsp_scopes(protocol)[1], vendor_source(protocol), 15000)
    with decrypted_bundle(
        case.output / "final" / "bundle.json",
        receipt["final"]["bundle"]["sha256"],
        partition="final",
        identity=keys.acquire("final"),
        scratch=opened,
        max_plaintext_bytes=10_000_000,
    ) as manifest:
        directory = manifest.parent
        for name, sql in queries.items():
            assert (directory / (name + ".sql")).read_text("utf-8") == sql
        assert (directory / "protocol.json").read_bytes() == case.protocol.read_bytes()
        assert sha256(manifest.read_bytes()).hexdigest() == receipt["final"]["manifest"]["sha256"]
    assert load_json_str((case.output / "metrics.json").read_text("utf-8")) == [
        {"name": f"{partition}_{name}_rows", "value": float(count)}
        for partition, item in (
            ("train_validation", receipt["train_validation"]),
            ("final", receipt["final"]),
        )
        for name, count in (
            ("membership", item["membership_rows"]),
            ("returns", item["returns_rows"]),
            ("calendar", item["calendar_rows"]),
            ("delists", item["delists_rows"]),
        )
    ] + [{"name": "plaintext_removed", "value": 1.0}]


def test_a_synthetic_source_carries_its_definitions_in_every_saved_query(
    tmp_path: Path, database: FakeDatabase
) -> None:
    case = extraction(tmp_path)
    setup = tmp_path / "definitions.sql"
    setup.write_text("dsenames(permno) AS (SELECT 1)\n", encoding="utf-8")
    receipt = extract_crsp(
        case.protocol,
        case.access,
        source="synthetic",
        setup=setup,
        scratch=case.scratch,
        output=case.output,
        limits=limits(),
    )
    assert (receipt["source"], receipt["setup_sha256"]) == (
        "synthetic",
        sha256(setup.read_bytes()).hexdigest(),
    )
    assert database.driver.splitlines()[0] == "BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;"
    keys = evaluator_keys(case.root)
    opened = case.root / "opened"
    opened.mkdir(mode=0o700)
    protocol = database_protocol()
    source = synthetic_source(setup.read_text("utf-8"))
    queries = crsp_relation_sql(protocol, crsp_scopes(protocol)[0], source, 20000)
    with decrypted_bundle(
        case.output / "train-validation" / "bundle.json",
        receipt["train_validation"]["bundle"]["sha256"],
        partition="train-validation",
        identity=keys.acquire("train-validation"),
        scratch=opened,
        max_plaintext_bytes=10_000_000,
    ) as manifest:
        for name, sql in queries.items():
            saved = (manifest.parent / (name + ".sql")).read_text("utf-8")
            assert saved == sql
            assert saved.startswith("COPY (WITH dsenames(permno) AS (SELECT 1)\nSELECT ")


def test_synthetic_definitions_that_could_end_a_statement_are_refused_before_any_query(
    tmp_path: Path, database: FakeDatabase
) -> None:
    case = extraction(tmp_path)
    setup = tmp_path / "definitions.sql"
    setup.write_text("dsenames(permno) AS (SELECT 1); DROP TABLE x\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"\[CRSP-010\] synthetic definitions must be"):
        extract_crsp(
            case.protocol,
            case.access,
            source="synthetic",
            setup=setup,
            scratch=case.scratch,
            output=case.output,
            limits=limits(),
        )
    assert database.calls == []


@pytest.mark.parametrize(
    ("source", "with_setup", "message"),
    [
        ("synthetic", False, "definitions are required exactly"),
        ("crsp_a_stock", True, "definitions are required exactly"),
        ("crsp", False, "the source must be the protocol's library"),
    ],
)
def test_source_must_be_the_vendor_library_or_an_explicit_synthetic_session(
    tmp_path: Path, database: FakeDatabase, source: str, with_setup: bool, message: str
) -> None:
    case = extraction(tmp_path)
    setup = tmp_path / "definitions.sql"
    setup.write_text("dsenames(permno) AS (SELECT 1)\n", encoding="utf-8")
    with pytest.raises(ValueError, match=rf"\[CRSP-010\] {message}"):
        extract_crsp(
            case.protocol,
            case.access,
            source=source,
            setup=setup if with_setup else None,
            scratch=case.scratch,
            output=case.output,
            limits=limits(),
        )
    assert database.calls == []


def test_coverage_must_end_before_the_extraction_day(
    tmp_path: Path, database: FakeDatabase
) -> None:
    case = extraction(tmp_path)
    document = narrow_json_to_dict(load_json_str(case.protocol.read_text("utf-8")))
    document["coverage_end"] = "2026-10-31"
    case.protocol.write_text(dump_json_str(document), encoding="utf-8")
    with pytest.raises(ValueError, match=r"\[CRSP-010\] coverage must end before"):
        run_vendor(case, limits())
    assert database.calls == []


@pytest.mark.parametrize("name", ["final", ".pending-train-validation", "metrics.json"])
def test_existing_output_names_are_never_replaced(
    tmp_path: Path, database: FakeDatabase, name: str
) -> None:
    case = extraction(tmp_path)
    (case.output / name).mkdir()
    with pytest.raises(FileExistsError, match=r"\[CRSP-011\] refusing to replace"):
        run_vendor(case, limits())
    assert database.calls == []


def _refused(case: Extraction) -> None:
    """No bundle, receipt or plaintext remains after a refusal."""
    assert list(case.scratch.iterdir()) == []
    assert list(case.output.iterdir()) == []


def test_a_failed_client_publishes_nothing(tmp_path: Path, database: FakeDatabase) -> None:
    database.code = 3
    case = extraction(tmp_path)
    with pytest.raises(ValueError, match=r"\[CRSP-011\] psql exited 3"):
        run_vendor(case, limits())
    _refused(case)
    assert list(case.output.iterdir()) == []


def test_differing_snapshot_readings_are_refused(tmp_path: Path, database: FakeDatabase) -> None:
    database.snapshots = ("750:750:,t\n", "751:751:,t\n")
    case = extraction(tmp_path)
    with pytest.raises(ValueError, match=r"\[CRSP-011\] all relations must come from one"):
        run_vendor(case, limits())
    assert list(case.output.iterdir()) == []
    _refused(case)


def test_partition_plaintext_beyond_its_budget_is_refused_before_reading(
    tmp_path: Path, database: FakeDatabase
) -> None:
    case = extraction(tmp_path)
    with pytest.raises(ValueError, match=r"\[CRSP-012\] partition plaintext exceeds"):
        run_vendor(case, limits(max_relation_bytes=100_000, max_partition_bytes=100_000))
    assert list(case.output.iterdir()) == []
    _refused(case)


def test_final_period_rows_in_the_validation_partition_are_refused(
    tmp_path: Path, database: FakeDatabase
) -> None:
    protocol = database_protocol()
    final = partition_rows(database_rows(), protocol, crsp_scopes(protocol)[1])
    database.extra = {
        "train-validation/returns.csv": copy_csv(
            final["returns"], relation_columns(protocol)["returns"]
        )
    }
    case = extraction(tmp_path)
    with pytest.raises(ValueError, match=r"\[CRSP-012\] dlycaldt lies outside"):
        run_vendor(case, limits())
    assert list(case.output.iterdir()) == []
    _refused(case)


def test_a_truncated_relation_is_refused(tmp_path: Path, database: FakeDatabase) -> None:
    protocol = database_protocol()
    rows = len(partition_rows(database_rows(), protocol, crsp_scopes(protocol)[0])["returns"])
    case = extraction(tmp_path)
    with pytest.raises(ValueError, match=r"\[CRSP-012\] returns exceeds its row limit"):
        run_vendor(case, limits(max_rows=rows - 1))
    _refused(case)


def test_a_malformed_relation_is_refused_by_the_existing_reader(
    tmp_path: Path, database: FakeDatabase
) -> None:
    database.extra = {"final/calendar.csv": b"caldt,exchangegroup\n"}
    case = extraction(tmp_path)
    with pytest.raises(ValueError, match=r"\[CRSP-007\] CSV projection differs"):
        run_vendor(case, limits())
    assert list(case.output.iterdir()) == []
    _refused(case)
