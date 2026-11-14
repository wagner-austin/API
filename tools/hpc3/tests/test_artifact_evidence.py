"""An array's artifacts are bound to completed allocations and to the local copies claimed."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import narrow_json_to_dict, require_dict, require_list

from hpc3.contracts.ledger import LedgerEntry
from hpc3.core.artifact_evidence import (
    array_members,
    artifact_report,
    measure_member,
    require_artifact,
)
from hpc3.core.digest import sha256_hex
from tests.conftest import FakeRun, cluster
from tests.execution_fixtures import (
    END,
    IMAGE,
    INSTANT,
    START,
    array_member,
    entry,
    serve_member,
)


def test_members_are_the_arrays_own_ids_in_member_order() -> None:
    entries = [
        array_member(10),
        entry("1011_0"),
        entry("101"),
        entry("101_x"),
        array_member(2),
        entry("102_0"),
    ]
    assert [value["job_id"] for value in array_members(entries, "101")] == ["101_2", "101_10"]


@pytest.mark.parametrize(
    ("entries", "array", "message"),
    [
        ([array_member(0)], "101_0", "numeric array id required"),
        ([array_member(0)], "1o1", "numeric array id required"),
        ([entry("101"), entry("1011_0")], "101", "records no member of array 101"),
        ([array_member(0), array_member(1), array_member(0)], "101", "recorded twice"),
    ],
)
def test_an_unknown_or_ambiguous_array_is_refused(
    entries: list[LedgerEntry], array: str, message: str
) -> None:
    with pytest.raises(ValueError, match=f"HPC_EVIDENCE_ARRAY: .*{message}"):
        array_members(entries, array)


def test_a_member_without_an_artifact_names_none() -> None:
    assert require_artifact(array_member(3)) == "/pub/out/shard-3.json"
    with pytest.raises(ValueError, match="HPC_EVIDENCE_IDENTITY: member 101_3 names no artifact"):
        require_artifact(array_member(3, artifact=None))


def test_a_member_is_its_accounting_digest_and_instant(fake_run: FakeRun, tmp_path: Path) -> None:
    value = array_member(0)
    local = tmp_path / "shard-0.json"
    local.write_bytes(b"shard zero")
    serve_member(fake_run, value, b"shard zero")
    measured = measure_member(value, "/pub/out/shard-0.json", cluster(), local)
    assert measured["job_id"] == "101_0"
    assert measured["name"] == "abl.sweep-shard0"
    assert measured["experiment"] == {"member": "shard0"}
    assert measured["image_digest"] == IMAGE
    assert measured["artifact"] == "/pub/out/shard-0.json"
    assert measured["artifact_sha256"] == sha256_hex(b"shard zero")
    assert measured["artifact_mtime"] == INSTANT
    assert measured["local_path"] == str(local)
    accounting = require_dict(measured, "accounting")
    assert accounting["exit_code"] == "0:0"
    assert (accounting["started_at"], accounting["ended_at"]) == (START, END)
    assert require_dict(accounting, "job")["name"] == "abl.sweep"


def test_a_local_copy_that_differs_is_refused(fake_run: FakeRun, tmp_path: Path) -> None:
    value = array_member(0)
    local = tmp_path / "shard-0.json"
    local.write_bytes(b"edited")
    serve_member(fake_run, value, b"shard zero")
    with pytest.raises(AppError, match=r"shard-0\.json"):
        measure_member(value, "/pub/out/shard-0.json", cluster(), local)


@pytest.mark.parametrize(
    "changes",
    [{"state": "FAILED"}, {"exit": "1:0"}, {"name": "abl.sweep-shard0"}, {"name": "other"}],
)
def test_a_member_that_did_not_complete_cleanly_under_its_array_is_refused(
    fake_run: FakeRun, tmp_path: Path, changes: dict[str, str]
) -> None:
    value = array_member(0)
    (tmp_path / "shard-0.json").write_bytes(b"shard zero")
    serve_member(fake_run, value, b"shard zero", **changes)
    with pytest.raises(ValueError, match="HPC_EVIDENCE_ACCOUNTING"):
        measure_member(value, "/pub/out/shard-0.json", cluster(), tmp_path / "shard-0.json")


def test_an_artifact_written_outside_its_allocation_is_refused(
    fake_run: FakeRun, tmp_path: Path
) -> None:
    value = array_member(0)
    (tmp_path / "shard-0.json").write_bytes(b"shard zero")
    fake_run.add("stat -c %Y -- /pub/out/shard-0.json", stdout=str(INSTANT + 600), once=True)
    serve_member(fake_run, value, b"shard zero")
    with pytest.raises(ValueError, match="HPC_EVIDENCE_TIME"):
        measure_member(value, "/pub/out/shard-0.json", cluster(), tmp_path / "shard-0.json")


def test_the_report_measures_every_member_against_its_named_copy(
    fake_run: FakeRun, tmp_path: Path
) -> None:
    for index in (0, 1):
        data = f"shard {index}".encode()
        (tmp_path / f"shard-{index}.json").write_bytes(data)
        serve_member(fake_run, array_member(index), data)
    members = [entry("7_0"), array_member(1), array_member(0)]
    report = artifact_report(members, "101", cluster(), tmp_path)
    assert (report["array"], report["local_dir"]) == ("101", str(tmp_path))
    measured = [narrow_json_to_dict(item) for item in require_list(report, "members")]
    assert [item["job_id"] for item in measured] == ["101_0", "101_1"]
    assert [item["artifact_sha256"] for item in measured] == [
        sha256_hex(b"shard 0"),
        sha256_hex(b"shard 1"),
    ]


def test_members_sharing_an_artifact_name_or_lacking_one_are_refused_before_any_read(
    fake_run: FakeRun, tmp_path: Path
) -> None:
    shared = [array_member(0), array_member(1, artifact="/pub/elsewhere/shard-0.json")]
    with pytest.raises(ValueError, match="share an artifact name"):
        artifact_report(shared, "101", cluster(), tmp_path)
    lacking = [array_member(0), array_member(1, artifact=None)]
    with pytest.raises(ValueError, match="names no artifact"):
        artifact_report(lacking, "101", cluster(), tmp_path)
    assert fake_run.commands() == []
