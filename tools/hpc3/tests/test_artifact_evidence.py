"""A sweep's artifacts are bound to the job that completed each and to the local copies claimed."""

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
from hpc3.core.execution_evidence import accounting
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


def write_shards(local: Path, count: int) -> None:
    """Keep a local copy of each of ``count`` shards, ``shard <index>``.

    Args:
        local: The directory.
        count: How many.
    """
    for index in range(count):
        (local / f"shard-{index}.json").write_bytes(f"shard {index}".encode())


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
    evidence = accounting(value, cluster())
    measured = measure_member(value, "/pub/out/shard-0.json", evidence, local)
    assert measured["job_id"] == "101_0"
    assert measured["name"] == "abl.sweep-shard0"
    assert measured["experiment"] == {"member": "shard0"}
    assert measured["image_digest"] == IMAGE
    assert measured["artifact"] == "/pub/out/shard-0.json"
    assert measured["artifact_sha256"] == sha256_hex(b"shard zero")
    assert measured["artifact_mtime"] == INSTANT
    assert measured["local_path"] == str(local)
    recorded = require_dict(measured, "accounting")
    assert recorded["exit_code"] == "0:0"
    assert (recorded["started_at"], recorded["ended_at"]) == (START, END)
    assert require_dict(recorded, "job")["name"] == "abl.sweep"


def test_a_local_copy_that_differs_is_refused(fake_run: FakeRun, tmp_path: Path) -> None:
    value = array_member(0)
    local = tmp_path / "shard-0.json"
    local.write_bytes(b"edited")
    serve_member(fake_run, value, b"shard zero")
    with pytest.raises(AppError, match=r"shard-0\.json"):
        measure_member(value, "/pub/out/shard-0.json", accounting(value, cluster()), local)


def test_an_artifact_written_outside_its_allocation_is_refused(
    fake_run: FakeRun, tmp_path: Path
) -> None:
    value = array_member(0)
    (tmp_path / "shard-0.json").write_bytes(b"shard zero")
    fake_run.add("stat -c %Y -- /pub/out/shard-0.json", stdout=str(INSTANT + 600), once=True)
    serve_member(fake_run, value, b"shard zero")
    evidence = accounting(value, cluster())
    with pytest.raises(ValueError, match="HPC_EVIDENCE_TIME"):
        measure_member(value, "/pub/out/shard-0.json", evidence, tmp_path / "shard-0.json")


def test_the_report_measures_every_member_against_its_named_copy(
    fake_run: FakeRun, tmp_path: Path
) -> None:
    write_shards(tmp_path, 2)
    for index in (0, 1):
        serve_member(fake_run, array_member(index), f"shard {index}".encode())
    members = [entry("7_0"), array_member(1), array_member(0)]
    report = artifact_report(members, ["101"], cluster(), tmp_path)
    assert (report["arrays"], report["local_dir"]) == (["101"], str(tmp_path))
    measured = [narrow_json_to_dict(item) for item in require_list(report, "members")]
    assert [item["job_id"] for item in measured] == ["101_0", "101_1"]
    assert [item["artifact_sha256"] for item in measured] == [
        sha256_hex(b"shard 0"),
        sha256_hex(b"shard 1"),
    ]
    assert report["superseded"] == []


def test_a_campaign_resubmission_settles_the_artifact_its_first_member_failed(
    fake_run: FakeRun, tmp_path: Path
) -> None:
    """Array 59589599's shards 2 and 3 failed and campaign array 59590234 reran them."""
    write_shards(tmp_path, 2)
    serve_member(fake_run, array_member(0), b"shard 0")
    serve_member(fake_run, array_member(1), b"shard 1", state="FAILED", exit="1:0")
    rerun = array_member(1, "202")
    serve_member(fake_run, rerun, b"shard 1")
    entries = [array_member(0), array_member(1), rerun]
    report = artifact_report(entries, ["101", "202"], cluster(), tmp_path)
    measured = [narrow_json_to_dict(item) for item in require_list(report, "members")]
    assert [item["job_id"] for item in measured] == ["101_0", "202_1"]
    assert report["superseded"] == [
        {
            "job_id": "101_1",
            "artifact": "/pub/out/shard-1.json",
            "state": "FAILED",
            "exit_code": "1:0",
        }
    ]


@pytest.mark.parametrize("name", ["abl.sweep-shard0", "other"])
def test_accounting_under_another_job_name_is_refused(
    fake_run: FakeRun, tmp_path: Path, name: str
) -> None:
    """A member's row must carry its array's one Slurm name, not its own or another's."""
    write_shards(tmp_path, 1)
    serve_member(fake_run, array_member(0), b"shard 0", name=name)
    with pytest.raises(ValueError, match="HPC_EVIDENCE_ACCOUNTING: allocation does not match"):
        artifact_report([array_member(0)], ["101"], cluster(), tmp_path)


@pytest.mark.parametrize(
    ("first", "second", "message"),
    [
        ({"state": "FAILED", "exit": "1:0"}, {"state": "TIMEOUT", "exit": "0:15"}, "completed"),
        ({}, {}, "was completed twice"),
        ({"state": "FAILED", "exit": "1:0"}, {"state": "RUNNING"}, "still being written"),
    ],
)
def test_an_artifact_no_single_finished_member_completed_is_refused(
    fake_run: FakeRun,
    tmp_path: Path,
    first: dict[str, str],
    second: dict[str, str],
    message: str,
) -> None:
    write_shards(tmp_path, 1)
    original, rerun = array_member(0), array_member(0, "202")
    serve_member(fake_run, original, b"shard 0", **first)
    serve_member(fake_run, rerun, b"shard 0", **second)
    with pytest.raises(ValueError, match=f"HPC_EVIDENCE_ARRAY: .*{message}"):
        artifact_report([original, rerun], ["101", "202"], cluster(), tmp_path)


@pytest.mark.parametrize("arrays", [[], ["101", "101"]])
def test_each_array_is_named_once(arrays: list[str], fake_run: FakeRun, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="name each of the sweep's arrays once"):
        artifact_report([array_member(0)], arrays, cluster(), tmp_path)
    assert fake_run.commands() == []


def test_members_sharing_an_artifact_name_or_lacking_one_are_refused_before_any_read(
    fake_run: FakeRun, tmp_path: Path
) -> None:
    shared = [array_member(0), array_member(1, artifact="/pub/elsewhere/shard-0.json")]
    with pytest.raises(ValueError, match="share a file name"):
        artifact_report(shared, ["101"], cluster(), tmp_path)
    lacking = [array_member(0), array_member(1, artifact=None)]
    with pytest.raises(ValueError, match="names no artifact"):
        artifact_report(lacking, ["101"], cluster(), tmp_path)
    assert fake_run.commands() == []
