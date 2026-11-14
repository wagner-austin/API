"""An array's artifacts are bound to completed allocations and to the local copies claimed."""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import JSONValue, narrow_json_to_dict, require_dict, require_list

from hpc3.contracts.ledger import LedgerEntry
from hpc3.core.artifact_evidence import (
    array_members,
    artifact_report,
    measure_member,
    require_artifact,
)
from hpc3.core.digest import sha256_hex
from tests.conftest import FakeRun, cluster
from tests.execution_fixtures import END, IMAGE, INSTANT, START, accounting_row, entry


def member(index: int, /, **changes: JSONValue) -> LedgerEntry:
    """A recorded member of array 101 whose artifact is ``/pub/out/shard-<index>.json``.

    Args:
        index: Its member number.
        **changes: Alternate ledger fields.

    Returns:
        The decoded entry.
    """
    fields: dict[str, JSONValue] = {"artifact": f"/pub/out/shard-{index}.json"}
    fields.update(changes)
    return entry(f"101_{index}", **fields)


def serve_member(fake: FakeRun, value: LedgerEntry, data: bytes, **accounting: str) -> None:
    """Serve one member's accounting row, artifact digest and modification time.

    Args:
        fake: The remote-command seam.
        value: The member.
        data: The artifact's bytes on the cluster.
        **accounting: Accounting column replacements.
    """
    path = value["artifact"]
    assert path is not None
    fake.add(f"-j {value['job_id']} ", stdout=accounting_row(value, **accounting))
    fake.add(f"sha256sum '{path}'", stdout=f"{sha256_hex(data)}  {path}\n")
    fake.add(f"stat -c %Y -- {path}", stdout=str(INSTANT))


def test_members_are_the_arrays_own_ids_in_member_order() -> None:
    entries = [
        member(10),
        entry("1011_0"),
        entry("101"),
        entry("101_x"),
        member(2),
        entry("102_0"),
    ]
    assert [value["job_id"] for value in array_members(entries, "101")] == ["101_2", "101_10"]


@pytest.mark.parametrize(
    ("entries", "array", "message"),
    [
        ([member(0)], "101_0", "numeric array id required"),
        ([member(0)], "1o1", "numeric array id required"),
        ([entry("101"), entry("1011_0")], "101", "records no member of array 101"),
        ([member(0), member(1), member(0)], "101", "recorded twice"),
    ],
)
def test_an_unknown_or_ambiguous_array_is_refused(
    entries: list[LedgerEntry], array: str, message: str
) -> None:
    with pytest.raises(ValueError, match=f"HPC_EVIDENCE_ARRAY: .*{message}"):
        array_members(entries, array)


def test_a_member_without_an_artifact_names_none() -> None:
    assert require_artifact(member(3)) == "/pub/out/shard-3.json"
    with pytest.raises(ValueError, match="HPC_EVIDENCE_IDENTITY: member 101_3 names no artifact"):
        require_artifact(member(3, artifact=None))


def test_a_member_is_its_accounting_digest_and_instant(fake_run: FakeRun, tmp_path: Path) -> None:
    value = member(0)
    local = tmp_path / "shard-0.json"
    local.write_bytes(b"shard zero")
    serve_member(fake_run, value, b"shard zero")
    measured = measure_member(value, "/pub/out/shard-0.json", cluster(), local)
    assert measured["job_id"] == "101_0"
    assert measured["name"] == value["name"]
    assert measured["experiment"] == {"arm": "B", "repetition": "101_0"}
    assert measured["image_digest"] == IMAGE
    assert measured["artifact"] == "/pub/out/shard-0.json"
    assert measured["artifact_sha256"] == sha256_hex(b"shard zero")
    assert measured["artifact_mtime"] == INSTANT
    assert measured["local_path"] == str(local)
    accounting = require_dict(measured, "accounting")
    assert accounting["exit_code"] == "0:0"
    assert (accounting["started_at"], accounting["ended_at"]) == (START, END)


def test_a_local_copy_that_differs_is_refused(fake_run: FakeRun, tmp_path: Path) -> None:
    value = member(0)
    local = tmp_path / "shard-0.json"
    local.write_bytes(b"edited")
    serve_member(fake_run, value, b"shard zero")
    with pytest.raises(AppError, match=r"shard-0\.json"):
        measure_member(value, "/pub/out/shard-0.json", cluster(), local)


@pytest.mark.parametrize(
    ("changes", "code"),
    [
        ({"state": "FAILED"}, "HPC_EVIDENCE_ACCOUNTING"),
        ({"exit": "1:0"}, "HPC_EVIDENCE_ACCOUNTING"),
    ],
)
def test_a_member_that_did_not_complete_cleanly_is_refused(
    fake_run: FakeRun, tmp_path: Path, changes: dict[str, str], code: str
) -> None:
    value = member(0)
    (tmp_path / "shard-0.json").write_bytes(b"shard zero")
    serve_member(fake_run, value, b"shard zero", **changes)
    with pytest.raises(ValueError, match=code):
        measure_member(value, "/pub/out/shard-0.json", cluster(), tmp_path / "shard-0.json")


def test_an_artifact_written_outside_its_allocation_is_refused(
    fake_run: FakeRun, tmp_path: Path
) -> None:
    value = member(0)
    (tmp_path / "shard-0.json").write_bytes(b"shard zero")
    fake_run.add("stat -c %Y -- /pub/out/shard-0.json", stdout=str(INSTANT + 600), once=True)
    serve_member(fake_run, value, b"shard zero")
    with pytest.raises(ValueError, match="HPC_EVIDENCE_TIME"):
        measure_member(value, "/pub/out/shard-0.json", cluster(), tmp_path / "shard-0.json")


def test_the_report_measures_every_member_against_its_named_copy(
    fake_run: FakeRun, tmp_path: Path
) -> None:
    members = [member(1), member(0)]
    for index, value in enumerate(sorted(members, key=lambda value: value["job_id"])):
        data = f"shard {index}".encode()
        (tmp_path / f"shard-{index}.json").write_bytes(data)
        serve_member(fake_run, value, data)
    report = artifact_report([entry("7_0"), *members], "101", cluster(), tmp_path)
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
    shared = [member(0), member(1, artifact="/pub/elsewhere/shard-0.json")]
    with pytest.raises(ValueError, match="share an artifact name"):
        artifact_report(shared, "101", cluster(), tmp_path)
    with pytest.raises(ValueError, match="names no artifact"):
        artifact_report([member(0), member(1, artifact=None)], "101", cluster(), tmp_path)
    assert fake_run.commands() == []
