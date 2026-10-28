"""Catalog validation and the WRDS filesystem inventory over Grid Engine."""

from __future__ import annotations

import hashlib
import pathlib
import runpy
import sys

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import JSONTypeError, JSONValue, load_json_str
from platform_core.mcp_testing import DECLARED_TASKBOARD_URL, sent_arguments

from hpc3.cli import wrds_catalog as cli
from hpc3.contracts.wrds_catalog import (
    decode_catalog_entry,
    decode_catalog_receipt,
    encode_catalog_entry,
    encode_catalog_receipt,
)
from hpc3.core.wrds_catalog import inventory, probe_archive, read_catalog
from tests.conftest import FakeRun
from tests.wrds_board import TASK, accepting_board

CATALOG = (
    '["directory", "/wrds", 0, true, "/wrdslin"]\n'
    '["file", "/wrds/comp/sasdata/funda.sas7bdat", 42, true, "/data/funda"]\n'
    '["file", "/wrds/other/table.sas7bdat", 8, false, "/data/table"]\n'
    '["denied_directory", "/wrds/secret", 0, false, "/data/secret"]\n'
    '["unresolved_link", "/wrds/gone", 0, false, "/data/gone"]\n'
    '["complete", "/wrds", 0, false, ""]\n'
)
CATALOG_SHA = hashlib.sha256(CATALOG.encode()).hexdigest()
SOURCE_SHA = hashlib.sha256(probe_archive()).hexdigest()


def prepare(
    fake_run: FakeRun, *, source_sha: str | None = None, catalog_sha: str | None = None
) -> None:
    """Supply scheduler and digest replies for the transport contract.

    Args:
        fake_run: Existing process seam fixture.
        source_sha: Explicit digest for a corrupt upload test.
        catalog_sha: Explicit digest for a corrupt download test.
    """
    fake_run.add(
        "sha256sum",
        stdout=(SOURCE_SHA if source_sha is None else source_sha) + "  scan.pyz\n",
        once=True,
    )
    fake_run.add(
        "sha256sum",
        stdout=(CATALOG_SHA if catalog_sha is None else catalog_sha) + "  catalog.jsonl\n",
    )
    fake_run.add("qsub", stdout="12345\nJob 12345 exited with exit code 0.\n")
    fake_run.add("cat --", stdout=CATALOG)


@pytest.mark.parametrize("value", [None, [], ["file"], ["file", "/x", 1, True, "", 7]])
def test_wrong_record_shape(value: JSONValue) -> None:
    with pytest.raises(JSONTypeError, match="WRDS_CATALOG_RECORD"):
        decode_catalog_entry(value)


@pytest.mark.parametrize(
    "value",
    [
        ["unknown", "/x", 1, True, ""],
        ["file", "", 1, True, ""],
        ["file", "/x", -1, True, ""],
        ["file", "/x", True, True, ""],
        ["file", "/x", 1, "yes", ""],
    ],
)
def test_invalid_record_fields(value: JSONValue) -> None:
    with pytest.raises(JSONTypeError):
        decode_catalog_entry(value)


def test_wire_roundtrip_and_incomplete_artifacts() -> None:
    for record in read_catalog(CATALOG):
        assert decode_catalog_entry(encode_catalog_entry(record)) == record
    for text in ("", CATALOG.splitlines()[0], CATALOG + CATALOG):
        with pytest.raises(JSONTypeError, match="WRDS_CATALOG_INCOMPLETE"):
            read_catalog(text)


def test_inventory_copies_only_verified_metadata_and_receipt(
    tmp_path: pathlib.Path,
    fake_run: FakeRun,
) -> None:
    prepare(fake_run)
    output = tmp_path / "catalog.jsonl"
    run = inventory("wrds", "/home/uci/awagner14/catalog one", output)
    assert run["entries"] == read_catalog(CATALOG)
    assert run["receipt"] == {
        "host": "wrds",
        "scheduler": "grid-engine",
        "job_id": "12345",
        "source_sha256": SOURCE_SHA,
        "catalog_sha256": CATALOG_SHA,
        "remote_catalog": "/home/uci/awagner14/catalog one/catalog.jsonl",
        "scope": "/wrds filesystem metadata, including historical trees",
    }
    assert output.read_text() == CATALOG
    saved = load_json_str(output.with_suffix(".jsonl.receipt.json").read_text())
    assert decode_catalog_receipt(saved) == run["receipt"]
    commands = fake_run.commands()
    assert commands[0] == "mkdir -m 700 -- '/home/uci/awagner14/catalog one'"
    submitted = [command for command in commands if command.startswith("qsub ")]
    assert len(submitted) == 1
    assert "-sync y -b y" in submitted[0]
    assert "-V" not in submitted[0].split()
    assert "--root /wrds" in submitted[0]


def test_receipt_roundtrip_and_refusal() -> None:
    receipt = decode_catalog_receipt(
        {
            "host": "wrds",
            "scheduler": "grid-engine",
            "job_id": "1",
            "source_sha256": "a",
            "catalog_sha256": "b",
            "remote_catalog": "/home/x",
            "scope": "s",
        }
    )
    assert decode_catalog_receipt(encode_catalog_receipt(receipt)) == receipt
    with pytest.raises(JSONTypeError):
        decode_catalog_receipt({"host": "wrds"})


@pytest.mark.parametrize("existing", ["out", "out.receipt.json"])
def test_existing_artifacts_are_not_overwritten(
    existing: str, tmp_path: pathlib.Path, fake_run: FakeRun
) -> None:
    (tmp_path / existing).write_text("keep")
    with pytest.raises(FileExistsError):
        inventory("wrds", "/home/uci/test", tmp_path / "out")
    assert (tmp_path / existing).read_text() == "keep"
    assert fake_run.commands() == []


@pytest.mark.parametrize("bad", ["source", "catalog"])
def test_digest_mismatch_refuses_local_success(
    bad: str,
    tmp_path: pathlib.Path,
    fake_run: FakeRun,
) -> None:
    prepare(
        fake_run,
        source_sha="0" * 64 if bad == "source" else None,
        catalog_sha="0" * 64 if bad == "catalog" else None,
    )
    with pytest.raises(AppError):
        inventory("wrds", "/home/uci/test", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_failed_job_does_not_copy_a_partial_catalog(
    tmp_path: pathlib.Path, fake_run: FakeRun
) -> None:
    fake_run.add("qsub", returncode=1, stderr="job exited 1")
    prepare(fake_run)
    with pytest.raises(AppError, match="job exited 1"):
        inventory("wrds", "/home/uci/test", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_cli_posts_its_own_load_report(
    tmp_path: pathlib.Path,
    fake_run: FakeRun,
    emitted: list[str],
) -> None:
    prepare(fake_run)
    post = accepting_board(2)
    argv = ["--host", "wrds", "--remote-dir", "/home/uci/test", "--output", str(tmp_path / "out")]
    assert cli.main([*argv, "--report-task", TASK]) == 0
    report = (
        f"WRDS FILESYSTEM LOAD-REPORT job 12345 on wrds (grid-engine): catalog sha256 "
        f"{CATALOG_SHA}; 2 visible files, 1 readable files, 42 readable logical bytes, "
        "1 inaccessible directories, 1 unresolved links; "
        f"probe sha256 {SOURCE_SHA}. A visible file is not a readable one, "
        "and neither is a SQL privilege. Metadata only; no dataset rows."
    )
    assert emitted[0] == report
    assert emitted[-1] == f"Posted on board task {TASK}"
    assert post.urls == [DECLARED_TASKBOARD_URL, DECLARED_TASKBOARD_URL]
    checkin, note = (sent_arguments(body) for body in post.bodies)
    assert checkin["kind"] == "checkin"
    assert checkin["harness"] == "hpc3-wrds-inventory"
    assert note == {
        "room": "main",
        "taskId": TASK,
        "kind": "note",
        "body": report,
        "agent": "load-report-c4522545",
        "sessionId": checkin["sessionId"],
        "cwd": "service://hpc3-wrds-inventory",
    }


def test_cli_refuses_a_task_prefix_before_submitting(
    tmp_path: pathlib.Path, fake_run: FakeRun
) -> None:
    argv = ["--host", "wrds", "--remote-dir", "/home/uci/test", "--output", str(tmp_path / "out")]
    with pytest.raises(ValueError, match="WRDS_REPORT_TASK"):
        cli.main([*argv, "--report-task", "c4522545"])
    assert fake_run.commands() == []


def test_cli_entrypoint_rejects_missing_flags() -> None:
    original = sys.argv
    sys.argv = ["wrds_catalog"]
    with pytest.raises(SystemExit) as caught:
        runpy.run_module("hpc3.cli.wrds_catalog", run_name="__main__")
    sys.argv = original
    assert caught.value.code != 0


def test_all_record_kinds_decode() -> None:
    for kind in ("alias", "unresolved_link", "special"):
        value: list[JSONValue] = [kind, "/wrds/example", 0, False, "/target"]
        assert decode_catalog_entry(value)["kind"] == kind
