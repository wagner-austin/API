"""Catalog validation and the WRDS Grid Engine transport contract."""

from __future__ import annotations

import hashlib
import pathlib
import runpy
import sys

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import JSONTypeError, JSONValue

from hpc3.cli import wrds_catalog as cli
from hpc3.contracts.wrds_catalog import decode_catalog_entry, encode_catalog_entry
from hpc3.core.wrds_catalog import inventory, probe_archive, read_catalog
from tests.conftest import FakeRun

CATALOG = (
    '["directory", "/wrds", 0, true, "/wrdslin"]\n'
    '["file", "/wrds/comp/sasdata/funda.sas7bdat", 42, true, "/data/funda"]\n'
    '["file", "/wrds/other/table.sas7bdat", 8, false, "/data/table"]\n'
    '["denied_directory", "/wrds/secret", 0, false, "/data/secret"]\n'
    '["complete", "/wrds", 0, false, ""]\n'
)


def prepare(
    fake_run: FakeRun, *, source_sha: str | None = None, catalog_sha: str | None = None
) -> None:
    """Supply scheduler and digest replies for the transport contract.

    Args:
        fake_run: Existing process seam fixture.
        source_sha: Explicit digest for a corrupt upload test.
        catalog_sha: Explicit digest for a corrupt download test.
    """
    source = probe_archive()
    fake_run.add(
        "sha256sum",
        stdout=(hashlib.sha256(source).hexdigest() if source_sha is None else source_sha)
        + "  scan.py\n",
        once=True,
    )
    fake_run.add(
        "sha256sum",
        stdout=(
            hashlib.sha256(CATALOG.encode()).hexdigest() if catalog_sha is None else catalog_sha
        )
        + "  catalog.jsonl\n",
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
    entries = inventory("wrds", "/home/uci/awagner14/catalog one", output)
    assert entries == read_catalog(CATALOG)
    assert output.read_text() == CATALOG
    receipt = output.with_suffix(".jsonl.receipt.json").read_text()
    assert '"job_id":"12345"' in receipt
    assert '"scheduler":"grid-engine"' in receipt
    assert hashlib.sha256(CATALOG.encode()).hexdigest() in receipt
    commands = fake_run.commands()
    assert commands[0] == "mkdir -m 700 -- '/home/uci/awagner14/catalog one'"
    submitted = [command for command in commands if command.startswith("qsub ")]
    assert len(submitted) == 1
    assert "-sync y -b y" in submitted[0]
    assert "h_rt=3600" in submitted[0]
    assert "--root /wrds" in submitted[0]


@pytest.mark.parametrize("remote", ["relative", "/tmp/wrds", "/home/../wrds"])
def test_remote_path_validation(remote: str, tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="WRDS_CATALOG_PATH"):
        inventory("wrds", remote, tmp_path / "out")


@pytest.mark.parametrize("existing", ["out", "out.receipt.json"])
def test_existing_artifacts_are_not_overwritten(existing: str, tmp_path: pathlib.Path) -> None:
    (tmp_path / existing).write_text("keep")
    with pytest.raises(FileExistsError):
        inventory("wrds", "/home/uci/test", tmp_path / "out")
    assert (tmp_path / existing).read_text() == "keep"


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


@pytest.mark.parametrize("response", ["", "not a scheduler receipt"])
def test_invalid_scheduler_receipt(
    response: str, tmp_path: pathlib.Path, fake_run: FakeRun
) -> None:
    fake_run.add("qsub", stdout=response)
    prepare(fake_run)
    with pytest.raises(ValueError, match="WRDS_JOB_ID"):
        inventory("wrds", "/home/uci/test", tmp_path / "out")


def test_failed_job_does_not_copy_a_partial_catalog(
    tmp_path: pathlib.Path, fake_run: FakeRun
) -> None:
    fake_run.add("qsub", returncode=1, stderr="job exited 1")
    prepare(fake_run)
    with pytest.raises(AppError, match="job exited 1"):
        inventory("wrds", "/home/uci/test", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_cli_reports_metadata_only(
    tmp_path: pathlib.Path,
    fake_run: FakeRun,
    emitted: list[str],
) -> None:
    prepare(fake_run)
    assert (
        cli.main(
            ["--host", "wrds", "--remote-dir", "/home/uci/test", "--output", str(tmp_path / "out")]
        )
        == 0
    )
    assert emitted[:3] == [
        "WRDS files: 2 visible, 1 readable",
        "Readable logical bytes: 42",
        "Inaccessible directories: 1; SQL catalog not inventoried",
    ]


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
