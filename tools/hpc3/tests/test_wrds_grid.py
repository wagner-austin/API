"""The WRDS Grid Engine transport both inventories share."""

from __future__ import annotations

import hashlib
import pathlib

import pytest
from platform_core.errors import AppError

from hpc3.core import artifacts, wrds_grid
from tests.conftest import FakeRun

REMOTE = pathlib.PurePosixPath("/home/uci/test/run one")


@pytest.mark.parametrize("remote", ["relative", "/tmp/wrds", "/home/../wrds"])
def test_remote_directory_must_be_beneath_home(remote: str) -> None:
    with pytest.raises(ValueError, match="WRDS_CATALOG_PATH"):
        wrds_grid.remote_directory(remote)


def test_remote_directory_accepts_a_home_path() -> None:
    assert wrds_grid.remote_directory("/home/uci/test/run one") == REMOTE


def test_receipt_sits_beside_its_artifact(tmp_path: pathlib.Path) -> None:
    assert wrds_grid.receipt_path(tmp_path / "relations.csv") == tmp_path / (
        "relations.csv.receipt.json"
    )


def test_refuse_existing_names_the_first_existing_file(tmp_path: pathlib.Path) -> None:
    wrds_grid.refuse_existing([tmp_path / "a", tmp_path / "b"])
    (tmp_path / "b").write_text("keep")
    with pytest.raises(FileExistsError, match=r"/b$"):
        wrds_grid.refuse_existing([tmp_path / "a", tmp_path / "b"])
    assert (tmp_path / "b").read_text() == "keep"


def test_create_directory_is_private_and_new(fake_run: FakeRun) -> None:
    wrds_grid.create_directory("wrds", REMOTE)
    assert fake_run.commands() == ["mkdir -m 700 -- '/home/uci/test/run one'"]


def test_create_directory_refuses_a_reused_directory(fake_run: FakeRun) -> None:
    fake_run.add("mkdir", returncode=1, stderr="mkdir: cannot create directory: File exists")
    with pytest.raises(AppError, match="File exists"):
        wrds_grid.create_directory("wrds", REMOTE)


def test_upload_returns_the_digest_the_remote_confirms(fake_run: FakeRun) -> None:
    payload = b"SELECT 1;\n"
    digest = hashlib.sha256(payload).hexdigest()
    fake_run.add("sha256sum", stdout=f"{digest}  /home/uci/test/q.sql\n")
    assert wrds_grid.upload("wrds", "/home/uci/test/q.sql", payload) == digest
    assert fake_run.calls[0].stdin_bytes == payload
    assert fake_run.commands() == [
        "cat > '/home/uci/test/q.sql'",
        "sha256sum '/home/uci/test/q.sql'",
    ]


def test_upload_refuses_bytes_that_arrived_altered(fake_run: FakeRun) -> None:
    fake_run.add("sha256sum", stdout="0" * 64 + "  /home/uci/test/q.sql\n")
    with pytest.raises(AppError):
        wrds_grid.upload("wrds", "/home/uci/test/q.sql", b"SELECT 1;\n")


@pytest.mark.parametrize(("environment", "exported"), [(True, True), (False, False)])
def test_submit_waits_bounded_and_exports_only_when_asked(
    fake_run: FakeRun, environment: bool, exported: bool
) -> None:
    fake_run.add("qsub", stdout="40529817\nJob 40529817 exited with exit code 0.\n")
    job = wrds_grid.submit("wrds", REMOTE, "probe", ["psql", "-f", "x"], environment=environment)
    assert job == "40529817"
    command = fake_run.commands()[0]
    assert command.startswith("qsub -terse -sync y ")
    assert ("-V" in command.split()) is exported
    assert "-l h_rt=3600" in command
    assert "-o '/home/uci/test/run one/probe.out'" in command
    assert "-e '/home/uci/test/run one/probe.err'" in command
    assert command.endswith("psql -f x")


@pytest.mark.parametrize("response", ["", "not a scheduler receipt"])
def test_submit_refuses_an_unreadable_scheduler_receipt(fake_run: FakeRun, response: str) -> None:
    fake_run.add("qsub", stdout=response)
    with pytest.raises(ValueError, match="WRDS_JOB_ID"):
        wrds_grid.submit("wrds", REMOTE, "probe", ["true"], environment=False)


def test_submit_propagates_a_failed_job(fake_run: FakeRun) -> None:
    fake_run.add("qsub", returncode=1, stdout="7\n", stderr="Job 7 exited with exit code 3.")
    with pytest.raises(AppError, match="exit code 3"):
        wrds_grid.submit("wrds", REMOTE, "probe", ["false"], environment=True)


def test_download_returns_the_verified_text(fake_run: FakeRun) -> None:
    text = "a,b\n"
    digest = hashlib.sha256(text.encode()).hexdigest()
    fake_run.add("cat --", stdout=text)
    fake_run.add("sha256sum", stdout=f"{digest}  /home/uci/test/out.csv\n")
    assert artifacts.download("wrds", "/home/uci/test/out.csv") == artifacts.Artifact(
        text=text, sha256=digest
    )


def test_download_refuses_a_text_that_differs_from_the_remote_file(fake_run: FakeRun) -> None:
    fake_run.add("cat --", stdout="truncated")
    fake_run.add("sha256sum", stdout="0" * 64 + "  /home/uci/test/out.csv\n")
    with pytest.raises(AppError):
        artifacts.download("wrds", "/home/uci/test/out.csv")


def test_verified_artifact_codec_checks_content_and_exact_fields() -> None:
    text = "a,b\n"
    value = artifacts.Artifact(text=text, sha256=hashlib.sha256(text.encode()).hexdigest())
    assert artifacts.decode_artifact(artifacts.encode_artifact(value)) == value
    with pytest.raises(ValueError, match="HPC_ARTIFACT_FIELDS"):
        artifacts.decode_artifact({"text": text})
    with pytest.raises(AppError, match="digests"):
        artifacts.decode_artifact({"text": text, "sha256": "0" * 64})
