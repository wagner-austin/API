"""Actual directory reads cannot confuse wrong-node or failed access with absence."""

from __future__ import annotations

import pathlib
import socket
import subprocess
import sys
from collections.abc import Generator

import pytest
from platform_core.json_utils import load_json_str, narrow_json_to_dict

from hpc3.core.wrds_bundle import worker_bundle
from hpc3.wrds import _test_hooks, worker
from hpc3.wrds.scratch import observe

NOW = "2026-10-31T03:00:00+00:00"


@pytest.fixture(autouse=True)
def reset_worker_hooks() -> Generator[None, None, None]:
    _test_hooks.reset_hooks()
    yield
    _test_hooks.reset_hooks()


@pytest.mark.parametrize("state", ["scratch-absent", "private-absent", "private-present"])
def test_observe_lists_real_paths(tmp_path: pathlib.Path, state: str) -> None:
    _test_hooks.environment = lambda: {"JOB_ID": "456"}
    _test_hooks.now_iso = lambda: NOW
    private = tmp_path / "123.1.all.q/research-plaintext-canary"
    if state != "scratch-absent":
        private.parent.mkdir()
    if state == "private-present":
        private.mkdir(mode=0o700)
        (private / "secret").write_text("never read this file")
    assert observe(socket.gethostname(), "123", private, tmp_path) == {
        "node": socket.gethostname(),
        "job_id": "123",
        "observer_job_id": "456",
        "observed_at": NOW,
        "matching_scratch_roots": [] if state == "scratch-absent" else [private.parent.name],
        "scratch": str(private.parent),
        "private": str(private),
        "scratch_present": state != "scratch-absent",
        "private_present": state == "private-present",
    }


@pytest.mark.parametrize("subject,observer", [("bad", "456"), ("123", "bad"), ("123", "123")])
def test_invalid_or_same_job_is_refused(
    tmp_path: pathlib.Path, subject: str, observer: str
) -> None:
    _test_hooks.environment = lambda: {"JOB_ID": observer}
    with pytest.raises(ValueError, match="WRDS_SCRATCH_JOB"):
        observe(
            socket.gethostname(), subject, tmp_path / "123.1.all.q/research-plaintext-x", tmp_path
        )


@pytest.mark.parametrize(
    "relative",
    [
        "other/123.1.all.q/research-plaintext-x",
        "124.1.all.q/research-plaintext-x",
        "123.1.all.q/not-private",
    ],
)
def test_path_must_name_subject_scratch(tmp_path: pathlib.Path, relative: str) -> None:
    _test_hooks.environment = lambda: {"JOB_ID": "456"}
    with pytest.raises(ValueError, match="WRDS_SCRATCH_PATH"):
        observe(socket.gethostname(), "123", tmp_path / relative, tmp_path)


def test_wrong_node_and_unreadable_directory_fail(tmp_path: pathlib.Path) -> None:
    _test_hooks.environment = lambda: {"JOB_ID": "456"}
    private = tmp_path / "123.1.all.q/research-plaintext-x"
    with pytest.raises(ValueError, match="WRDS_SCRATCH_HOST"):
        observe("not-the-execution-host", "123", private, tmp_path)
    private.parent.write_text("not a directory")
    with pytest.raises(NotADirectoryError):
        observe(socket.gethostname(), "123", private, tmp_path)
    with pytest.raises(FileNotFoundError):
        observe(
            socket.gethostname(),
            "123",
            tmp_path / "absent/123.1.all.q/research-plaintext-x",
            tmp_path / "absent",
        )


def test_completed_job_scope_lists_real_roots_without_opening_them(tmp_path: pathlib.Path) -> None:
    _test_hooks.environment = lambda: {"JOB_ID": "456"}
    _test_hooks.now_iso = lambda: NOW
    (tmp_path / "123.1.all.q").write_text("even a nondirectory remainder is recorded")
    (tmp_path / "123.1.other.q").symlink_to(tmp_path / "absent")
    (tmp_path / "1234.1.all.q").mkdir()
    assert observe(socket.gethostname(), "123", None, tmp_path) == {
        "node": socket.gethostname(),
        "job_id": "123",
        "observer_job_id": "456",
        "observed_at": NOW,
        "matching_scratch_roots": ["123.1.all.q", "123.1.other.q"],
    }
    with pytest.raises(FileNotFoundError):
        observe(socket.gethostname(), "123", None, tmp_path / "absent")


def test_real_archive_completed_scope_observes_absence(tmp_path: pathlib.Path) -> None:
    _test_hooks.environment = lambda: {"JOB_ID": "456"}
    direct = tmp_path / "direct.json"
    assert worker.main(["scratch", socket.gethostname(), "999999999999999999999", str(direct)]) == 0
    direct_reading = narrow_json_to_dict(load_json_str(direct.read_text()))
    assert direct_reading["matching_scratch_roots"] == []
    archive = tmp_path / "worker.pyz"
    archive.write_bytes(worker_bundle())
    output = tmp_path / "observation.json"
    result = subprocess.run(
        [
            sys.executable,
            str(archive),
            "scratch",
            socket.gethostname(),
            "999999999999999999999",
            str(output),
        ],
        env={"JOB_ID": "456"},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert (result.returncode, result.stdout, result.stderr) == (0, "", "")
    reading = narrow_json_to_dict(load_json_str(output.read_text()))
    assert reading["node"] == socket.gethostname()
    assert reading["job_id"] == "999999999999999999999"
    assert reading["observer_job_id"] == "456"
    assert reading["matching_scratch_roots"] == []


def test_worker_and_real_archive_refuse_wrong_node(tmp_path: pathlib.Path) -> None:
    _test_hooks.environment = lambda: {"JOB_ID": "456"}
    _test_hooks.now_iso = lambda: NOW
    private = "/tmp/999999999999999999999.1.all.q/research-plaintext-canary"
    output = tmp_path / "observation.json"
    assert (
        worker.main(
            ["scratch", socket.gethostname(), "999999999999999999999", private, str(output)]
        )
        == 0
    )
    assert load_json_str(output.read_text()) == {
        "node": socket.gethostname(),
        "job_id": "999999999999999999999",
        "observer_job_id": "456",
        "observed_at": NOW,
        "matching_scratch_roots": [],
        "scratch": str(pathlib.Path(private).parent),
        "private": private,
        "scratch_present": False,
        "private_present": False,
    }
    archive = tmp_path / "worker.pyz"
    archive.write_bytes(worker_bundle())
    environment = _test_hooks.environment()
    result = subprocess.run(
        [sys.executable, str(archive), "scratch", "wrong-host", "123", private, str(output)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert (result.returncode, result.stdout) == (1, "")
    assert result.stderr.splitlines()[-1] == (
        "ValueError: WRDS_SCRATCH_HOST: observer is not on the original execution node"
    )
