"""Durable held submissions, interruption recovery and refusal before queueing."""

from __future__ import annotations

import hashlib
import pathlib

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import dump_json_str, load_json_str, narrow_json_to_dict

from hpc3.contracts.wrds_run import WrdsFile, encode_wrds_runtime, run_identity
from hpc3.core import wrds_jobs as jobs
from tests.conftest import FakeRun
from tests.test_wrds_scheduler import DONE, EMPTY, HELD
from tests.wrds_support import example
from tests.wrds_transport import NOW, prepare


def test_held_submission_is_recorded_before_release(
    tmp_path: pathlib.Path, fake_run: FakeRun
) -> None:
    run = example(tmp_path)
    directory = prepare(fake_run, run, tmp_path)
    ledger = tmp_path / "ledger.jsonl"
    assert jobs.submit(run, tmp_path, "one", ledger, NOW) == "123"
    events = [narrow_json_to_dict(load_json_str(line)) for line in ledger.read_text().splitlines()]
    assert [row["event"] for row in events] == ["preparing", "submitted-held", "released"]
    assert all(row["identity"] == run_identity(run) for row in events)
    commands = fake_run.commands()
    queued = [command for command in commands if command.startswith("qsub ")]
    assert len(queued) == 2
    assert queued[0].startswith("qsub -w v ")
    assert queued[1].startswith("qsub -h ")
    assert queued[1].endswith(" > " + directory + "/job.id")
    assert commands[-1] == "qrls 123"
    assert jobs.attempt_directory(run, "two") != directory


def test_runtime_and_source_drift_prevent_submission(
    tmp_path: pathlib.Path, fake_run: FakeRun
) -> None:
    run = example(tmp_path)
    runtime = encode_wrds_runtime(run["runtime"])
    runtime["version"] = "changed"
    fake_run.add(" inspect", stdout=dump_json_str(runtime))
    prepare(fake_run, run, tmp_path)
    with pytest.raises(ValueError, match="WRDS_RUNTIME_DRIFT"):
        jobs.submit(run, tmp_path, "one", tmp_path / "ledger", NOW)
    assert not any(command.startswith("qsub") for command in fake_run.commands())
    (tmp_path / "code/main.py").write_text("changed")
    with pytest.raises(ValueError, match="WRDS_CODE_DRIFT"):
        jobs.check_local_code(run, tmp_path)


def test_preflight_checks_input_readability(tmp_path: pathlib.Path, fake_run: FakeRun) -> None:
    run = example(tmp_path)
    run["inputs"] = [WrdsFile(path="/wrds/example/data", name="data", sha256="a" * 64)]
    prepare(fake_run, run, tmp_path)
    assert jobs.preflight(run, tmp_path) == run["runtime"]
    assert "test -r /wrds/example/data" in fake_run.commands()


def test_existing_attempt_is_not_resubmitted(tmp_path: pathlib.Path, fake_run: FakeRun) -> None:
    run = example(tmp_path)
    fake_run.add("mkdir -m 700", returncode=1, stderr="File exists")
    prepare(fake_run, run, tmp_path)
    with pytest.raises(AppError, match="File exists"):
        jobs.submit(run, tmp_path, "one", tmp_path / "ledger", NOW)
    assert not any(command.startswith("qsub") for command in fake_run.commands())


def test_lost_release_can_be_resumed_without_a_second_qsub(
    tmp_path: pathlib.Path, fake_run: FakeRun
) -> None:
    run = example(tmp_path)
    fake_run.add("qrls", returncode=255, stderr="connection lost", once=True)
    prepare(fake_run, run, tmp_path)
    ledger = tmp_path / "ledger"
    with pytest.raises(AppError, match="connection lost"):
        jobs.submit(run, tmp_path, "one", ledger, NOW)
    events = [narrow_json_to_dict(load_json_str(line)) for line in ledger.read_text().splitlines()]
    assert [row["event"] for row in events] == ["preparing", "submitted-held"]
    assert jobs.resume(run, "one", ledger, NOW) == "123"
    assert len([command for command in fake_run.commands() if command.startswith("qsub -h")]) == 1


def test_resume_refuses_a_completed_job(tmp_path: pathlib.Path, fake_run: FakeRun) -> None:
    run = example(tmp_path)
    fake_run.add("qstat", stdout=EMPTY)
    fake_run.add("qacct", stdout=DONE)
    prepare(fake_run, run, tmp_path)
    with pytest.raises(ValueError, match="WRDS_RESUME_STATE"):
        jobs.resume(run, "one", tmp_path / "ledger", NOW)
    assert "qrls 123" not in fake_run.commands()


def test_remote_spec_drift_refuses_status_identity(
    tmp_path: pathlib.Path, fake_run: FakeRun
) -> None:
    run = example(tmp_path)
    directory = jobs.attempt_directory(run, "one")
    fake_run.add("cat -- " + directory + "/run.json", stdout="{}")
    fake_run.add(
        "sha256sum '" + directory + "/run.json'",
        stdout=hashlib.sha256(b"{}").hexdigest() + "  file\n",
    )
    with pytest.raises(ValueError, match="WRDS_SPEC_DRIFT"):
        jobs.job_id(run, "one")


@pytest.mark.parametrize(
    ("xml", "expected"), [(EMPTY, "cancelled"), (HELD, "cancellation-pending")]
)
def test_cancel_records_observed_state(
    tmp_path: pathlib.Path, fake_run: FakeRun, xml: str, expected: str
) -> None:
    run = example(tmp_path)
    fake_run.add("qstat", stdout=xml)
    prepare(fake_run, run, tmp_path)
    ledger = tmp_path / "ledger"
    assert jobs.cancel(run, "one", ledger, NOW) == expected
    assert "qdel 123" in fake_run.commands()
    assert narrow_json_to_dict(load_json_str(ledger.read_text()))["event"] == expected


def test_probe_refuses_unsafe_roots_and_unexpected_directory(fake_run: FakeRun) -> None:
    with pytest.raises(ValueError, match="WRDS_ROOT"):
        jobs.inspect_runtime("wrds", "/tmp/probe", "/usr/bin/python3")
    fake_run.add("mktemp", stdout="/tmp/not-our-root\n")
    with pytest.raises(ValueError, match="WRDS_PROBE"):
        jobs.inspect_runtime("wrds", "/home/uci/probe", "/usr/bin/python3")


def test_large_code_is_refused(tmp_path: pathlib.Path) -> None:
    run = example(tmp_path)
    payload = b"x" * (100 * 1024 * 1024 + 1)
    (tmp_path / "code/main.py").write_bytes(payload)
    run["code"][0]["sha256"] = hashlib.sha256(payload).hexdigest()
    with pytest.raises(ValueError, match="WRDS_CODE_SIZE"):
        jobs.check_local_code(run, tmp_path)
