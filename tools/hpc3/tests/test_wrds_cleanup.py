"""Bind failed-subject metadata to a same-node scheduler observation and report."""

from __future__ import annotations

import hashlib
import pathlib

import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
)
from platform_core.mcp_testing import sent_arguments

from hpc3.cli import wrds
from hpc3.contracts.wrds_run import WrdsRun, encode_wrds_run, execution_document
from hpc3.core import wrds_cleanup
from hpc3.core.wrds_bundle import worker_bundle
from hpc3.core.wrds_jobs import attempt_directory
from tests.conftest import FakeRun
from tests.test_wrds_scheduler import DONE, EMPTY
from tests.wrds_board import TASK, accepting_board
from tests.wrds_support import example
from tests.wrds_transport import metadata

HOST = "wrds-sas35-w.wharton.private"
PRIVATE = "/tmp/123.1.all.q/research-plaintext-canary"


def opened() -> JSONObject:
    """The metadata the maintained storage probe publishes before its hold."""
    return {
        "node": HOST,
        "private": PRIVATE,
        "scratch": "/tmp/123.1.all.q",
        "mode": 448,
        "files": 10,
        "manifest_sha256": "a" * 64,
        "scratch_mode": 493,
    }


def observed() -> JSONObject:
    """A subsequent same-node observation, bound to both job identifiers."""
    return {
        "node": HOST,
        "private": PRIVATE,
        "scratch": "/tmp/123.1.all.q",
        "job_id": "123",
        "observer_job_id": "456",
        "observed_at": "2026-10-31T03:00:00Z",
        "scratch_present": False,
        "private_present": False,
    }


def prepare(fake: FakeRun, run: WrdsRun, before: JSONObject, after: JSONObject) -> None:
    """Supply transport metadata while running the actual orchestration and codecs."""
    directory = attempt_directory(run, "one")
    metadata(fake, directory + "/run.json", dump_json_str(execution_document(run)))
    fake.add("cat " + directory + "/job.id", stdout="123\n")
    fake.add("qstat", stdout=EMPTY)
    fake.add("qacct -j 123", stdout=DONE.replace("exit_status 0", "exit_status 1"))
    fake.add("test -e", stdout="absent\n")
    metadata(fake, directory + "/outputs/opened.json", dump_json_str(before))
    archive = directory + "/scratch-evidence/worker.pyz"
    fake.add(
        "sha256sum '" + archive,
        stdout=hashlib.sha256(worker_bundle()).hexdigest() + "  " + archive + "\n",
    )
    fake.add("qsub", stdout="456\nJob 456 exited with exit code 0.\n")
    fake.add("qacct -j 456", stdout=DONE.replace("jobnumber 123", "jobnumber 456"))
    metadata(fake, directory + "/scratch-evidence/observation.json", dump_json_str(after))


def test_cli_posts_measured_evidence(
    tmp_path: pathlib.Path, fake_run: FakeRun, emitted: list[str]
) -> None:
    run = example(tmp_path)
    prepare(fake_run, run, opened(), observed())
    spec = tmp_path / "experiment.json"
    spec.write_text(dump_json_str(encode_wrds_run(run)))
    output = tmp_path / "evidence.json"
    board = accepting_board(2)
    assert (
        wrds.main(
            [
                "scratch",
                "--spec",
                str(spec),
                "--attempt",
                "one",
                "--output",
                str(output),
                "--report-task",
                TASK,
            ]
        )
        == 0
    )
    evidence = narrow_json_to_dict(load_json_str(output.read_text()))
    assert evidence["opened"] == opened()
    assert evidence["observation"] == observed()
    assert evidence["observer_source_sha256"] == hashlib.sha256(worker_bundle()).hexdigest()
    assert evidence["opened_sha256"] == hashlib.sha256(dump_json_str(opened()).encode()).hexdigest()
    assert (
        evidence["observation_sha256"]
        == hashlib.sha256(dump_json_str(observed()).encode()).hexdigest()
    )
    assert evidence["observer_accounting"] == {"hostname": HOST, "failed": "0", "exit_status": "0"}
    note = sent_arguments(board.bodies[1])
    assert note["taskId"] == TASK
    assert note["body"] == "WRDS SCRATCH LOAD-REPORT\n" + dump_json_str(evidence)
    assert note["agent"] == "load-report-" + TASK[:8]
    assert load_json_str(emitted[-1]) == {"report_task": TASK, "output": str(output)}
    commands = fake_run.commands()
    assert any("-l hostname=" + HOST in command for command in commands)
    with pytest.raises(FileExistsError):
        wrds_cleanup.measure(run, "one", output)


@pytest.mark.parametrize(
    "key,value",
    [
        ("node", "other"),
        ("scratch", "/tmp/other"),
        ("mode", 493),
        ("files", 0),
        ("unexpected", "must never reach report"),
    ],
)
def test_opened_receipt_must_match_native_private_storage(
    tmp_path: pathlib.Path, fake_run: FakeRun, key: str, value: JSONValue
) -> None:
    run = example(tmp_path)
    before = opened()
    before[key] = value
    prepare(fake_run, run, before, observed())
    with pytest.raises(ValueError, match="WRDS_SCRATCH_RECEIPT"):
        wrds_cleanup.measure(run, "one", tmp_path / "evidence.json")
    assert not any("qsub" in command for command in fake_run.commands())


@pytest.mark.parametrize(
    "key,value",
    [
        ("node", "other"),
        ("job_id", "999"),
        ("observer_job_id", "999"),
        ("private", "/tmp/other"),
        ("scratch", "/tmp/other"),
        ("scratch_present", 0),
        ("private_present", "false"),
        ("unexpected", "must never reach report"),
    ],
)
def test_wrong_or_malformed_observer_is_not_published(
    tmp_path: pathlib.Path, fake_run: FakeRun, key: str, value: JSONValue
) -> None:
    run = example(tmp_path)
    after = observed()
    after[key] = value
    prepare(fake_run, run, opened(), after)
    with pytest.raises(ValueError, match="WRDS_SCRATCH_OBSERVER"):
        wrds_cleanup.measure(run, "one", tmp_path / "evidence.json")
    assert not (tmp_path / "evidence.json").exists()


def test_successful_subject_is_not_a_forced_termination(
    tmp_path: pathlib.Path, fake_run: FakeRun
) -> None:
    run = example(tmp_path)
    fake_run.add("qacct -j 123", stdout=DONE)
    prepare(fake_run, run, opened(), observed())
    with pytest.raises(ValueError, match="WRDS_SCRATCH_STATE"):
        wrds_cleanup.measure(run, "one", tmp_path / "evidence.json")


def test_observer_accounting_must_match_node_and_success(
    tmp_path: pathlib.Path, fake_run: FakeRun
) -> None:
    run = example(tmp_path)
    fake_run.add(
        "qacct -j 456", stdout=DONE.replace("jobnumber 123", "jobnumber 456").replace(HOST, "other")
    )
    prepare(fake_run, run, opened(), observed())
    with pytest.raises(ValueError, match="WRDS_SCRATCH_OBSERVER"):
        wrds_cleanup.measure(run, "one", tmp_path / "evidence.json")
