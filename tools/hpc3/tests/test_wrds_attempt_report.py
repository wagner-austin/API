"""An attempt's report states what collect would do with it, measured, never assumed."""

from __future__ import annotations

import hashlib
import pathlib

import pytest
from platform_core.json_utils import dump_json_str, load_json_str

from hpc3.contracts.wrds_run import run_identity
from hpc3.core.wrds_attempt_report import completion_presence, measure
from hpc3.core.wrds_jobs import attempt_directory
from hpc3.core.wrds_results import collect, verified_completion
from tests.conftest import FakeRun
from tests.test_wrds_results import actual_result
from tests.test_wrds_scheduler import DONE, HELD
from tests.wrds_transport import NOW, CopyTransport, completed

FAILED = DONE.replace("exit_status 0", "exit_status 1")


def test_completed_attempt_reports_what_collect_writes(tmp_path: pathlib.Path) -> None:
    run, result = actual_result(tmp_path)
    transport = CopyTransport(tmp_path / "outputs")
    completed(transport, run, result)
    body, completion_digest = measure(run, "one")
    lines = body.splitlines()
    destination = tmp_path / "collected"
    collect(run, "one", destination, tmp_path / "ledger", NOW)
    assert completion_digest == hashlib.sha256(
        dump_json_str(load_json_str((destination / "completion.json").read_text())).encode()
    ).hexdigest()
    metrics = hashlib.sha256((destination / "outputs/metrics.json").read_bytes()).hexdigest()
    record = hashlib.sha256((destination / "results.runrecord.json").read_bytes()).hexdigest()
    assert lines[0] == (
        "WRDS ATTEMPT LOAD-REPORT wrds-system-probe/seeded-sum attempt one: job 123 on wrds"
    )
    assert lines[1] == "identity " + run_identity(run)
    assert lines[2] == "code main.py sha256=" + run["code"][0]["sha256"]
    assert lines[3].startswith(
        f"runtime python={run['runtime']['python']} sha256={run['runtime']['sha256']} "
    )
    assert f" packages={len(run['runtime']['packages'])} pin_sha256=" in lines[3]
    assert lines[4:] == [
        "scheduler state=completed",
        "accounting job 123 hostname=wrds-sas35-w.wharton.private failed=0 exit_status=0",
        "output metrics.json sha256=" + metrics,
        "run-record sha256=" + record,
    ]


def test_failed_attempt_reports_the_refusal_and_a_missing_attestation(
    tmp_path: pathlib.Path,
) -> None:
    run, result = actual_result(tmp_path)
    numbers = hashlib.sha256(b"1\n2\n3\n").hexdigest()
    run["inputs"] = [
        {"path": "/home/uci/awagner14/numbers.txt", "name": "numbers", "sha256": numbers}
    ]
    transport = FakeRun()
    transport.add("qacct", stdout=FAILED)
    transport.add("test -e", stdout="absent\n")
    completed(transport, run, result)
    body, digest = measure(run, "one")
    assert digest is None
    lines = body.splitlines()
    presence = transport.commands()[-1]
    assert lines[1] == "identity " + run_identity(run)
    assert lines[3] == "input numbers sha256=" + numbers
    assert lines[-4:] == [
        "scheduler state=failed",
        "accounting job 123 hostname=wrds-sas35-w.wharton.private failed=0 exit_status=1",
        "collect refused: WRDS_NOT_COMPLETE: job 123 is failed",
        "completion.json absent",
    ]
    with pytest.raises(ValueError, match="WRDS_NOT_COMPLETE: job 123 is failed"):
        verified_completion(run, "one")
    path = attempt_directory(run, "one") + "/completion.json"
    assert f"test -e {path} && echo present || echo absent" in presence


def test_active_attempt_has_no_accounting_yet(tmp_path: pathlib.Path) -> None:
    run, result = actual_result(tmp_path)
    transport = FakeRun()
    transport.add("qstat", stdout=HELD)
    transport.add("test -e", stdout="present\n")
    completed(transport, run, result)
    body, digest = measure(run, "one")
    assert digest is None
    lines = body.splitlines()
    assert lines[-3:] == [
        "scheduler state=hqw",
        "collect refused: WRDS_NOT_COMPLETE: job 123 is hqw",
        "completion.json present",
    ]
    assert not any("qacct" in command for command in transport.commands())


def test_completed_attempt_with_a_mismatched_attestation_fails_the_report(
    tmp_path: pathlib.Path,
) -> None:
    run, result = actual_result(tmp_path)
    result["attempt"] = "other"
    completed(FakeRun(), run, result)
    with pytest.raises(ValueError, match="WRDS_RESULT_IDENTITY"):
        measure(run, "one")


def test_presence_refuses_an_unexpected_reply(fake_run: FakeRun) -> None:
    fake_run.add("test -e", stdout="maybe\n")
    with pytest.raises(ValueError, match="WRDS_PRESENCE"):
        completion_presence("wrds", "/home/uci/awagner14/run")
