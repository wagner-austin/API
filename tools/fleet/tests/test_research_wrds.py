"""WRDS intent recovery and native-bound collected evidence through real files."""

from __future__ import annotations

import hashlib
import pathlib

import pytest
from platform_core.json_utils import JSONObject, dump_json_str, require_dict
from platform_core.run_record import encode_run_record, run_record
from platform_core.testing import sample_run_fingerprint

from fleet.core import _test_hooks, research_wrds
from tests.conftest import FakeRun, ok


def _document(root: pathlib.Path, *, adopt: bool) -> JSONObject:
    """Create a pinned native specification.

    Args:
        root: Private store.
        adopt: Whether a native attempt already exists.

    Returns:
        Private controller document.
    """
    spec = root / "spec.json"
    spec.write_text("{}")
    return {
        "tenant": "ours",
        "backend": "wrds",
        "attempt": "pilot",
        "python": "/runtime/python",
        "spec": {"path": str(spec), "sha256": hashlib.sha256(b"{}").hexdigest()},
        "output": str(root / "collected"),
        "adopt": adopt,
    }


def _completion(root: pathlib.Path) -> JSONObject:
    """Write actual collection bytes and their metadata.

    Args:
        root: Private store.

    Returns:
        Completion manifest to compare against native measurement.
    """
    output = root / "collected"
    (output / "outputs").mkdir(parents=True)
    (output / "outputs/data").write_bytes(b"private data")
    record = encode_run_record(run_record("pilot", "one", sample_run_fingerprint(), (), "a" * 64))
    (output / "results.runrecord.json").write_text(dump_json_str(record))
    completion: JSONObject = {
        "job_id": "123",
        "attempt": "pilot",
        "identity": "identity",
        "outputs": {"data": hashlib.sha256(b"private data").hexdigest()},
        "record": record,
    }
    (output / "completion.json").write_text(dump_json_str(completion))
    return completion


@pytest.mark.parametrize("state", ["r", "hqw", "accounting-pending", "failed"])
def test_submission_intent_is_durable_and_never_replays_submit(
    tmp_path: pathlib.Path, state: str
) -> None:
    status = ok(dump_json_str({"job_id": "123", "state": state}))
    replies = [ok("{}"), status]
    if state == "hqw":
        replies.append(ok("{}"))
    replies.append(status)
    if state == "hqw":
        replies.append(ok("{}"))
    runner = FakeRun(replies)
    _test_hooks.run = runner
    document = _document(tmp_path, adopt=False)
    expected: JSONObject = {
        "state": "failed" if state == "failed" else "running",
        "native_job_id": "123",
        "record_sha256": None,
    }
    assert research_wrds.poll(document, tmp_path / "transcript", "task") == expected
    assert research_wrds.poll(document, tmp_path / "transcript", "task") == expected
    assert [call[3] for call in runner.calls].count("submit") == 1
    assert [call[3] for call in runner.calls].count("resume") == (2 if state == "hqw" else 0)
    assert (tmp_path / "submission.intent").read_text().startswith("Native submission")


def test_lost_submit_response_leaves_intent_for_reconciliation(tmp_path: pathlib.Path) -> None:
    failed = ok("{}")
    failed["returncode"] = 255
    runner = FakeRun([failed, ok('{"job_id":"123","state":"r"}')])
    _test_hooks.run = runner
    document = _document(tmp_path, adopt=False)
    with pytest.raises(ValueError, match="RESEARCH_CONTROLLER_PROCESS"):
        research_wrds.poll(document, tmp_path / "transcript", "task")
    assert research_wrds.poll(document, tmp_path / "transcript", "task")["native_job_id"] == "123"
    assert [call[3] for call in runner.calls] == ["submit", "status"]


@pytest.mark.parametrize("adopt", [True, False])
def test_submission_collection_and_verification_use_separate_bounded_polls(
    tmp_path: pathlib.Path, adopt: bool
) -> None:
    document = _document(tmp_path, adopt=adopt)
    status = ok('{"job_id":"123","state":"completed"}')
    replies = [] if adopt else [ok("{}"), status]
    replies.extend([status, ok("{}")])
    runner = FakeRun(replies)
    _test_hooks.run = runner
    log = tmp_path / "transcript"
    expected: JSONObject = {"state": "running", "native_job_id": "123", "record_sha256": None}
    if not adopt:
        assert research_wrds.poll(document, log, "task") == expected
        assert [call[3] for call in runner.calls] == ["submit", "status"]
        assert runner.timeouts == [300, 60]
    assert research_wrds.poll(document, log, "task") == expected
    assert [call[3] for call in runner.calls] == (
        ["status", "collect"] if adopt else ["submit", "status", "status", "collect"]
    )
    assert runner.timeouts[-2:] == [60, 900]
    # The collector's real durable files, not its process response, authorize success.
    completion = _completion(tmp_path)
    digest = hashlib.sha256(dump_json_str(completion).encode()).hexdigest()
    verifier = FakeRun(
        [status, ok(dump_json_str({"identity": "identity", "completion_sha256": digest}))]
    )
    _test_hooks.run = verifier
    assert research_wrds.poll(document, log, "task") == {
        "state": "complete",
        "native_job_id": "123",
        "record_sha256": hashlib.sha256(
            (tmp_path / "collected/results.runrecord.json").read_bytes()
        ).hexdigest(),
    }
    assert [call[3] for call in verifier.calls] == ["status", "report"]
    assert verifier.timeouts == [60, 300]
    assert (tmp_path / "submission.intent").exists() is not adopt


def test_completion_is_reverified_and_credentials_are_not_logged(
    tmp_path: pathlib.Path,
) -> None:
    document = _document(tmp_path, adopt=True)
    completion = _completion(tmp_path)
    digest = hashlib.sha256(dump_json_str(completion).encode()).hexdigest()
    replies = [ok('{"job_id":"123","state":"completed"}')]
    replies.append(ok(dump_json_str({"identity": "identity", "completion_sha256": digest})))
    runner = FakeRun(replies)
    _test_hooks.run = runner
    log = tmp_path / "transcript"
    result = research_wrds.poll(document, log, "task", (("PRIVATE_TOKEN", "secret-value"),))
    assert result == {
        "state": "complete",
        "native_job_id": "123",
        "record_sha256": hashlib.sha256(
            (tmp_path / "collected/results.runrecord.json").read_bytes()
        ).hexdigest(),
    }
    assert [call[3] for call in runner.calls] == ["status", "report"]
    assert all(value == (("PRIVATE_TOKEN", "secret-value"),) for value in runner.set_env)
    assert "secret-value" not in log.read_text()
    assert not (tmp_path / "submission.intent").exists()


@pytest.mark.parametrize("missing", ["completion.json", "results.runrecord.json", "outputs/data"])
def test_partial_collection_is_preserved_and_cannot_close_success(
    tmp_path: pathlib.Path, missing: str
) -> None:
    completion = _completion(tmp_path)
    digest = hashlib.sha256(dump_json_str(completion).encode()).hexdigest()
    (tmp_path / "collected" / missing).unlink()
    runner = FakeRun(
        [
            ok('{"job_id":"123","state":"completed"}'),
            ok(dump_json_str({"identity": "identity", "completion_sha256": digest})),
        ]
    )
    _test_hooks.run = runner
    with pytest.raises(FileNotFoundError):
        research_wrds.poll(_document(tmp_path, adopt=True), tmp_path / "transcript", "task")
    assert [call[3] for call in runner.calls] == ["status", "report"]
    assert not (tmp_path / "collected" / missing).exists()


@pytest.mark.parametrize(
    "drift", ["native", "job_id", "attempt", "identity", "record", "data", "escape"]
)
def test_changed_collection_cannot_close_success(tmp_path: pathlib.Path, drift: str) -> None:
    completion = _completion(tmp_path)
    if drift in {"job_id", "attempt", "identity"}:
        completion[drift] = "different"
    if drift == "record":
        require_dict(completion, "record")["label"] = "different"
    if drift == "escape":
        completion["outputs"] = {"../../outside": "a" * 64}
    if drift == "data":
        (tmp_path / "collected/outputs/data").write_text("changed")
    (tmp_path / "collected/completion.json").write_text(dump_json_str(completion))
    digest = (
        "a" * 64
        if drift == "native"
        else hashlib.sha256(dump_json_str(completion).encode()).hexdigest()
    )
    _test_hooks.run = FakeRun(
        [
            ok('{"job_id":"123","state":"completed"}'),
            ok(dump_json_str({"identity": "identity", "completion_sha256": digest})),
        ]
    )
    with pytest.raises(ValueError, match=r"RESEARCH_WRDS_(RECEIPT|ARTIFACT)"):
        research_wrds.poll(_document(tmp_path, adopt=True), tmp_path / "transcript", "task")


def test_unrecognized_fields_refuse_before_running(tmp_path: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="RESEARCH_WRDS_FIELDS"):
        research_wrds.poll({"shell": "forbidden"}, tmp_path / "transcript", "task")
