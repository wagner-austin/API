"""HPC polling reuses the admission journal and verifies terminal native evidence."""

from __future__ import annotations

import hashlib
import pathlib

import pytest
from platform_core.json_utils import JSONObject, dump_json_str, require_dict
from platform_core.run_record import encode_run_record, run_record
from platform_core.testing import sample_run_fingerprint

from fleet.core import _test_hooks, research_hpc
from tests.conftest import FakeRun, ok


def _document(directory: pathlib.Path) -> JSONObject:
    """Make real pinned controller input files for one journal attempt.

    Args:
        directory: Test-owned private directory.

    Returns:
        Specification whose files are read through the real file seam.
    """
    document: JSONObject = {
        "tenant": "ours",
        "backend": "hpc3",
        "attempt": "trial-1",
        "python": "/runtime/python",
        "ledger": str(directory / "journal.sqlite"),
        "directory": str(directory / "work"),
        "native_attempt": "policy-trial-1",
    }
    for key, content in (
        ("script", "# installed controller"),
        ("policy", "{}"),
        ("trial", '{"number":1}'),
        ("cluster", '{"workspace":"/shared/hpc3.json"}'),
    ):
        path = directory / key
        path.write_text(content)
        document[key] = {"path": str(path), "sha256": hashlib.sha256(content.encode()).hexdigest()}
    return document


def _row(state: str) -> JSONObject:
    """Encode a journal row without substituting another RunRecord shape.

    Args:
        state: Lifecycle state to observe.

    Returns:
        The controller's metadata fields this adapter consumes.
    """
    record = run_record("pilot", "candidate-1", sample_run_fingerprint(), (), "a" * 64)
    return {
        "number": 1,
        "status": state,
        "job_id": "12345",
        "record": dump_json_str(encode_run_record(record)),
    }


@pytest.mark.parametrize(
    "state, action", [("admitted", "submit"), ("submitting", "reconcile"), ("running", "reconcile")]
)
def test_only_admitted_work_submits_and_uncertain_work_reconciles(
    tmp_path: pathlib.Path, state: str, action: str
) -> None:
    runner = FakeRun(
        [
            ok(dump_json_str({"attempts": [_row(state)]})),
            ok(dump_json_str({"attempts": [_row("running")]})),
        ]
    )
    _test_hooks.run = runner
    result = research_hpc.poll(_document(tmp_path), tmp_path / "transcript", "board-task")
    assert result == {"state": "running", "native_job_id": "12345", "record_sha256": None}
    assert runner.calls[0][-1] == "status"
    assert runner.calls[1][6] == action
    assert runner.timeouts == [60, 300]


@pytest.mark.parametrize("state", ["failed", "timeout", "interrupted", "refused"])
def test_failed_attempt_never_resubmits(tmp_path: pathlib.Path, state: str) -> None:
    runner = FakeRun([ok(dump_json_str({"attempts": [_row(state)]}))])
    _test_hooks.run = runner
    assert research_hpc.poll(_document(tmp_path), tmp_path / "transcript", "task") == {
        "state": "failed",
        "native_job_id": "12345",
        "record_sha256": None,
    }
    assert len(runner.calls) == 1


@pytest.mark.parametrize("drift", ["none", "job", "record"])
def test_complete_attempt_requires_fresh_native_identity_and_record(
    tmp_path: pathlib.Path, drift: str
) -> None:
    from platform_core.json_utils import load_json_str, narrow_json_to_dict, require_str

    row = _row("complete")
    record = narrow_json_to_dict(load_json_str(require_str(row, "record")))
    original_digest = hashlib.sha256(dump_json_str(record).encode()).hexdigest()
    if drift == "record":
        record["label"] = "different"
    runner = FakeRun(
        [
            ok(dump_json_str({"attempts": [row]})),
            ok(
                dump_json_str(
                    {"ledger": {"job_id": "999" if drift == "job" else "12345"}, "record": record}
                )
            ),
        ]
    )
    _test_hooks.run = runner
    document = _document(tmp_path)
    if drift != "none":
        with pytest.raises(ValueError, match="RESEARCH_RECEIPT_DRIFT"):
            research_hpc.poll(document, tmp_path / "transcript", "task")
    else:
        assert research_hpc.poll(document, tmp_path / "transcript", "task") == {
            "state": "complete",
            "native_job_id": "12345",
            "record_sha256": original_digest,
        }
    assert runner.calls[1][2] == "hpc3.cli.execution_inspect"
    assert runner.calls[1][-2:] == ("--report-task", "task")


@pytest.mark.parametrize("rows", [[], [_row("running"), _row("running")]])
def test_missing_or_ambiguous_admission_refuses(
    tmp_path: pathlib.Path, rows: list[JSONObject]
) -> None:
    _test_hooks.run = FakeRun([ok(dump_json_str({"attempts": list(rows)}))])
    with pytest.raises(ValueError, match="RESEARCH_ATTEMPT_MISSING"):
        research_hpc.poll(_document(tmp_path), tmp_path / "transcript", "task")


def test_unknown_fields_and_states_and_nonpositive_trial_refuse(tmp_path: pathlib.Path) -> None:
    document = _document(tmp_path)
    with pytest.raises(ValueError, match="RESEARCH_HPC_FIELDS"):
        research_hpc.poll({**document, "shell": "forbidden"}, tmp_path / "transcript", "task")
    _test_hooks.run = FakeRun([ok(dump_json_str({"attempts": [_row("unrecognized")]}))])
    with pytest.raises(ValueError, match="RESEARCH_CONTROLLER_STATE"):
        research_hpc.poll(document, tmp_path / "transcript", "task")
    trial = tmp_path / "trial"
    trial.write_text('{"number":0}')
    require_dict(document, "trial")["sha256"] = hashlib.sha256(trial.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="RESEARCH_ATTEMPT_MISSING"):
        research_hpc.poll(document, tmp_path / "transcript", "task")
