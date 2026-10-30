"""The evaluator's proposer-to-admission path; the host twin runs the real daemon."""

from __future__ import annotations

import hashlib
import pathlib
import runpy
import sys

import pytest
from platform_core.json_utils import JSONObject, dump_json_str, load_json_str

from fleet.cli import research_propose
from fleet.core import _test_hooks, research_proposal
from tests.conftest import FakeRun, failed, ok
from tests.test_research_proposer import IMAGE, SOCKET, replies

PROPOSAL: JSONObject = {"partition": "validation", "seed": 11}
PROGRAM = "import json, sys; print(json.dumps(json.loads(sys.argv[1])['proposal']))"


def _pinned(path: pathlib.Path, text: str) -> JSONObject:
    """Write one evaluator input and pin its exact bytes."""
    path.write_text(text)
    return {"path": str(path), "sha256": hashlib.sha256(text.encode()).hexdigest()}


def specification(root: pathlib.Path) -> JSONObject:
    """An evaluator-owned proposer specification over real private files.

    Args:
        root: Private evaluator directory.

    Returns:
        The specification document.
    """
    return {
        "python": sys.executable,
        "script": _pinned(root / "research_proposals.py", "evaluator controller"),
        "policy": _pinned(root / "policy.json", '{"evaluator":"policy"}'),
        "ledger": str(root / "journal.sqlite"),
        "host": None,
        "socket": SOCKET,
        "image": IMAGE,
        "program": _pinned(root / "program.py", PROGRAM),
        "context": _pinned(root / "context.json", dump_json_str({"proposal": PROPOSAL})),
        "timeout": 10,
    }


def report(status: str, reason: str) -> str:
    """The admission controller's report for attempt 3."""
    row: JSONObject = {"number": 3, "status": status, "reason": reason, "candidate": "c" * 64}
    return dump_json_str({"policy_digest": "a" * 64, "attempts": [row]})


@pytest.mark.parametrize(
    ("result", "status", "reason"),
    [
        (ok(report("admitted", "")), "admitted", ""),
        (
            _test_hooks.CommandResult(
                returncode=2,
                stdout=report("refused", "holdout_forbidden"),
                stderr="",
                timed_out=False,
            ),
            "refused",
            "holdout_forbidden",
        ),
    ],
)
def test_isolated_output_is_recorded_by_the_maintained_admission(
    tmp_path: pathlib.Path, result: _test_hooks.CommandResult, status: str, reason: str
) -> None:
    document = specification(tmp_path)
    fake = FakeRun([*replies(dump_json_str(PROPOSAL)), result])
    _test_hooks.run = fake
    log = tmp_path / "admission.jsonl"
    outcome = research_proposal.propose(document, tmp_path, log)
    digest = hashlib.sha256(dump_json_str(PROPOSAL).encode()).hexdigest()
    assert outcome == {
        "number": 3,
        "status": status,
        "reason": reason,
        "candidate": "c" * 64,
        "proposal_sha256": digest,
    }
    proposal = tmp_path / "proposals" / f"{digest}.json"
    assert load_json_str(proposal.read_text()) == PROPOSAL
    assert fake.calls[-1] == (
        sys.executable,
        str(tmp_path / "research_proposals.py"),
        "--policy",
        str(tmp_path / "policy.json"),
        "--ledger",
        str(tmp_path / "journal.sqlite"),
        "propose",
        "--proposal",
        str(proposal),
    )
    # The container receives the pinned program and context and no evaluator path.
    create = fake.calls[2]
    assert create[-2:] == (PROGRAM, dump_json_str({"proposal": PROPOSAL}))
    private = [str(tmp_path / name) for name in ("research_proposals.py", "policy.json")]
    for call in fake.calls[:-1]:
        assert not any(item in arg for item in [*private, "journal.sqlite"] for arg in call)
    assert len(log.read_text().splitlines()) == 1


@pytest.mark.parametrize("attempts", [0, 2])
def test_admission_must_report_exactly_one_attempt(tmp_path: pathlib.Path, attempts: int) -> None:
    rows = [load_json_str(report("admitted", ""))] * attempts
    answer = ok(dump_json_str({"attempts": rows}))
    _test_hooks.run = FakeRun([*replies(dump_json_str(PROPOSAL)), answer])
    with pytest.raises(ValueError, match="RESEARCH_PROPOSAL_REPORT"):
        research_proposal.propose(specification(tmp_path), tmp_path, tmp_path / "log.jsonl")


def test_controller_failure_propagates_with_its_transcript(tmp_path: pathlib.Path) -> None:
    _test_hooks.run = FakeRun([*replies(dump_json_str(PROPOSAL)), failed(1, "locked")])
    with pytest.raises(ValueError, match="RESEARCH_CONTROLLER_PROCESS: exit 1"):
        research_proposal.propose(specification(tmp_path), tmp_path, tmp_path / "log.jsonl")


@pytest.mark.parametrize("case", ["fields", "program", "context", "policy"])
def test_unpinned_or_changed_evaluator_input_runs_nothing(
    tmp_path: pathlib.Path, case: str
) -> None:
    document = specification(tmp_path)
    error = "RESEARCH_INPUT_DRIFT"
    if case == "fields":
        document["network"] = "host"
        error = "RESEARCH_PROPOSAL_FIELDS"
    else:
        (tmp_path / f"{case}.{'py' if case == 'program' else 'json'}").write_text("changed")
    fake = FakeRun([])
    _test_hooks.run = fake
    with pytest.raises(ValueError, match=error):
        research_proposal.propose(document, tmp_path, tmp_path / "log.jsonl")
    assert fake.calls == []
    assert not (tmp_path / "proposals").exists()


def test_isolation_refusal_reaches_no_admission(tmp_path: pathlib.Path) -> None:
    values = replies()
    fake = FakeRun([*values[:4], ok("7"), *values[6:]])
    _test_hooks.run = fake
    with pytest.raises(ValueError, match="RESEARCH_PROPOSER_EXIT"):
        research_proposal.propose(specification(tmp_path), tmp_path, tmp_path / "log.jsonl")
    assert all(call[0] == "/usr/bin/env" for call in fake.calls)
    assert not (tmp_path / "proposals").exists()


def test_cli_resolves_the_pinned_specification_and_emits_metadata(
    tmp_path: pathlib.Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = dump_json_str(specification(tmp_path)).encode()
    digest = hashlib.sha256(payload).hexdigest()
    (tmp_path / digest).mkdir()
    (tmp_path / digest / "proposal.json").write_bytes(payload)
    _test_hooks.run = FakeRun([*replies(dump_json_str(PROPOSAL)), ok(report("admitted", ""))])
    assert research_propose.main(["--root", str(tmp_path), "--digest", digest]) == 0
    printed = load_json_str(capsys.readouterr().out)
    assert printed == {
        "number": 3,
        "status": "admitted",
        "reason": "",
        "candidate": "c" * 64,
        "proposal_sha256": hashlib.sha256(dump_json_str(PROPOSAL).encode()).hexdigest(),
    }
    assert (tmp_path / digest / "admission.jsonl").is_file()
    (tmp_path / digest / "proposal.json").write_text("{}")
    with pytest.raises(ValueError, match="RESEARCH_SPEC_DRIFT"):
        research_propose.main(["--root", str(tmp_path), "--digest", digest])


def test_module_entrypoint_refuses_missing_arguments() -> None:
    module = "fleet.cli.research_propose"
    saved = sys.argv
    loaded = sys.modules.pop(module)
    sys.argv = [module]
    try:
        with pytest.raises(SystemExit) as raised:
            runpy.run_module(module, run_name="__main__", alter_sys=False)
    finally:
        sys.argv = saved
        sys.modules[module] = loaded
    assert raised.value.code != 0
