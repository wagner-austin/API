"""Remote placement preserves quoting, deadlines, cleanup and local admission."""

import pathlib
import shlex

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import JSONObject, dump_json_str

from fleet.core import _test_hooks, research_proposal
from fleet.core.linux_isolated_build import EXEC_USER
from fleet.core.research_proposer import run_proposer
from fleet.core.ssh_options import SSH_OPTIONS
from tests.conftest import FakeRun, failed, ok, timed_out
from tests.test_research_proposal import PROPOSAL, report, specification
from tests.test_research_proposer import CID, IMAGE, SOCKET, replies


def test_remote_words_are_one_quoted_command_and_deadlines_are_preserved() -> None:
    program = "import json; print(json.dumps({'text': \"$HOME `id` 'quoted'\\nline\"}))"
    context: JSONObject = {"text": "$(touch /tmp/should-not-exist)"}
    fake = FakeRun(replies(dump_json_str(context)))
    _test_hooks.run = fake
    assert run_proposer(SOCKET, IMAGE, program, context, host="colossus", timeout=11) == context
    assert all(call[: 1 + len(SSH_OPTIONS)] == ("ssh", *SSH_OPTIONS) for call in fake.calls)
    assert all(call[-2] == "colossus" for call in fake.calls)
    commands = [shlex.split(call[-1]) for call in fake.calls]
    assert commands[2][:6] == ["sudo", "-n", "-u", EXEC_USER, "/usr/bin/env", "-i"]
    assert commands[2][-2:] == [program, dump_json_str(context)]
    assert commands[-2][-4:] == ["container", "rm", "--force", CID]
    assert fake.timeouts == [30, 30, 30, 30, 11, 30, 30, 30, 30]


@pytest.mark.parametrize("host", ["", "-oProxyCommand=bad", "node;id", "user@node", "node/path"])
def test_remote_alias_is_validated_before_transport(host: str) -> None:
    fake = FakeRun([])
    _test_hooks.run = fake
    with pytest.raises(ValueError, match="RESEARCH_PROPOSER_HOST"):
        run_proposer(SOCKET, IMAGE, "print('{}')", {}, host=host, timeout=10)
    assert fake.calls == []


@pytest.mark.parametrize("failure", [failed(255, "connection lost"), timed_out(2)])
def test_remote_wait_failure_preserves_cleanup(failure: _test_hooks.CommandResult) -> None:
    values = replies()
    fake = FakeRun([*values[:4], failure, *values[6:]])
    _test_hooks.run = fake
    with pytest.raises(AppError, match="colossus"):
        run_proposer(SOCKET, IMAGE, "print('{}')", {}, host="colossus", timeout=2)
    assert shlex.split(fake.calls[-2][-1])[-4:] == ["container", "rm", "--force", CID]
    assert "label=corvis.research.proposer=" in shlex.split(fake.calls[-1][-1])[-1]


@pytest.mark.parametrize(("status", "reason"), [("admitted", ""), ("refused", "holdout_forbidden")])
def test_remote_output_reaches_local_existing_admission(
    tmp_path: pathlib.Path, status: str, reason: str
) -> None:
    spec = specification(tmp_path)
    spec["host"] = "colossus"
    fake = FakeRun([*replies(dump_json_str(PROPOSAL)), ok(report(status, reason))])
    _test_hooks.run = fake
    result = research_proposal.propose(spec, tmp_path, tmp_path / "admission.jsonl")
    assert (result["status"], result["reason"]) == (status, reason)
    assert fake.calls[-1][0] == spec["python"]
    assert fake.calls[-1][1] == str(tmp_path / "research_proposals.py")
    assert all(call[0] == "ssh" for call in fake.calls[:-1])
    assert all(str(tmp_path) not in word for call in fake.calls[:-1] for word in call)
