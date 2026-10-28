"""Research agents run through the existing CLI, identity and scheduling surfaces."""

from __future__ import annotations

import hashlib
import pathlib
import runpy
import sys

import pytest
from board_watch import _test_hooks as board_hooks
from platform_core.json_utils import JSONObject, dump_json_str, load_json_str

from fleet.cli import research, research_native, rolled, tick
from fleet.contracts.research_worker import encode_worker
from fleet.core import _test_hooks
from tests._queue_fakes import FakeQueue, queue_env
from tests._research_fixtures import research_job, worker
from tests.conftest import FakeRun, ok
from tests.test_research_hpc import _document as hpc_document
from tests.test_research_hpc import _row
from tests.test_research_wrds import _document as wrds_document


@pytest.mark.parametrize("backend", ["hpc3", "wrds"])
def test_native_cli_resolves_authorized_bytes_and_emits_only_metadata(
    tmp_path: pathlib.Path, backend: str, capsys: pytest.CaptureFixture[str]
) -> None:
    document = hpc_document(tmp_path) if backend == "hpc3" else wrds_document(tmp_path, adopt=True)
    payload = dump_json_str(document).encode()
    digest = hashlib.sha256(payload).hexdigest()
    directory = tmp_path / digest
    directory.mkdir()
    (directory / "request.json").write_bytes(payload)
    env = tmp_path / "env.ps1"
    env.write_text("$env:TASKBOARD_MCP_API_KEY = 'private-token'\n")
    reply: JSONObject = (
        {"attempts": [_row("running")]} if backend == "hpc3" else {"job_id": "123", "state": "r"}
    )
    _test_hooks.run = FakeRun([ok(dump_json_str(reply)), ok(dump_json_str(reply))])
    args = [
        "--root",
        str(tmp_path),
        "--backend",
        backend,
        "--digest",
        digest,
        "--attempt",
        "trial-1" if backend == "hpc3" else "pilot",
        "--tenant",
        "ours",
        "--task",
        "33333333-cccc-4ccc-8ccc-333333333333",
        "--env-file",
        str(env),
    ]
    assert research_native.main(args) == 0
    assert load_json_str(capsys.readouterr().out) == {
        "state": "running",
        "native_job_id": "12345" if backend == "hpc3" else "123",
        "record_sha256": None,
    }
    assert not (directory / "poll.lock").exists()


@pytest.mark.parametrize("announce,active", [(True, False), (False, False), (False, True)])
def test_queue_cli_announces_or_runs_one_tick(
    config_path: pathlib.Path,
    tmp_path: pathlib.Path,
    capsys: pytest.CaptureFixture[str],
    announce: bool,
    active: bool,
) -> None:
    env = queue_env()
    _test_hooks.env = env
    board_hooks.env = env
    root = tmp_path / "api"
    config = root / "tools/fleet/runs/research.json"
    config.parent.mkdir(parents=True)
    config.write_text(dump_json_str(encode_worker(worker())))
    row = research_job(status="running", runId="receipt")
    replies = (
        ["posted"]
        if announce
        else [dump_json_str({"jobs": []}), dump_json_str({"job": row if active else None})]
    )
    if active:
        replies.append(dump_json_str({"job": row}))
    endpoint = FakeQueue(replies)
    _test_hooks.http_post = endpoint
    _test_hooks.run = FakeRun(
        [ok(""), ok('{"state":"running","native_job_id":"123","record_sha256":null}')]
    )
    args = [
        "--config",
        str(config_path),
        "--repo-root",
        str(root),
        "--agent",
        "fleet-research-test",
        "--session",
        "33333333-cccc-4ccc-8ccc-333333333333",
    ]
    if announce:
        args.append("--announce")
    assert research.main(args) == 0
    output = capsys.readouterr().out
    if announce:
        assert output == "posted\n"
        assert endpoint.tools == ["task_post"]
    else:
        assert ("running research request" in output) == active
        assert endpoint.tools[:2] == ["dispatch_list", "dispatch_claim"]


@pytest.mark.parametrize("module", ["fleet.cli.research", "fleet.cli.research_native"])
def test_module_entrypoints_refuse_missing_arguments(module: str) -> None:
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


@pytest.mark.parametrize("lane", [tick.Lane.RESEARCH, tick.Lane.RESEARCH_ANNOUNCE])
def test_existing_tick_and_rolled_launcher_select_research_lane(lane: tick.Lane) -> None:
    plan = tick.plan_tick(pathlib.Path("/api"), lane, None, {})
    assert plan["arguments"][3] == "fleet-research"
    assert rolled.AGENT_MODULES["fleet-research"] == "fleet.cli.research"
    assert "--lane" not in plan["arguments"]
    assert ("--announce" in plan["arguments"]) == (lane is tick.Lane.RESEARCH_ANNOUNCE)
    assert plan["stem"].startswith("fleet-research-")
