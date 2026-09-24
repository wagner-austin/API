"""What the node-runner tests share: a workspace whose demo project has a
remote, the runner's argument list, the archive ``git archive`` would have
produced, and the exact command sequence one claim tick issues.

Support module, not a test module: ``test_node_agent.py`` (the claim tick and
its refusals), ``test_node_agent_collect.py`` (the collect tick, the announce
and the entry points) and ``test_node_collect_stop.py`` (the runs a tick
stops) all import from here, split at the file-size ceiling by which part of
the tick they exercise.
"""

from __future__ import annotations

import pathlib

import pytest
from board_watch import _test_hooks as board_watch_hooks
from board_watch import config as board_config
from platform_core.json_utils import JSONObject, dump_json_str
from platform_core.mcp_testing import DECLARED_TASKBOARD_URL

from fleet.cli import _config, node_agent
from fleet.contracts.source import ProjectCompanion, ProjectSource, encode_project_source
from fleet.core import _test_hooks, queue, staging
from tests._queue_fakes import DEFAULT_SHA, FakeEnv, FakeQueue, queue_job
from tests._toolchain_fixtures import LAVENDER_2026_09_23
from tests.conftest import (
    DEMO_PROJECT,
    DEMO_RUN_ID,
    PROBE_OK,
    FakeRun,
    failed,
    ok,
    workspace_document,
)

REMOTE = "https://github.com/wagner-austin/API.git"
VERDICT_TASK = "fd5cabfa-a328-48f4-b5e9-3a02dd531ea5"

#: The workspace a project can declare as exported beside it, and the commit
#: its ref resolves to in these tests (MCPs board task 0515040d).
COMPANION_REMOTE = "https://github.com/wagner-austin/MCPs.git"
COMPANION_REF = "main"
COMPANION_DIRECTORY = "MCPs"
COMPANION_SHA = "7b3d51c0e9a2f4681becd3057a9f2416c8d0e5b9"

#: The pre-claim probes every claiming tick pays, each script sent and then
#: run: lavender answering room for a dispatch, then its toolchain as it
#: answered on 2026-09-23, ready (MCPs board task bad56f65).
PROBED: tuple[_test_hooks.CommandResult, ...] = (
    ok(""),
    ok(PROBE_OK),
    ok(""),
    ok(LAVENDER_2026_09_23),
)

#: A vitest transcript tail with the banner, as the collect pass reads it.
PASSING_TAIL = (
    "All files |     100 |      100 |     100 |     100 |\n"
    " Test Files  60 passed (60)\n"
    "      Tests  887 passed (887)\n"
    "=== ALL CHECKS PASSED ===\n"
)


@pytest.fixture(name="credentials_in_env", autouse=True)
def _credentials_in_env() -> None:
    """Give every test the queue's and the board's variables."""
    _test_hooks.env = FakeEnv(
        {queue.API_KEY_VARIABLE: "test-key", queue.TENANT_ID_VARIABLE: "tenant"}
    )
    board_watch_hooks.env = FakeEnv(
        {
            board_config.API_KEY_VARIABLE: "board-key",
            board_config.TENANT_ID_VARIABLE: "tenant",
            board_config.URL_VARIABLE: DECLARED_TASKBOARD_URL,
        }
    )


def sourced_document(
    install: tuple[tuple[str, ...], ...],
    companions: tuple[ProjectCompanion, ...] = (),
) -> JSONObject:
    """The shared workspace with the demo project given a source.

    Built through ``encode_project_source`` rather than spelled as a literal:
    a required field added to the source is then a type error in one place
    here instead of a runtime refusal in two dozen tests, which is how
    ``source`` itself landed red (commit f95338cd).

    Args:
        install: The install steps to declare.
        companions: The repositories to declare as exported beside it.

    Returns:
        The document.
    """
    document = workspace_document()
    projects = document["projects"]
    assert isinstance(projects, dict)
    project = projects[DEMO_PROJECT]
    assert isinstance(project, dict)
    project["source"] = encode_project_source(
        ProjectSource(remote=REMOTE, path=DEMO_PROJECT, install=install, companions=companions)
    )
    return document


@pytest.fixture(name="sourced_config")
def _sourced_config(config_path: pathlib.Path) -> pathlib.Path:
    """Rewrite the workspace so the demo project has a remote.

    Args:
        config_path: The shared workspace document, clock pinned.

    Returns:
        The same path, rewritten.
    """
    config_path.write_text(dump_json_str(sourced_document((("npm", "ci"),))), encoding="utf-8")
    return config_path


def node_argv(config_path: pathlib.Path) -> list[str]:
    """The node runner's argument list for lavender.

    Args:
        config_path: The workspace document.

    Returns:
        The argument list.
    """
    return ["--config", str(config_path), node_agent.NODE_FLAG, "lavender"]


def prebuilt_export(config_path: pathlib.Path) -> bytes:
    """Write the archive ``git archive`` would have, where the tick reads it.

    Args:
        config_path: The workspace document.

    Returns:
        The archive bytes.
    """
    loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
    destination = loaded.archives / f"{DEMO_RUN_ID}-lavender.tgz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = b"\x1f\x8b" + b"export-of-" + DEFAULT_SHA.encode("ascii")
    destination.write_bytes(payload)
    return payload


def prebuilt_companion(config_path: pathlib.Path) -> bytes:
    """Write the archive a companion's ``git archive`` would have, where the
    tick reads it.

    Named by the companion's own commit rather than by the run, which is the
    name :func:`fleet.core.export.export_companions` writes.

    Args:
        config_path: The workspace document.

    Returns:
        The archive bytes.
    """
    loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})
    destination = loaded.archives / f"companion-wagner-austin-MCPs-{COMPANION_SHA}.tgz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = b"\x1f\x8b" + b"workspace-at-" + COMPANION_SHA.encode("ascii")
    destination.write_bytes(payload)
    return payload


def claim_replies(digest: str, *, commit_present: bool) -> list[_test_hooks.CommandResult]:
    """Every command a node-lane claim tick runs, in order.

    Args:
        digest: What the node reports having reassembled.
        commit_present: Whether the mirror already holds the sha, which
            decides whether a fetch runs.

    Returns:
        One result per call.
    """
    replies = [
        *PROBED,  # the probe, before any claim
        ok(""),  # git init --bare (the mirror is new)
    ]
    if commit_present:
        replies.append(ok(""))  # git cat-file -e: present
    else:
        replies.append(failed(128, "missing"))  # git cat-file -e: absent
        replies.append(ok(""))  # git fetch
    replies += [
        ok(""),  # git archive -o
        ok(""),  # stage: send mkdir script
        ok(""),  # stage: run mkdir
        ok(""),  # stage: send the base64 payload
        ok(""),  # stage: send reassemble script
        ok(digest),  # stage: run reassemble
        ok(""),  # stage: send extract script
        ok(""),  # stage: run extract
        ok(""),  # stage: send the git-init script
        ok(""),  # stage: run git init
        ok(""),  # launch: send the build script
        ok(""),  # launch: send the registration script
        ok("launched"),  # launch: run the registration script
    ]
    return replies


def launch(config_path: pathlib.Path) -> None:
    """Run one tick that claims and launches, leaving a live ledger row.

    Lifted from ``test_node_agent_collect.py`` when the stop tests needed
    the same starting point: a run lavender is building, which a later tick
    collects, stops, or finds cancelled.

    Args:
        config_path: The workspace document.
    """
    payload = prebuilt_export(config_path)
    _test_hooks.run = FakeRun(claim_replies(staging.digest(payload), commit_present=True))
    _test_hooks.http_post = FakeQueue(
        [
            dump_json_str({"jobs": []}),
            dump_json_str({"claimed": queue_job(status="claimed", taskId=VERDICT_TASK)}),
            dump_json_str({"job": queue_job(status="running", node="lavender")}),
        ]
    )
    node_agent.main(node_argv(config_path))


def held_answer(**overrides: str | None) -> str:
    """The queue's answer listing the launched job as this runner's.

    Args:
        **overrides: Fields to vary on the row.

    Returns:
        The rendered ``dispatch_list`` answer.
    """
    return dump_json_str(
        {"jobs": [queue_job(status="running", node="lavender", runId=DEMO_RUN_ID, **overrides)]}
    )


__all__ = [
    "COMPANION_DIRECTORY",
    "COMPANION_REF",
    "COMPANION_REMOTE",
    "COMPANION_SHA",
    "PASSING_TAIL",
    "PROBED",
    "REMOTE",
    "VERDICT_TASK",
    "claim_replies",
    "held_answer",
    "launch",
    "node_argv",
    "prebuilt_companion",
    "prebuilt_export",
    "sourced_document",
]
