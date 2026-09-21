"""What the node-runner tests share: a workspace whose demo project has a
remote, the runner's argument list, the archive ``git archive`` would have
produced, and the exact command sequence one claim tick issues.

Support module, not a test module: ``test_node_agent.py`` (the claim tick and
its refusals) and ``test_node_agent_collect.py`` (the collect tick, the
announce and the entry points) both import from here, split at the file-size
ceiling by which half of the tick they exercise.
"""

from __future__ import annotations

import pathlib

import pytest
from board_watch import _test_hooks as board_watch_hooks
from board_watch import config as board_config
from platform_core.json_utils import JSONObject, JSONValue, dump_json_str

from fleet.cli import _config, node_agent
from fleet.core import _test_hooks, queue
from tests._queue_fakes import DEFAULT_SHA, FakeEnv
from tests.conftest import DEMO_PROJECT, DEMO_RUN_ID, PROBE_OK, failed, ok, workspace_document

REMOTE = "https://github.com/wagner-austin/API.git"
VERDICT_TASK = "fd5cabfa-a328-48f4-b5e9-3a02dd531ea5"

#: The pre-claim probe every claiming tick pays: the script sent, then run,
#: with lavender answering room for a dispatch.
PROBED: tuple[_test_hooks.CommandResult, ...] = (ok(""), ok(PROBE_OK))

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
        {board_config.API_KEY_VARIABLE: "board-key", board_config.TENANT_ID_VARIABLE: "tenant"}
    )


def sourced_document(install: list[JSONValue]) -> JSONObject:
    """The shared workspace with the demo project given a source.

    Args:
        install: The install steps to declare, each a list of tokens.

    Returns:
        The document.
    """
    document = workspace_document()
    projects = document["projects"]
    assert isinstance(projects, dict)
    project = projects[DEMO_PROJECT]
    assert isinstance(project, dict)
    project["source"] = {"remote": REMOTE, "path": DEMO_PROJECT, "install": install}
    return document


@pytest.fixture(name="sourced_config")
def _sourced_config(config_path: pathlib.Path) -> pathlib.Path:
    """Rewrite the workspace so the demo project has a remote.

    Args:
        config_path: The shared workspace document, clock pinned.

    Returns:
        The same path, rewritten.
    """
    config_path.write_text(dump_json_str(sourced_document([["npm", "ci"]])), encoding="utf-8")
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


__all__ = [
    "PASSING_TAIL",
    "PROBED",
    "REMOTE",
    "VERDICT_TASK",
    "claim_replies",
    "node_argv",
    "prebuilt_export",
    "sourced_document",
]
