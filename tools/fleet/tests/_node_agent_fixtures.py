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
from fleet.contracts.source import (
    InstallStep,
    ProjectCompanion,
    ProjectSource,
    encode_project_source,
)
from fleet.core import _test_hooks, staging
from fleet.core import _test_hooks_attempts as attempt_hooks
from tests._queue_fakes import (
    DEFAULT_JOB_ID,
    DEFAULT_SHA,
    FakeEnv,
    FakeQueue,
    queue_env,
    queue_job,
)
from tests._registry import write_registry
from tests._commit_stage_replies import commit_stage_replies
from tests._thread_fakes import in_order_executor
from tests._toolchain_fixtures import DIPHTHERIA_2026_09_23
from tests.conftest import (
    DEMO_PROJECT,
    DEMO_RUN_ID,
    PROBE_OK,
    FakeRun,
    failed,
    ok,
    registry_replies,
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

#: The launch attempt's own bundle directory under the archives, which
#: :func:`attempt_directory` binds the attempt seam to.
ATTEMPT_DIRECTORY = f"{DEFAULT_JOB_ID}-attempt"

#: The commit's bundle as the hub writes it in the attempt's directory:
#: the demo remote's mirror key and the commit, and what it holds.
EXPORT_BUNDLE = f"wagner-austin-API-{DEFAULT_SHA}.bundle"
EXPORT_BUNDLE_BYTES = b"# v2 git bundle\n" + b"export-of-" + DEFAULT_SHA.encode("ascii")

#: The pre-claim probes every claiming tick pays, each script sent and then
#: run: lavender answering room for a dispatch, then its toolchain as it
#: answered on 2026-09-23, ready (MCPs board task bad56f65).
#: What lavender logs, and records as its tick's verdict, when it asked the
#: queue for the one registered project and the lane held nothing for it.
NOTHING_MATCHED = "lavender asked for 1 fitting project(s); nothing in the node lane matched"

#: The last line of every lavender fill pass that launched nothing, whichever
#: gate or refusal ended it (MCPs board task 48842bfd).
NOTHING_LAUNCHED = "lavender launched 0 job(s) this pass"

#: The line that ends every lavender serve of zero seconds whose passes left
#: it holding no running job, started at DEMO_NOW, 20 s after a fire
#: boundary: it sleeps to 10 s before the next and hands over there, its
#: watch having read nothing (MCPs board task 8993c306).
SERVED_HOLDING_NOTHING = (
    "lavender served 150 s from 2025-09-04T15:33:20+00:00: 0 fire(s), 1 fill pass(es), "
    "0 poll(s), 0 run(s) closed, 0 still watched; handed over at 2025-09-04T15:35:50+00:00 "
    "before the 2025-09-04T15:36:00+00:00 fire: its node_serve_seconds is 0"
)

PROBED: tuple[_test_hooks.CommandResult, ...] = (
    ok(""),
    ok(PROBE_OK),
    ok(""),
    ok(DIPHTHERIA_2026_09_23),
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
    """Give every test the queue's and the board's variables, and its launches in order.

    Each claimed job's launch finishes before its claim goes on, because the
    runner's answers are scripted in one order, which a launch beside the
    next claim would interleave (:mod:`tests._thread_fakes`); a case about
    launching beside the claim binds the real pool after this.
    """
    _test_hooks.executor = in_order_executor
    _test_hooks.env = queue_env()
    board_watch_hooks.env = FakeEnv(
        {
            board_config.API_KEY_VARIABLE: "board-key",
            board_config.TENANT_ID_VARIABLE: "tenant",
            board_config.URL_VARIABLE: DECLARED_TASKBOARD_URL,
        }
    )


#: The install step most sourced fixtures declare.
NPM_CI = InstallStep(phase="install", argv=("npm", "ci"))


def sourced_document(
    install: tuple[InstallStep, ...],
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
    return write_registry(config_path, sourced_document((NPM_CI,)))


def node_argv(config_path: pathlib.Path) -> list[str]:
    """The node runner's argument list for lavender.

    Args:
        config_path: The workspace document.

    Returns:
        The argument list.
    """
    return ["--config", str(config_path), node_agent.NODE_FLAG, "lavender"]


def attempt_directory(config_path: pathlib.Path, job_id: str) -> pathlib.Path:
    """Bind each launch attempt's own directory to ``<job_id>-attempt`` under the archives.

    :func:`fleet.cli.node_prepare.prepare` makes the attempt's directory, and
    the hub writes the commit's and the companions' bundles there, so a
    bundle a faked ``git bundle create`` would have written can be placed
    before the tick reads it. :data:`ATTEMPT_DIRECTORY` is the default job's.

    Args:
        config_path: The workspace document.
        job_id: The job whose attempt's directory is wanted.

    Returns:
        That directory, made.
    """
    loaded = _config.load_workspace({_config.CONFIG_FLAG: str(config_path)})

    def fresh(parent: pathlib.Path, prefix: str) -> pathlib.Path:
        assert parent == loaded.archives
        made = parent / f"{prefix}attempt"
        made.mkdir(parents=True, exist_ok=True)
        return made

    attempt_hooks.fresh_directory = fresh
    return fresh(loaded.archives, f"{job_id}-")


def prebuilt_export(config_path: pathlib.Path, job_id: str = DEFAULT_JOB_ID) -> bytes:
    """Write the bundle the commit's ``git bundle create`` would have, where the tick reads it.

    Named by the demo remote's mirror key and the commit, in the launch
    attempt's own directory (:func:`attempt_directory`).

    Args:
        config_path: The workspace document.
        job_id: The job whose attempt stages the commit.

    Returns:
        The bundle bytes.
    """
    (attempt_directory(config_path, job_id) / EXPORT_BUNDLE).write_bytes(EXPORT_BUNDLE_BYTES)
    return EXPORT_BUNDLE_BYTES


def prebuilt_companion(config_path: pathlib.Path) -> bytes:
    """Write the bundle a companion's ``git bundle create`` would have, where
    the tick reads it.

    Named by the companion's own commit, in the launch attempt's own
    directory (:func:`attempt_directory`), which is where
    :func:`fleet.cli.node_prepare.prepare` has
    :func:`fleet.core.export.export_companions` write it.

    Args:
        config_path: The workspace document.

    Returns:
        The bundle bytes.
    """
    attempt = attempt_directory(config_path, DEFAULT_JOB_ID)
    destination = attempt / f"companion-wagner-austin-MCPs-{COMPANION_SHA}.bundle"
    payload = b"# v2 git bundle\n" + b"workspace-at-" + COMPANION_SHA.encode("ascii")
    destination.write_bytes(payload)
    return payload


def claim_replies(
    digest: str,
    *,
    registry: pathlib.Path,
    commit_present: bool,
    after_launch: tuple[_test_hooks.CommandResult, ...] = (),
) -> list[_test_hooks.CommandResult]:
    """Every command a node-lane claim tick that launches its job runs, in order.

    Args:
        digest: What the node reports having reassembled.
        registry: The registry directory the tick loaded.
        commit_present: Whether the mirror already holds the sha, which
            decides whether a fetch runs.
        after_launch: What the start report runs once the job is launched,
            before the tick probes again: the stop and retire of a job
            cancelled while it launched.

    Returns:
        One result per call.
    """
    return [
        *PROBED,  # the probe, before any claim
        *launch_steps(digest, registry=registry, commit_present=commit_present),
        *after_launch,
        # The tick claims again after a launch (MCPs board task 48842bfd):
        # the re-probe finds the one registered project held by the run just
        # launched, so it asks the queue for nothing more.
        *PROBED,
    ]


def launch_steps(
    digest: str, *, registry: pathlib.Path, commit_present: bool
) -> list[_test_hooks.CommandResult]:
    """Every command between a claim and its launch, for a project whose mirror is new.

    Args:
        digest: What the node reports having reassembled.
        registry: The registry directory the tick loaded.
        commit_present: Whether the mirror already holds the sha, which
            decides whether a fetch runs.

    Returns:
        One result per call.
    """
    fetched = [ok("")] if commit_present else [failed(128, "missing"), ok("")]
    return [
        ok(""),  # git init --bare (the mirror is new)
        *fetched,  # git cat-file -e: present; or absent, then git fetch
        *commit_stage_replies(digest),
        *registry_replies(registry),
        ok(""),  # launch: send the build script
        ok(""),  # launch: send the registration script
        ok("launched"),  # launch: run the registration script
    ]


def launch(config_path: pathlib.Path) -> None:
    """Run one tick that claims and launches, leaving a live ledger row.

    Lifted from ``test_node_agent_collect.py`` when the stop tests needed
    the same starting point: a run lavender is building, which a later tick
    collects, stops, or finds cancelled.

    Args:
        config_path: The workspace document.
    """
    payload = prebuilt_export(config_path)
    _test_hooks.run = FakeRun(
        claim_replies(staging.digest(payload), registry=config_path, commit_present=True)
    )
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
    "ATTEMPT_DIRECTORY",
    "COMPANION_DIRECTORY",
    "COMPANION_REF",
    "COMPANION_REMOTE",
    "COMPANION_SHA",
    "EXPORT_BUNDLE",
    "EXPORT_BUNDLE_BYTES",
    "NOTHING_LAUNCHED",
    "NOTHING_MATCHED",
    "PASSING_TAIL",
    "PROBED",
    "REMOTE",
    "SERVED_HOLDING_NOTHING",
    "VERDICT_TASK",
    "attempt_directory",
    "claim_replies",
    "held_answer",
    "launch",
    "launch_steps",
    "node_argv",
    "prebuilt_companion",
    "prebuilt_export",
    "sourced_document",
]
