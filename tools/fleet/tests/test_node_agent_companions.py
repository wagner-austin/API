"""A node runner's claim of a project that reads a second repository (MCPs board task 0515040d).

Split from ``test_node_agent.py`` at the 600-line ceiling: the same tick,
against the same faked queue and fleet, for a project that declares one
companion, so what is asserted is where the companion is fetched, staged
and recorded.
"""

from __future__ import annotations

import pathlib

from platform_core.json_utils import dump_json_str

from fleet.cli import _config, node_agent
from fleet.contracts.node import NodePlatform
from fleet.contracts.source import ProjectCompanion
from fleet.core import _test_hooks, dialect, records, stage_repository, staging
from tests._node_agent_fixtures import (
    COMPANION_DIRECTORY,
    COMPANION_REF,
    COMPANION_REMOTE,
    COMPANION_SHA,
    NPM_CI,
    PROBED,
    _credentials_in_env,
    node_argv,
    prebuilt_companion,
    prebuilt_export,
    sourced_document,
)
from tests._queue_fakes import DEFAULT_JOB_ID, FakeQueue, queue_job
from tests._registry import write_registry
from tests.conftest import FakeRun, ok, registry_replies, stage_replies

__all__ = ["_credentials_in_env"]


def _companion_config(config_path: pathlib.Path) -> pathlib.Path:
    """Rewrite the workspace so the demo project declares one companion.

    Args:
        config_path: The shared workspace document, clock pinned.

    Returns:
        The same path, rewritten.
    """
    return write_registry(
        config_path,
        sourced_document(
            (NPM_CI,),
            (
                ProjectCompanion(
                    remote=COMPANION_REMOTE, ref=COMPANION_REF, directory=COMPANION_DIRECTORY
                ),
            ),
        ),
    )


def _companion_commit_script() -> str:
    """The script that clones lavender's staged companion from its bundle.

    Returns:
        Its text, as the dialect renders it for a Windows node.
    """
    return dialect.for_platform(NodePlatform.WINDOWS).checked_script(
        stage_repository.companion_repository_commands(
            "C:/fleet/stage/MCPs", "C:/fleet/stage/MCPs.stage/tree.tgz", COMPANION_SHA, "main"
        )
    )


def _companion_replies(
    export_digest: str, companion_digest: str, registry: pathlib.Path
) -> list[_test_hooks.CommandResult]:
    """Every command a claim tick runs for a project carrying one companion.

    Args:
        export_digest: What the node reports for the project's archive.
        companion_digest: What it reports for the companion's.
        registry: The registry directory the tick loaded.

    Returns:
        One result per call, in order.
    """
    return [
        *PROBED,
        ok(""),  # the project's mirror: git init --bare
        ok(""),  # the project's mirror: git cat-file -e, the commit is held
        ok(""),  # the companion's mirror: git init --bare
        ok(""),  # the companion: git fetch the ref
        ok(f"{COMPANION_SHA}\n"),  # the companion: git rev-parse the tip
        ok(""),  # the companion: git bundle create
        ok(""),  # the project: git archive
        *stage_replies(export_digest),
        *registry_replies(registry),
        ok(""),  # companion: send reset script
        ok(""),  # companion: run reset
        ok(""),  # companion: send mkdir script
        ok(""),  # companion: run mkdir
        ok(""),  # companion: send the base64 payload
        ok(""),  # companion: send reassemble script
        ok(companion_digest),  # companion: run reassemble
        ok(""),  # companion: send the clone script
        ok(""),  # companion: run the clone
        ok(""),  # launch: send the build script
        ok(""),  # launch: send the registration script
        ok("launched"),  # launch: run the registration script
        *PROBED,  # the re-probe after the launch, which finds the project held
    ]


class TestCompanions:
    """A project whose check reads a second repository (MCPs board task
    0515040d). slime lints its lifted code against the committed workspace
    beside it, and a node that was handed slime's commit alone stopped at that
    gate on every sha: job f25cb840 on sedona, exit 2 in under three minutes,
    'no git checkout at C:\\fleet\\stage\\MCPs'."""

    def test_the_companion_is_fetched_at_its_tip_before_the_lease_and_staged_beside_the_run(
        self, config_path: pathlib.Path
    ) -> None:
        """The whole path, through the real prepare, the real export and the
        real staging: the ref resolved to a commit, the ref bundled, and the
        clone landed at ``<stage_root>/MCPs`` -- which is ``../MCPs`` from
        the export root, the one spelling that answers on a node and on a
        workstation alike."""
        sourced = _companion_config(config_path)
        export_payload = prebuilt_export(sourced)
        companion_payload = prebuilt_companion(sourced)
        runner = FakeRun(
            _companion_replies(
                staging.digest(export_payload), staging.digest(companion_payload), sourced
            )
        )
        _test_hooks.run = runner
        _test_hooks.http_post = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                dump_json_str({"claimed": queue_job(status="claimed")}),
                dump_json_str({"job": queue_job(status="running", node="lavender")}),
            ]
        )

        assert node_agent.main(node_argv(sourced)) == 0

        mirror = str(config_path.parent / "runs" / "mirrors" / "companion-wagner-austin-MCPs.git")
        # After the pre-claim probes and the project's two mirror commands.
        base = len(PROBED) + 2
        assert runner.calls[base] == ("git", "init", "--bare", "--quiet", mirror)
        assert runner.calls[base + 1] == (
            "git",
            "-C",
            mirror,
            "fetch",
            "--quiet",
            "--no-tags",
            "--force",
            COMPANION_REMOTE,
            f"+{COMPANION_REF}:refs/fleet/companion",
        )
        assert runner.calls[base + 2][3] == "rev-parse"
        assert runner.calls[base + 3][3:5] == ("bundle", "create")
        assert runner.calls[base + 3][-1] == "refs/fleet/companion"
        # Written in the claimed job's own directory, so no other runner's
        # export of the same tip can replace it, and gone once staged.
        bundles = _config.load_workspace({_config.CONFIG_FLAG: str(sourced)}).archives
        bundle = bundles / DEFAULT_JOB_ID / f"companion-wagner-austin-MCPs-{COMPANION_SHA}.bundle"
        assert runner.calls[base + 3][-2] == str(bundle)
        assert not bundle.parent.exists()
        sent = [body or b"" for body in runner.stdin]
        assert _companion_commit_script().encode() in sent

    def test_the_companion_lands_before_the_build_script_is_sent(
        self, config_path: pathlib.Path
    ) -> None:
        """The install steps are part of that script, so a project whose
        install or recipe reads its companion would otherwise find nothing
        there."""
        sourced = _companion_config(config_path)
        export_payload = prebuilt_export(sourced)
        companion_payload = prebuilt_companion(sourced)
        runner = FakeRun(
            _companion_replies(
                staging.digest(export_payload), staging.digest(companion_payload), sourced
            )
        )
        _test_hooks.run = runner
        _test_hooks.http_post = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                dump_json_str({"claimed": queue_job(status="claimed")}),
                dump_json_str({"job": queue_job(status="running", node="lavender")}),
            ]
        )

        assert node_agent.main(node_argv(sourced)) == 0

        sent = [body or b"" for body in runner.stdin]
        committed = sent.index(_companion_commit_script().encode())
        built = next(index for index, body in enumerate(sent) if b"[string[]]$Install" in body)
        assert committed < built

    def test_the_feed_names_the_commit_the_companion_was_measured_with(
        self, config_path: pathlib.Path
    ) -> None:
        """A companion is the tip of a ref, so two runs of one sha can
        legitimately measure different things; the line is where that shows."""
        sourced = _companion_config(config_path)
        export_payload = prebuilt_export(sourced)
        companion_payload = prebuilt_companion(sourced)
        _test_hooks.run = FakeRun(
            _companion_replies(
                staging.digest(export_payload), staging.digest(companion_payload), sourced
            )
        )
        _test_hooks.http_post = FakeQueue(
            [
                dump_json_str({"jobs": []}),
                dump_json_str({"claimed": queue_job(status="claimed")}),
                dump_json_str({"job": queue_job(status="running", node="lavender")}),
            ]
        )

        assert node_agent.main(node_argv(sourced)) == 0

        loaded = _config.load_workspace({_config.CONFIG_FLAG: str(sourced)})
        staged = [
            event["detail"] for event in records.read_feed(loaded.feed) if event["kind"] == "staged"
        ]
        assert len(staged) == 2
        assert staged[1] == (
            f"{len(companion_payload)} bytes, the MCPs companion at {COMPANION_SHA}, "
            "to C:/fleet/stage/MCPs"
        )
