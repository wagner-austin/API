"""What a node is asked to do to stage one companion, in order (fleet.core.companion_stage).

The repository a project's check reads BESIDE its export (MCPs board task
0515040d), staged since MCPs board task 5280a25d out of the node's own mirror
of its remote. Only the command boundary is faked here, so every call and
every script sent is asserted; ``tests/test_companion_stage_parallel.py``
runs the same scripts and the hub's git for real.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.core import (
    _test_hooks,
    commit_stage,
    companion_stage,
    dialect,
    export,
    names,
    stage_repository,
    staging,
)
from tests.conftest import DEMO_RUN_ID, FakeRun, ok

#: The commit a staged companion's ref resolved to, in these tests.
COMPANION_SHA = "9f1c0b7a2d3e4f5061728394a5b6c7d8e9f01234"

#: The stage root every node declares.
STAGE_ROOT = "/home/corvis/fleet/stage"

#: Where the companion lands, its staging directory, and the node's mirror.
TREE = f"{STAGE_ROOT}/MCPs"
STAGED = f"{STAGE_ROOT}/MCPs.stage"
MIRROR = f"{STAGE_ROOT}/cache/mirrors/companion-wagner-austin-MCPs.git"

#: The bundle's bytes as the hub's git would have written them.
PAYLOAD = b"# v2 git bundle\ncompanion"


def _companion(tmp_path: pathlib.Path) -> export.CompanionExport:
    """The MCPs companion as the hub resolved it, its bundle already written.

    Args:
        tmp_path: The case's directory, standing for the job's attempt.

    Returns:
        The export.
    """
    bundle = tmp_path / f"companion-wagner-austin-MCPs-{COMPANION_SHA}.bundle"
    bundle.write_bytes(PAYLOAD)
    return export.CompanionExport(
        directory="MCPs",
        ref="main",
        sha=COMPANION_SHA,
        mirror=tmp_path / "hub.git",
        key="companion-wagner-austin-MCPs",
        bundle=bundle,
    )


def _sent_replies(digest: str) -> list[_test_hooks.CommandResult]:
    """Every command staging a companion the node's mirror has never seen runs.

    Args:
        digest: What the node reports for the bundle.

    Returns:
        One result per call, in order.
    """
    return [
        ok(""),  # send the prepare script
        ok(""),  # run it: the tree is replaced, and the node has no mirror yet
        ok(""),  # hub: git update-ref of the bundle's ref
        ok(""),  # hub: git bundle create
        ok(""),  # hub: git update-ref -d of the bundle's ref
        ok(""),  # scp the bundle into the staging directory
        ok(""),  # send the digest script
        ok(digest),  # run it
        ok(""),  # send the fetch-and-clone script
        ok(""),  # run it
    ]


def _stage(companion: export.CompanionExport) -> companion_stage.StagedCompanion:
    """Stage it on lavender.

    Args:
        companion: The companion.

    Returns:
        Its stage.
    """
    return companion_stage.stage_companion(
        "lavender", stage_root=STAGE_ROOT, run_id=DEMO_RUN_ID, companion=companion
    )


class TestStagingACompanion:
    def test_one_the_node_lacks_is_bundled_sent_and_cloned_out_of_its_mirror(
        self, tmp_path: pathlib.Path
    ) -> None:
        """``<stage_root>/<directory>`` is ``../<directory>`` from an export
        root, which is where a workstation keeps the same checkout, so the
        recipe names one path on either machine."""
        companion = _companion(tmp_path)
        runner = FakeRun(_sent_replies(staging.digest(PAYLOAD)))
        _test_hooks.run = runner

        staged = _stage(companion)

        prepared = companion_stage.prepare_script(
            tree=TREE, staged=STAGED, mirror=MIRROR, sha=COMPANION_SHA, ref="main"
        )
        assert runner.calls[0][-1] == dialect.write_command(f"{STAGE_ROOT}/reset-MCPs.sh")
        assert runner.stdin[0] == prepared.encode()
        bundle_ref = f"refs/fleet/bundle/{DEMO_RUN_ID}-MCPs"
        assert runner.calls[3][3:5] == ("bundle", "create")
        assert runner.calls[3][-1] == bundle_ref
        assert runner.calls[5][-1] == f"lavender:{STAGED}/{names.ARCHIVE_NAME}"
        assert runner.stdin[8] == (
            dialect.checked_script(
                (
                    *commit_stage.fetch_commands(
                        mirror=MIRROR,
                        bundle=f"{STAGED}/{names.ARCHIVE_NAME}",
                        run_id=f"{DEMO_RUN_ID}-MCPs",
                        base_branch=False,
                    ),
                    *stage_repository.companion_repository_commands(
                        TREE, MIRROR, COMPANION_SHA, "main"
                    ),
                )
            ).encode()
        )
        assert len(runner.calls) == 10
        assert staged == companion_stage.StagedCompanion(
            tree=TREE,
            detail=(
                f"{len(PAYLOAD)} bytes, the MCPs companion at {COMPANION_SHA} over an empty "
                f"mirror, to {TREE}"
            ),
        )
        assert not companion["bundle"].exists()

    def test_the_bundle_never_enters_the_tree_that_is_cloned(self, tmp_path: pathlib.Path) -> None:
        """A ``tree.tgz`` at the root of a staged workspace would be a file the
        workspace does not have, sitting in the tree a check compares against
        it, so the transport files stay in a staging directory beside it."""
        runner = FakeRun(_sent_replies(staging.digest(PAYLOAD)))
        _test_hooks.run = runner

        _stage(_companion(tmp_path))

        written = [argument for call in runner.calls for argument in call]
        assert not any(f"{TREE}/{names.ARCHIVE_NAME}" in text for text in written)
        assert f"lavender:{STAGED}/{names.ARCHIVE_NAME}" in written

    def test_one_the_node_mirror_reaches_is_cloned_with_nothing_sent(
        self, tmp_path: pathlib.Path
    ) -> None:
        companion = _companion(tmp_path)
        held = ok(f"{COMPANION_SHA}\n{commit_stage.HELD_ANSWER}\n")
        runner = FakeRun([ok(""), held, ok(""), ok("")])
        _test_hooks.run = runner

        staged = _stage(companion)

        assert not any(call[0] == "scp" or "bundle" in call for call in runner.calls)
        assert runner.stdin[2] == (
            dialect.checked_script(
                stage_repository.companion_repository_commands(TREE, MIRROR, COMPANION_SHA, "main")
            ).encode()
        )
        assert staged["detail"] == (
            f"0 bytes, the MCPs companion at {COMPANION_SHA} already in the node's mirror, "
            f"to {TREE}"
        )

    def test_one_kept_clean_at_its_commit_runs_its_prepare_script_alone(
        self, tmp_path: pathlib.Path
    ) -> None:
        runner = FakeRun([ok(""), ok(f"{companion_stage.REUSED_ANSWER}\n")])
        _test_hooks.run = runner

        staged = _stage(_companion(tmp_path))

        assert len(runner.calls) == 2
        assert staged["detail"] == (
            f"0 bytes, the MCPs companion at {COMPANION_SHA} kept clean at {TREE}"
        )

    def test_a_mismatched_digest_names_the_companion_and_clones_nothing(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The message says WHICH payload disagreed, because a dispatch
        carries more than one and a mismatch that named none would leave a
        reader looking at the wrong transfer."""
        runner = FakeRun(_sent_replies("0" * 64)[:8])
        _test_hooks.run = runner

        with pytest.raises(AppError) as excinfo:
            _stage(_companion(tmp_path))

        assert excinfo.value.code is FleetErrorCode.STAGE_DIGEST_MISMATCH
        assert f"the MCPs companion at {COMPANION_SHA}" in excinfo.value.message
        assert not any(b"git clone" in (sent or b"") for sent in runner.stdin)


class TestThePrepareScript:
    def test_a_clean_tree_is_kept_before_any_tree_is_replaced(self) -> None:
        """Every run carrying a companion writes the same directory, so a
        file the workspace has since deleted would otherwise survive into the
        tree a check then reads AS the workspace: a tree is kept only when its
        refs, its branch and an empty ``git status --ignored`` say it is the
        clone this stage would make, and otherwise removed first."""
        script = companion_stage.prepare_script(
            tree=TREE, staged=STAGED, mirror=MIRROR, sha=COMPANION_SHA, ref="refs/heads/main"
        )

        kept = script.index("status --porcelain --ignored --untracked-files=all")
        removed = script.index(f"rm -rf {TREE}\n")
        made = script.index(f"mkdir -p {TREE} {STAGED}\n")
        assert kept < removed < made
        refs = f"refs/heads/main {COMPANION_SHA}\nrefs/remotes/origin/main {COMPANION_SHA}"
        assert refs in script
        assert script.endswith(
            commit_stage.mirror_probe_lines(mirror=MIRROR, sha=COMPANION_SHA, base_tip=None)
        )
