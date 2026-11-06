"""A commit staged beside its project's base branch (MCPs board task a48b5236).

A staged tree was one root commit in a repository with no branch, so a check
asking which files its commit changes against ``origin/main`` (the MCPs
TypeScript packages' mutation step) had nothing to ask. A project declaring
``base_branch`` now has the branch's tip resolved on the hub, carried in the
commit's bundle when the node lacks it and written as
``refs/remotes/origin/<branch>`` in the stage, whose own commit is the queued
commit's child. The git half is real, as in ``tests/test_commit_stage.py``: a
hub repository whose main moved on after a feature branch left it, bundles
written by real ``git bundle create``, and the node's scripts run by real
``/bin/sh``.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence

import pytest
from platform_core.errors import AppError
from platform_core.json_utils import JSONObject, dump_json_str

from fleet.cli import _config, node_prepare
from fleet.contracts.dispatch import decode_job
from fleet.contracts.source import ProjectSource, encode_project_source
from fleet.contracts.workspace import require_project
from fleet.core import (
    _test_hooks,
    commit_stage,
    dialect,
    mirror_fetch,
    names,
    node_mirror_probe,
    stage_repository,
    staging,
)
from tests._node_agent_fixtures import NPM_CI, REMOTE, _credentials_in_env
from tests._queue_fakes import DEFAULT_SHA, queue_job
from tests._registry import write_registry
from tests.conftest import DEMO_PROJECT, DEMO_RUN_ID, FakeRun, failed, ok, workspace_document
from tests.test_commit_stage import (
    NOTHING_HELD,
    SHA,
    STAGE_ROOT,
    TIP,
    _commit,
    _git,
    _sh,
    export_of,
)

__all__ = ["_credentials_in_env"]

#: A base branch tip the faked stages are given.
BASE = "0a1b2c3d4e5f60718293a4b5c6d7e8f901234567"


def _diverged(hub: pathlib.Path) -> tuple[str, str, str]:
    """A hub repository whose main moved on after a feature branch left it.

    Args:
        hub: The repository, made here.

    Returns:
        The fork point, the feature commit and main's tip.
    """
    fork = _commit(hub, {"src/a.ts": "export const a = 1;\n", "src/b.ts": "export const b = 1;\n"})
    _git(hub, "checkout", "--quiet", "-b", "feature")
    feature = _commit(hub, {"src/a.ts": "export const a = 2;\n"})
    _git(hub, "checkout", "--quiet", "main")
    tip = _commit(hub, {"src/b.ts": "export const b = 2;\n"})
    return fork, feature, tip


def _based_document() -> JSONObject:
    """The shared workspace with the demo project sourced and declaring main as its base branch.

    Returns:
        The document.
    """
    document = workspace_document()
    projects = document["projects"]
    assert isinstance(projects, dict)
    project = projects[DEMO_PROJECT]
    assert isinstance(project, dict)
    source = ProjectSource(remote=REMOTE, path=DEMO_PROJECT, install=(NPM_CI,), companions=())
    source["base_branch"] = "main"
    project["source"] = encode_project_source(source)
    return document


def _unpack(
    node: pathlib.Path,
    run_id: str,
    sha: str,
    branch: stage_repository.BaseBranch,
    probed: node_mirror_probe.NodeMirror,
    bundle: str,
) -> pathlib.Path:
    """Unpack one bundle into a run's export, as a node does.

    Args:
        node: The node's mirror.
        run_id: The run.
        sha: The queued commit.
        branch: Its base branch and tip.
        probed: What the node's mirror held before the bundle was made.
        bundle: The verified bundle on the node.

    Returns:
        The export's directory.
    """
    target = node.parent / run_id
    target.mkdir()
    (node.parent / f"{run_id}.stage").mkdir()
    _sh(
        commit_stage.unpack_script(
            mirror=str(node),
            bundle=bundle,
            run_id=run_id,
            sha=sha,
            scope=(),
            staged=f"{target}.stage",
            target=str(target),
            base_branch=branch,
            node=probed,
        )
    )
    return target


class TestAStageBesideItsBaseBranch:
    def test_the_stage_answers_merge_base_and_the_changed_files_as_a_checkout_does(
        self, tmp_path: pathlib.Path
    ) -> None:
        hub = tmp_path / "hub"
        fork, feature, tip = _diverged(hub)
        node = tmp_path / "node" / "repo.git"
        node.parent.mkdir()
        bundle = tmp_path / "run-1.bundle"
        branch = stage_repository.BaseBranch(ref="main", sha=tip)

        commit_stage.bundle_commit(
            hub, sha=feature, run_id="run-1", base=None, base_tip=tip, destination=bundle
        )
        tree = _unpack(node, "run-1", feature, branch, NOTHING_HELD, str(bundle))

        assert _git(hub, "bundle", "list-heads", str(bundle)).splitlines() == [
            f"{feature} refs/fleet/bundle/run-1",
            f"{tip} refs/fleet/bundle-base/run-1",
        ]
        assert _git(hub, "for-each-ref", "refs/fleet/") == ""
        assert _git(node, "rev-parse", node_mirror_probe.NODE_TIP_REF) == feature
        assert _git(node, "rev-parse", node_mirror_probe.NODE_BASE_REF) == tip
        assert _git(tree, "rev-parse", "refs/remotes/origin/main") == tip
        assert _git(tree, "rev-parse", "HEAD^") == feature
        committed = _git(hub, "rev-parse", f"{feature}^{{tree}}")
        assert _git(tree, "rev-parse", "HEAD^{tree}") == committed
        base = _git(tree, "merge-base", "HEAD", "refs/remotes/origin/main")
        assert base == fork
        assert _git(tree, "diff", "--name-only", base, "HEAD") == "src/a.ts"
        assert _git(tree, "status", "--porcelain") == ""

    def test_a_node_holding_the_commit_is_sent_and_fetches_the_tip_alone(
        self, tmp_path: pathlib.Path
    ) -> None:
        """git bundle create leaves out a ref its exclusions reach, so the
        commit the node's tip already reaches is not bundled and not
        fetched; the stage still sits on it beside the tip."""
        hub = tmp_path / "hub"
        fork, feature, tip = _diverged(hub)
        node = tmp_path / "node" / "repo.git"
        node.parent.mkdir()
        alone = tmp_path / "alone.bundle"
        commit_stage.bundle_commit(
            hub, sha=feature, run_id="run-1", base=None, base_tip=None, destination=alone
        )
        _sh(
            dialect.checked_script(
                commit_stage.fetch_commands(
                    mirror=str(node),
                    bundle=str(alone),
                    run_id="run-1",
                    commit=True,
                    base_branch=False,
                )
            )
        )
        probed = node_mirror_probe.NodeMirror(tip=feature, held=True, base_held=False)
        tip_only = tmp_path / "tip.bundle"

        commit_stage.bundle_commit(
            hub, sha=None, run_id="run-2", base=feature, base_tip=tip, destination=tip_only
        )
        tree = _unpack(
            node,
            "run-2",
            feature,
            stage_repository.BaseBranch(ref="main", sha=tip),
            probed,
            str(tip_only),
        )

        assert _git(hub, "bundle", "list-heads", str(tip_only)) == (
            f"{tip} refs/fleet/bundle-base/run-2"
        )
        assert _git(node, "rev-parse", node_mirror_probe.NODE_TIP_REF) == feature
        assert _git(node, "rev-parse", node_mirror_probe.NODE_BASE_REF) == tip
        assert _git(tree, "merge-base", "HEAD", "refs/remotes/origin/main") == fork

    def test_a_base_ref_git_refuses_removes_the_commit_s_ref_and_bundles_nothing(
        self, tmp_path: pathlib.Path
    ) -> None:
        runner = FakeRun([ok(""), failed(128, "fatal: cannot lock ref"), ok("")])
        _test_hooks.run = runner

        refusal = r"^git update-ref refs/fleet/bundle-base/run-1 in .*lock ref$"
        with pytest.raises(AppError, match=refusal):
            commit_stage.bundle_commit(
                tmp_path,
                sha=SHA,
                run_id="run-1",
                base=None,
                base_tip=BASE,
                destination=tmp_path / "b.bundle",
            )

        assert runner.calls[2] == (
            "git",
            "-C",
            str(tmp_path),
            *commit_stage.REF_LOCK_WAITS,
            "update-ref",
            "-d",
            "refs/fleet/bundle/run-1",
        )
        assert len(runner.calls) == 3


class TestStagingOneCommitBesideItsBaseBranch:
    TARGET = f"{STAGE_ROOT}/{DEMO_RUN_ID}/{DEMO_RUN_ID}"
    MIRROR = f"{STAGE_ROOT}/cache/mirrors/wagner-austin-MCPs.git"
    STAGED = f"{TARGET}.stage"

    def _unpack_of(self, node: node_mirror_probe.NodeMirror) -> bytes:
        """The unpack script the stage of :data:`SHA` beside :data:`BASE` sends.

        Args:
            node: What the probe said.

        Returns:
            Its bytes.
        """
        return commit_stage.unpack_script(
            mirror=self.MIRROR,
            bundle=f"{self.STAGED}/{names.ARCHIVE_NAME}",
            run_id=DEMO_RUN_ID,
            sha=SHA,
            scope=(),
            staged=self.STAGED,
            target=self.TARGET,
            base_branch=stage_repository.BaseBranch(ref="main", sha=BASE),
            node=node,
        ).encode("utf-8")

    def test_a_node_lacking_both_is_sent_both_and_fetches_both(
        self, tmp_path: pathlib.Path
    ) -> None:
        payload = b"# v2 git bundle\nbundle"
        branch = stage_repository.BaseBranch(ref="main", sha=BASE)
        commit = export_of(tmp_path, payload, base_branch=branch)
        runner = FakeRun(
            [ok(""), ok(f"{TIP}\n")] + [ok("")] * 8 + [ok(staging.digest(payload))] + [ok("")] * 2
        )
        _test_hooks.run = runner

        staged = commit_stage.stage_commit(
            "lavender", stage_root=STAGE_ROOT, run_id=DEMO_RUN_ID, commit=commit
        )

        assert staged["detail"] == (
            f"{len(payload)} bytes, git bundle of {SHA} over {TIP}, beside main at {BASE}, "
            f"to {self.TARGET}"
        )
        prepared = node_mirror_probe.prepare_script(
            target=self.TARGET, staged=self.STAGED, mirror=self.MIRROR, sha=SHA, base_tip=BASE
        )
        assert runner.stdin[0] == prepared.encode("utf-8")
        created = [call for call in runner.calls if call[3:5] == ("bundle", "create")]
        assert [call[-3:] for call in created] == [
            (f"refs/fleet/bundle/{DEMO_RUN_ID}", f"refs/fleet/bundle-base/{DEMO_RUN_ID}", f"^{TIP}")
        ]
        unpack = self._unpack_of(node_mirror_probe.NodeMirror(tip=TIP, held=False, base_held=False))
        assert runner.stdin[11] == unpack
        assert f"update-ref refs/remotes/origin/main {BASE}".encode() in unpack
        assert f"+refs/fleet/bundle/{DEMO_RUN_ID}:refs/fleet/staged".encode() in unpack
        assert f"+refs/fleet/bundle-base/{DEMO_RUN_ID}:refs/fleet/base".encode() in unpack

    def test_a_node_holding_the_commit_is_sent_the_tip_alone(self, tmp_path: pathlib.Path) -> None:
        payload = b"# v2 git bundle\ntip"
        commit = export_of(
            tmp_path, payload, base_branch=stage_repository.BaseBranch(ref="main", sha=BASE)
        )
        held = f"{TIP}\n{node_mirror_probe.HELD_ANSWER}\n"
        runner = FakeRun(
            [ok(""), ok(held)] + [ok("")] * 6 + [ok(staging.digest(payload))] + [ok("")] * 2
        )
        _test_hooks.run = runner

        commit_stage.stage_commit(
            "lavender", stage_root=STAGE_ROOT, run_id=DEMO_RUN_ID, commit=commit
        )

        created = [call for call in runner.calls if call[3:5] == ("bundle", "create")]
        assert [call[-2:] for call in created] == [
            (f"refs/fleet/bundle-base/{DEMO_RUN_ID}", f"^{TIP}")
        ]
        unpack = self._unpack_of(node_mirror_probe.NodeMirror(tip=TIP, held=True, base_held=False))
        assert runner.stdin[9] == unpack
        assert f"refs/fleet/bundle/{DEMO_RUN_ID}:".encode() not in unpack

    def test_a_node_holding_both_is_sent_nothing(self, tmp_path: pathlib.Path) -> None:
        commit = export_of(
            tmp_path, b"unsent", base_branch=stage_repository.BaseBranch(ref="main", sha=BASE)
        )
        held = f"{TIP}\n{node_mirror_probe.HELD_ANSWER}\n{node_mirror_probe.BASE_HELD_ANSWER}\n"
        runner = FakeRun([ok(""), ok(held), ok(""), ok("")])
        _test_hooks.run = runner

        staged = commit_stage.stage_commit(
            "lavender", stage_root=STAGE_ROOT, run_id=DEMO_RUN_ID, commit=commit
        )

        assert staged["detail"] == (
            f"0 bytes, git bundle of {SHA} already in the node's mirror, beside main at {BASE}, "
            f"to {self.TARGET}"
        )
        assert [call[0] for call in runner.calls] == ["ssh"] * 4
        assert commit["bundle"].exists()


class TestPrepareResolvesTheBaseBranch:
    def test_the_hub_fetches_the_branch_into_the_project_s_mirror_and_names_its_tip(
        self, config_path: pathlib.Path
    ) -> None:
        loaded = _config.load_workspace(
            {_config.CONFIG_FLAG: str(write_registry(config_path, _based_document()))}
        )
        admitted = node_prepare.Admitted(
            plan=require_project(loaded.workspace, DEMO_PROJECT), workers=1
        )
        calls: list[tuple[str, ...]] = []

        def hub_git(
            argv: Sequence[str],
            *,
            timeout_seconds: int,
            stdin_bytes: bytes | None = None,
            unset_env: Sequence[str] = (),
            set_env: Sequence[tuple[str, str]] = (),
        ) -> _test_hooks.CommandResult:
            calls.append(tuple(argv))
            if f"{mirror_fetch.COMPANION_REF}^{{commit}}" in argv:
                return ok(f"{BASE}\n")
            return ok("")

        _test_hooks.run = hub_git
        row = queue_job(id="dddddddd-4444-4444-8444-dddddddddddd", status="claimed")
        job = decode_job(row, answer=dump_json_str({"claimed": row}))

        prepared = node_prepare.prepare(loaded, job, admitted=admitted, sha=DEFAULT_SHA)

        assert prepared["base_branch"] == stage_repository.BaseBranch(ref="main", sha=BASE)
        fetches = [call for call in calls if call[-1] == f"+main:{mirror_fetch.COMPANION_REF}"]
        assert [call[2] for call in fetches] == [str(prepared["mirror"])]
