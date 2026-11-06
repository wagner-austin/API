"""A commit staged beside its project's base branch (MCPs board task a48b5236).

A staged tree was one root commit in a repository with no branch, so a check
asking which files its commit changes against ``origin/main`` (the MCPs
TypeScript packages' mutation step) had nothing to ask. A project declaring
``base_branch`` now has the branch's tip resolved on the hub, carried in the
commit's bundle and written as ``refs/remotes/origin/<branch>`` in the stage,
whose own commit is the queued commit's child. The git half is real, as in
``tests/test_commit_stage.py``: a hub repository whose main moved on after a
feature branch left it, the bundle written by real ``git bundle create``, and
the node's scripts run by real ``/bin/sh``.
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
    stage_repository,
    staging,
)
from tests._node_agent_fixtures import NPM_CI, REMOTE, _credentials_in_env
from tests._queue_fakes import DEFAULT_SHA, queue_job
from tests._registry import write_registry
from tests.conftest import DEMO_PROJECT, DEMO_RUN_ID, FakeRun, failed, ok, workspace_document
from tests.test_commit_stage import SHA, STAGE_ROOT, TIP, _commit, _git, _sh, export_of

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
    node: pathlib.Path, run_id: str, sha: str, branch: stage_repository.BaseBranch, bundle: str
) -> pathlib.Path:
    """Unpack one bundle into a run's export, as a node does.

    Args:
        node: The node's mirror.
        run_id: The run.
        sha: The queued commit.
        branch: Its base branch and tip.
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

        commit_stage.bundle_commit(
            hub, sha=feature, run_id="run-1", base=None, base_tip=tip, destination=bundle
        )
        tree = _unpack(
            node, "run-1", feature, stage_repository.BaseBranch(ref="main", sha=tip), str(bundle)
        )

        assert _git(hub, "bundle", "list-heads", str(bundle)).splitlines() == [
            f"{feature} refs/fleet/bundle/run-1",
            f"{tip} refs/fleet/bundle-base/run-1",
        ]
        assert _git(hub, "for-each-ref", "refs/fleet/") == ""
        assert _git(node, "rev-parse", commit_stage.NODE_TIP_REF) == feature
        assert _git(node, "rev-parse", commit_stage.NODE_BASE_REF) == tip
        assert _git(tree, "rev-parse", "refs/remotes/origin/main") == tip
        assert _git(tree, "rev-parse", "HEAD^") == feature
        committed = _git(hub, "rev-parse", f"{feature}^{{tree}}")
        assert _git(tree, "rev-parse", "HEAD^{tree}") == committed
        base = _git(tree, "merge-base", "HEAD", "refs/remotes/origin/main")
        assert base == fork
        assert _git(tree, "diff", "--name-only", base, "HEAD") == "src/a.ts"
        assert _git(tree, "status", "--porcelain") == ""

    def test_a_tip_the_node_already_reaches_is_held_and_one_it_lacks_is_not(
        self, tmp_path: pathlib.Path
    ) -> None:
        hub = tmp_path / "hub"
        _fork, feature, tip = _diverged(hub)
        node = tmp_path / "node" / "repo.git"
        node.parent.mkdir()
        alone = tmp_path / "alone.bundle"
        commit_stage.bundle_commit(
            hub, sha=feature, run_id="run-1", base=None, base_tip=None, destination=alone
        )
        _sh(
            dialect.checked_script(
                commit_stage.fetch_commands(
                    mirror=str(node), bundle=str(alone), run_id="run-1", base_branch=False
                )
            )
        )

        def asked(base_tip: str | None) -> commit_stage.NodeMirror:
            script = commit_stage.prepare_script(
                target=str(tmp_path / "t"),
                staged=str(tmp_path / "s"),
                mirror=str(node),
                sha=feature,
                base_tip=base_tip,
            )
            return commit_stage.read_node_mirror(_sh(script))

        assert asked(None) == commit_stage.NodeMirror(tip=feature, held=True)
        assert asked(tip) == commit_stage.NodeMirror(tip=feature, held=False)
        both = tmp_path / "both.bundle"
        commit_stage.bundle_commit(
            hub, sha=feature, run_id="run-2", base=feature, base_tip=tip, destination=both
        )
        _unpack(node, "run-2", feature, stage_repository.BaseBranch(ref="main", sha=tip), str(both))
        assert asked(tip) == commit_stage.NodeMirror(tip=feature, held=True)

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

    def test_the_probe_bundle_and_unpack_each_carry_the_branch(
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

        staged_dir = f"{self.TARGET}.stage"
        assert staged["detail"] == (
            f"{len(payload)} bytes, git bundle of {SHA} over {TIP}, beside main at {BASE}, "
            f"to {self.TARGET}"
        )
        prepared = commit_stage.prepare_script(
            target=self.TARGET, staged=staged_dir, mirror=self.MIRROR, sha=SHA, base_tip=BASE
        )
        assert runner.stdin[0] == prepared.encode("utf-8")
        created = [call for call in runner.calls if call[3:5] == ("bundle", "create")]
        assert [call[-3:] for call in created] == [
            (f"refs/fleet/bundle/{DEMO_RUN_ID}", f"refs/fleet/bundle-base/{DEMO_RUN_ID}", f"^{TIP}")
        ]
        unpack = commit_stage.unpack_script(
            mirror=self.MIRROR,
            bundle=f"{staged_dir}/{names.ARCHIVE_NAME}",
            run_id=DEMO_RUN_ID,
            sha=SHA,
            scope=(),
            staged=staged_dir,
            target=self.TARGET,
            base_branch=branch,
        )
        assert runner.stdin[11] == unpack.encode("utf-8")
        assert f"update-ref refs/remotes/origin/main {BASE}" in unpack
        assert f"+refs/fleet/bundle-base/{DEMO_RUN_ID}:refs/fleet/base" in unpack


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
