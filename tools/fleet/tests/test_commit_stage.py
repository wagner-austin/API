"""Staging a commit from the node's mirror, sending only what it lacks (MCPs board task 1e0ee821).

The git half is real: real repositories in a temporary directory stand in
for the hub's mirror and the node's, the hub's bundle is written by real
``git bundle create``, and the node's prepare and unpack scripts are run by
real ``/bin/sh`` exactly as a node runs them, so what is asserted is that a
bundle of the commit less the node's tip carries what the node lacks and
nothing it holds, and that the tree it unpacks is the commit's. The stage's
order over ssh is asserted with only the transport faked, as
``tests/test_staging.py`` asserts a working tree's.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.core import _test_hooks, commit_stage, dialect, names, remote, staging
from fleet.core.ssh_options import SSH_OPTIONS
from tests.conftest import DEMO_RUN_ID, FakeRun, failed, ok

#: The stage root every node declares.
STAGE_ROOT = "/home/corvis/fleet/stage"

#: The commit the faked stages are asked for, and a node mirror's tip.
SHA = "4e3c6bc1d9f0a7b2c3e4f5061728394a5b6c7d8e"
TIP = "9f1c0b7a2d3e4f5061728394a5b6c7d8e9f01234"

#: The deadline for the real git and sh these tests run over tiny trees.
GIT_SECONDS = 60


def _git(repo: pathlib.Path, *args: str) -> str:
    """Run real git in a test's repository.

    Args:
        repo: The repository.
        *args: The arguments after ``git -C <repo>``.

    Returns:
        Its standard output, stripped.
    """
    done = subprocess.run(
        ("git", "-C", str(repo), *args),
        check=True,
        capture_output=True,
        text=True,
        timeout=GIT_SECONDS,
    )
    return done.stdout.strip()


def _commit(repo: pathlib.Path, files: dict[str, str]) -> str:
    """Write files into a repository and commit them.

    Args:
        repo: The repository, made on its first commit.
        files: Repo-relative paths and their text.

    Returns:
        The new commit.
    """
    if not (repo / ".git").is_dir():
        repo.mkdir(parents=True)
        _git(repo, "init", "--quiet", "-b", "main")
    for name, text in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    _git(repo, "add", "--all")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "--quiet", "-m", "c")
    return _git(repo, "rev-parse", "HEAD")


def _sh(script: str) -> str:
    """Run a script as a node runs one, requiring it to pass.

    Args:
        script: Its text.

    Returns:
        What it printed.
    """
    done = subprocess.run(
        ("/bin/sh", "-c", script), capture_output=True, text=True, timeout=GIT_SECONDS, check=False
    )
    assert done.returncode == 0, done.stderr
    return done.stdout


def _node_mirror_at(hub: pathlib.Path, node: pathlib.Path, run_id: str, base: str | None) -> None:
    """Bundle the hub's HEAD less ``base`` and fetch it into the node's mirror, as a stage does.

    Args:
        hub: The hub's repository.
        node: The node's mirror, made when absent.
        run_id: The run whose ref carries the bundle.
        base: The node mirror's tip, or None.
    """
    head = _git(hub, "rev-parse", "HEAD")
    bundle = node.parent / f"{run_id}.bundle"
    commit_stage.bundle_commit(hub, sha=head, run_id=run_id, base=base, destination=bundle)
    commands = commit_stage.unpack_commands(
        mirror=str(node),
        bundle=str(bundle),
        run_id=run_id,
        sha=head,
        scope=(),
        staged=str(node.parent / f"{run_id}.stage"),
        target=str(node.parent / run_id),
    )
    (node.parent / f"{run_id}.stage").mkdir()
    (node.parent / run_id).mkdir()
    _sh(dialect.checked_script(commands))


class TestTheNodeMirrorSaysWhatItHolds:
    def test_a_node_without_the_mirror_gets_its_directories_and_says_nothing(
        self, tmp_path: pathlib.Path
    ) -> None:
        target = tmp_path / "run" / "run"
        said = _sh(
            commit_stage.prepare_script(
                target=str(target),
                staged=f"{target}.stage",
                mirror=str(tmp_path / "mirrors" / "repo.git"),
                sha=SHA,
            )
        )

        assert said == ""
        assert target.is_dir()
        assert pathlib.Path(f"{target}.stage").is_dir()
        assert commit_stage.read_node_mirror(said) == commit_stage.NodeMirror(tip=None, held=False)

    def test_a_mirror_names_its_tip_and_whether_it_holds_the_commit(
        self, tmp_path: pathlib.Path
    ) -> None:
        hub = tmp_path / "hub"
        first = _commit(hub, {"a.txt": "one\n"})
        node = tmp_path / "node" / "repo.git"
        node.parent.mkdir()
        _node_mirror_at(hub, node, "run-1", None)
        second = _commit(hub, {"a.txt": "two\n"})

        def asked(sha: str) -> commit_stage.NodeMirror:
            script = commit_stage.prepare_script(
                target=str(tmp_path / "t"), staged=str(tmp_path / "s"), mirror=str(node), sha=sha
            )
            return commit_stage.read_node_mirror(_sh(script))

        assert asked(first) == commit_stage.NodeMirror(tip=first, held=True)
        assert asked(second) == commit_stage.NodeMirror(tip=first, held=False)

    def test_words_that_are_no_commit_are_no_tip(self) -> None:
        said = "fatal: something\nheld\n"

        assert commit_stage.read_node_mirror(said) == commit_stage.NodeMirror(tip=None, held=True)


class TestTheBundleCarriesWhatTheNodeLacks:
    def test_a_known_tip_leaves_its_history_out(self, tmp_path: pathlib.Path) -> None:
        hub = tmp_path / "hub"
        counted = "".join(f"{index:08x}" for index in range(25_000))
        first = _commit(hub, {"big.txt": counted, "a.txt": "one\n"})
        node = tmp_path / "node" / "repo.git"
        node.parent.mkdir()
        _node_mirror_at(hub, node, "run-1", None)
        second = _commit(hub, {"a.txt": "two\n"})
        delta = tmp_path / "delta.bundle"
        whole = tmp_path / "whole.bundle"

        sent = commit_stage.bundle_commit(
            hub, sha=second, run_id="run-2", base=first, destination=delta
        )
        commit_stage.bundle_commit(hub, sha=second, run_id="run-3", base=None, destination=whole)

        assert sent == delta.read_bytes()
        # The node's mirror holds the base, so the bundle verifies there; an
        # empty repository lacks it, so the bundle carries none of the base's
        # history, and its 200 kB file not at all.
        _git(node, "bundle", "verify", "--quiet", str(delta))
        empty = tmp_path / "empty.git"
        _git(tmp_path, "init", "--bare", "--quiet", str(empty))
        refused = subprocess.run(
            ("git", "-C", str(empty), "bundle", "verify", "--quiet", str(delta)),
            capture_output=True,
            text=True,
            timeout=GIT_SECONDS,
            check=False,
        )
        assert refused.returncode != 0
        assert len(sent) * 10 < whole.stat().st_size
        assert _git(hub, "bundle", "list-heads", str(delta)) == f"{second} refs/fleet/bundle/run-2"
        # Each run's ref goes once its bundle is written.
        assert _git(hub, "for-each-ref", "refs/fleet/bundle/") == ""

    def test_no_tip_or_one_the_hub_lacks_sends_the_whole_history(
        self, tmp_path: pathlib.Path
    ) -> None:
        hub = tmp_path / "hub"
        sha = _commit(hub, {"a.txt": "one\n"})
        empty = tmp_path / "empty.git"
        _git(tmp_path, "init", "--bare", "--quiet", str(empty))

        for base in (None, TIP):
            bundle = tmp_path / f"{base}.bundle"
            commit_stage.bundle_commit(hub, sha=sha, run_id="run-1", base=base, destination=bundle)

            _git(empty, "bundle", "verify", "--quiet", str(bundle))

    def test_a_bundle_git_refuses_still_removes_the_run_s_ref(self, tmp_path: pathlib.Path) -> None:
        runner = FakeRun([ok(""), failed(128, "fatal: Refusing to create empty bundle."), ok("")])
        _test_hooks.run = runner
        mirror = tmp_path / "hub.git"

        with pytest.raises(AppError) as raised:
            commit_stage.bundle_commit(
                mirror, sha=SHA, run_id="run-1", base=None, destination=tmp_path / "b.bundle"
            )

        assert raised.value.code is FleetErrorCode.EXPORT_FAILED
        assert raised.value.message == (
            f"git bundle create {SHA} from {mirror}: fatal: Refusing to create empty bundle."
        )
        ref = "refs/fleet/bundle/run-1"
        assert runner.calls[2] == (
            "git",
            "-C",
            str(mirror),
            *commit_stage.REF_LOCK_WAITS,
            "update-ref",
            "-d",
            ref,
        )

    def test_a_ref_git_refuses_bundles_nothing(self, tmp_path: pathlib.Path) -> None:
        runner = FakeRun([failed(128, "fatal: cannot lock ref")])
        _test_hooks.run = runner

        refusal = r"^git update-ref refs/fleet/bundle/run-1 in .*lock ref$"
        with pytest.raises(AppError, match=refusal):
            commit_stage.bundle_commit(
                tmp_path, sha=SHA, run_id="run-1", base=None, destination=tmp_path / "b.bundle"
            )

        assert len(runner.calls) == 1


class TestTheNodeUnpacksTheCommitFromItsMirror:
    def test_the_first_stage_the_next_and_a_held_commit_each_land_the_commit_s_tree(
        self, tmp_path: pathlib.Path
    ) -> None:
        hub = tmp_path / "hub"
        first = _commit(hub, {"src/a.py": "print(1)\n", "data/big.bin": "d\n"})
        node = tmp_path / "node" / "repo.git"
        node.parent.mkdir()

        _node_mirror_at(hub, node, "run-1", None)
        second = _commit(hub, {"src/a.py": "print(2)\n"})
        _node_mirror_at(hub, node, "run-2", first)

        assert (node.parent / "run-1" / "src" / "a.py").read_text(encoding="utf-8") == "print(1)\n"
        assert (node.parent / "run-2" / "src" / "a.py").read_text(encoding="utf-8") == "print(2)\n"
        assert _git(node, "rev-parse", commit_stage.NODE_TIP_REF) == second
        # Each tree is a repository of one commit under the export identity.
        assert _git(node.parent / "run-2", "log", "--format=%ae") == "fleet@corvis.invalid"
        held = node.parent / "run-3"
        (held.parent / "run-3.stage").mkdir()
        held.mkdir()
        _sh(
            dialect.checked_script(
                commit_stage.unpack_commands(
                    mirror=str(node),
                    bundle=None,
                    run_id="run-3",
                    sha=first,
                    scope=(".", ":(exclude)data"),
                    staged=f"{held}.stage",
                    target=str(held),
                )
            )
        )
        assert (held / "src" / "a.py").read_text(encoding="utf-8") == "print(1)\n"
        assert not (held / "data").exists()
        # A commit the mirror held was fetched into nothing: the tip stays.
        assert _git(node, "rev-parse", commit_stage.NODE_TIP_REF) == second

    def test_the_tree_s_objects_are_the_mirror_s_own_files_and_the_add_writes_none(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The export is cloned from the node's mirror by hard links before
        the tree lands, so the add finds each blob it hashes already there
        and writes it again nowhere (MCPs board task 5280a25d: 0.83 s for
        MCPs' add on pendragon, against 2.11 s writing all of them). The
        repository names no remote, and its one commit is the archive's."""
        hub = tmp_path / "hub"
        _commit(hub, {"src/a.py": "print(1)\n", "src/b.py": "print(2)\n"})
        node = tmp_path / "node" / "repo.git"
        node.parent.mkdir()

        _node_mirror_at(hub, node, "run-1", None)

        tree = node.parent / "run-1"
        # A bundle is fetched as one pack, which the clone links.
        packs = sorted((node / "objects" / "pack").glob("*.pack"))
        linked = sorted((tree / ".git" / "objects" / "pack").glob("*.pack"))
        assert len(packs) == 1
        assert [pack.stat().st_ino for pack in linked] == [packs[0].stat().st_ino]
        for path in ("src/a.py", "src/b.py"):
            blob = _git(tree, "rev-parse", f"HEAD:{path}")
            assert not (tree / ".git" / "objects" / blob[:2] / blob[2:]).exists()
        assert _git(tree, "remote") == ""
        assert _git(tree, "status", "--porcelain") == ""
        assert _git(tree, "rev-parse", "HEAD^{tree}") == _git(hub, "rev-parse", "HEAD^{tree}")

    def test_a_bundle_whose_base_the_node_lacks_stops_before_any_tree(
        self, tmp_path: pathlib.Path
    ) -> None:
        hub = tmp_path / "hub"
        first = _commit(hub, {"a.txt": "one\n"})
        second = _commit(hub, {"a.txt": "two\n"})
        bundle = tmp_path / "b.bundle"
        commit_stage.bundle_commit(hub, sha=second, run_id="run-1", base=first, destination=bundle)
        target = tmp_path / "run-1"
        target.mkdir()
        script = dialect.checked_script(
            commit_stage.unpack_commands(
                mirror=str(tmp_path / "node.git"),
                bundle=str(bundle),
                run_id="run-1",
                sha=second,
                scope=(),
                staged=str(tmp_path),
                target=str(target),
            )
        )

        done = subprocess.run(
            ("/bin/sh", "-c", script),
            capture_output=True,
            text=True,
            timeout=GIT_SECONDS,
            check=False,
        )

        assert done.returncode != 0
        assert list(target.iterdir()) == []


def _scp(remote_path: str, source: pathlib.Path) -> tuple[str, ...]:
    """The argv staging hands scp for the bundle.

    Args:
        remote_path: Where the bundle lands on the node.
        source: The local bundle.

    Returns:
        The argv :func:`fleet.core.remote.send_file` runs.
    """
    return ("scp", "-q", *SSH_OPTIONS, str(source), f"lavender:{remote_path}")


def _export(tmp_path: pathlib.Path, payload: bytes) -> commit_stage.CommitExport:
    """A commit to stage, its bundle already where the faked git would write it.

    Args:
        tmp_path: The test's directory.
        payload: The bundle's bytes.

    Returns:
        The export.
    """
    bundle = tmp_path / "attempt" / f"wagner-austin-MCPs-{SHA}.bundle"
    bundle.parent.mkdir()
    bundle.write_bytes(payload)
    return commit_stage.CommitExport(
        mirror=tmp_path / "hub.git", key="wagner-austin-MCPs", sha=SHA, scope=(), bundle=bundle
    )


class TestStagingOneCommit:
    TARGET = f"{STAGE_ROOT}/{DEMO_RUN_ID}/{DEMO_RUN_ID}"
    MIRROR = f"{STAGE_ROOT}/cache/mirrors/wagner-austin-MCPs.git"

    def test_a_node_without_the_commit_is_sent_its_bundle_verified_then_unpacks_it(
        self, tmp_path: pathlib.Path
    ) -> None:
        payload = b"# v2 git bundle\nbundle"
        commit = _export(tmp_path, payload)
        runner = FakeRun(
            [
                ok(""),
                ok(f"{TIP}\n"),
                ok(""),
                ok(""),
                ok(""),
                ok(""),
                ok(""),
                ok(""),
                ok(staging.digest(payload)),
                ok(""),
                ok(""),
            ]
        )
        _test_hooks.run = runner

        staged = commit_stage.stage_commit(
            "lavender", stage_root=STAGE_ROOT, run_id=DEMO_RUN_ID, commit=commit
        )

        staged_dir = f"{self.TARGET}.stage"
        assert staged == commit_stage.StagedCommit(
            target=self.TARGET,
            detail=f"{len(payload)} bytes, git bundle of {SHA} over {TIP}, to {self.TARGET}",
        )
        assert runner.calls[1][-1] == f"{STAGE_ROOT}/mkdir-{DEMO_RUN_ID}.sh"
        prepared = commit_stage.prepare_script(
            target=self.TARGET, staged=staged_dir, mirror=self.MIRROR, sha=SHA
        )
        assert runner.stdin[0] == prepared.encode("utf-8")
        hub = str(tmp_path / "hub.git")
        assert runner.calls[3] == ("git", "-C", hub, "cat-file", "-e", f"{TIP}^{{commit}}")
        assert runner.calls[4][-2:] == (f"refs/fleet/bundle/{DEMO_RUN_ID}", f"^{TIP}")
        assert runner.calls[6] == _scp(f"{staged_dir}/{names.ARCHIVE_NAME}", commit["bundle"])
        unpack = commit_stage.unpack_commands(
            mirror=self.MIRROR,
            bundle=f"{staged_dir}/{names.ARCHIVE_NAME}",
            run_id=DEMO_RUN_ID,
            sha=SHA,
            scope=(),
            staged=staged_dir,
            target=self.TARGET,
        )
        assert runner.stdin[9] == dialect.checked_script(unpack).encode("utf-8")
        assert runner.timeouts[10] == remote.send_deadline_seconds(len(payload))
        assert not commit["bundle"].exists()

    def test_a_commit_the_node_s_mirror_holds_is_sent_nothing(self, tmp_path: pathlib.Path) -> None:
        commit = _export(tmp_path, b"unsent")
        runner = FakeRun([ok(""), ok(f"{TIP}\nheld\n"), ok(""), ok("")])
        _test_hooks.run = runner

        staged = commit_stage.stage_commit(
            "lavender", stage_root=STAGE_ROOT, run_id=DEMO_RUN_ID, commit=commit
        )

        assert staged["detail"] == (
            f"0 bytes, git bundle of {SHA} already in the node's mirror, to {self.TARGET}"
        )
        assert [call[0] for call in runner.calls] == ["ssh"] * 4
        assert b" fetch " not in (runner.stdin[2] or b"")
        assert runner.timeouts[3] == remote.SSH_TIMEOUT_SECONDS
        # Never bundled, so it is still the attempt's, which the launch removes.
        assert commit["bundle"].exists()

    def test_a_mismatched_digest_unpacks_nothing_and_leaves_no_bundle(
        self, tmp_path: pathlib.Path
    ) -> None:
        commit = _export(tmp_path, b"# v2 git bundle\nbundle")
        runner = FakeRun([ok(""), ok(""), ok(""), ok(""), ok(""), ok(""), ok(""), ok("0" * 64)])
        _test_hooks.run = runner

        with pytest.raises(AppError) as raised:
            commit_stage.stage_commit(
                "lavender", stage_root=STAGE_ROOT, run_id=DEMO_RUN_ID, commit=commit
            )

        assert raised.value.code is FleetErrorCode.STAGE_DIGEST_MISMATCH
        assert raised.value.message.startswith(f"lavender received the bundle of {SHA} digesting ")
        assert not commit["bundle"].exists()
        assert len(runner.calls) == 8

    def test_the_node_s_mirror_outlives_every_run_under_its_cache(self) -> None:
        assert names.node_mirror(STAGE_ROOT, "wagner-austin-MCPs") == self.MIRROR
