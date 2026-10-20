"""The action archive cache's fill runs for real (MCPs board task ce2b92e0).

The fill script runs under a real bash over checkouts laid out as the runners
leave them, ``<workdir>/<repo>/<repo>/.github/...``, against real git
repositories standing in for github.com: ``git`` is a function that rewrites
``https://github.com/`` to their ``file://`` root, records each fetch, and
otherwise runs the real git, so ``ls-remote``, the shallow fetch and ``git
archive`` all run as rendered and nothing reaches GitHub. Each archive is
opened and held to what the runner asks of it: one top directory holding the
action at that commit. The units and the runner drop-in run in
test_runner_action_cache_units.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import tarfile
import time

import pytest

from fleet.core import runner_action_cache
from tests._host_bash import host_bash

#: ``git`` with every github.com URL pointed at the test's repositories; a
#: fetch is recorded as ``<owner>/<repo> <sha>``, and refused while the
#: case's ``refuse`` marker exists, as a fetch the network drops would be.
FILL_STAND_INS = """
git() {
    local arguments=() argument
    for argument in "$@"; do
        arguments+=("${argument/https:\\/\\/github.com\\//$REMOTES/}")
    done
    if [ "${2:-}" = fetch ]; then
        local url=${@: -2:1}
        echo "${url#https://github.com/} ${@: -1}" >> "$TABLE/fetched"
        if [ -e "$TABLE/refuse" ]; then
            return 128
        fi
    fi
    command git "${arguments[@]}"
}
"""


def _git(cwd: pathlib.Path, *arguments: str) -> str:
    """Run git in a test repository.

    Args:
        cwd: The repository.
        arguments: Git's arguments.

    Returns:
        Its standard output, stripped.
    """
    ran = subprocess.run(
        [
            "git",
            "-c",
            "user.name=fleet-test",
            "-c",
            "user.email=fleet-test@example.com",
            *arguments,
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return ran.stdout.strip()


class Fill:
    """A laid-out host: repositories standing in for GitHub, checkouts under workdirs, a cache.

    The repositories: actions/checkout with a lightweight tag v7, actions/
    setup-node with an annotated tag v7, github/codeql-action with a branch
    v3, and actions/cache, which a workflow names by its full sha. Each
    holds one commit whose action.yml names the repository.
    """

    def __init__(self, root: pathlib.Path) -> None:
        """Lay out the repositories and an empty cache.

        Args:
            root: The test's directory.
        """
        self.root = root
        self.table = root / "table"
        self.table.mkdir()
        self.cache = root / "cache"
        remotes = root / "remotes"
        self.sha = {
            "actions/checkout": self._repository(remotes, "actions/checkout", "tag", "v7"),
            "actions/setup-node": self._repository(remotes, "actions/setup-node", "tag -a", "v7"),
            "github/codeql-action": self._repository(
                remotes, "github/codeql-action", "branch", "v3"
            ),
            "actions/cache": self._repository(remotes, "actions/cache", "", ""),
        }
        script = (
            f"TABLE='{self.table.as_posix()}'\nREMOTES='{remotes.as_uri()}'\n"
            + FILL_STAND_INS
            + runner_action_cache.FILL_SCRIPT
        )
        (root / "fill.sh").write_bytes(script.encode())

    @staticmethod
    def _repository(remotes: pathlib.Path, name: str, ref_kind: str, ref: str) -> str:
        """One repository with one commit, and the ref the workflows name.

        Args:
            remotes: Where the repositories live.
            name: ``owner/repo``.
            ref_kind: ``tag``, ``tag -a``, ``branch``, or empty for none.
            ref: The ref's name.

        Returns:
            The commit's sha.
        """
        path = remotes / name
        path.mkdir(parents=True)
        _git(path, "init", "-q", "-b", "main")
        (path / "action.yml").write_text(f"name: {name}\n", encoding="utf-8")
        _git(path, "add", "action.yml")
        _git(path, "commit", "-q", "-m", name)
        if ref_kind == "tag -a":
            _git(path, "tag", "-a", "-m", "release", ref)
        elif ref_kind != "":
            _git(path, ref_kind, ref)
        return _git(path, "rev-parse", "HEAD")

    def workflow(self) -> str:
        """The workflow a checkout holds.

        A lightweight tag, a quoted annotated tag in a list item, a subpath
        action on a branch, a pinned sha, a local composite action, a docker
        image and a comment, the last three never fetched.

        Returns:
            The workflow's text.
        """
        return (
            "name: Check\n"
            "jobs:\n"
            "  test:\n"
            "    steps:\n"
            "      - uses: actions/checkout@v7\n"
            '      - uses: "actions/setup-node@v7"\n'
            "      - uses: github/codeql-action/init@v3  # a subpath, on a branch\n"
            f"      - uses: actions/cache@{self.sha['actions/cache']}\n"
            "      - uses: ./.github/actions/setup-workspace\n"
            "      - uses: docker://alpine:3\n"
            "      # uses: actions/never@v1 is a comment and is not read\n"
        )

    def checkout(self, workdir: str, repo: str, workflow: str, composite: str | None) -> None:
        """A checkout the way a runner leaves it under its work tree.

        Args:
            workdir: The work tree's name under the test's directory.
            repo: The checked-out repository's name.
            workflow: Its one workflow.
            composite: Its composite setup action, or None for none.
        """
        github = self.root / workdir / repo / repo / ".github"
        (github / "workflows").mkdir(parents=True)
        (github / "workflows" / "check.yml").write_text(workflow, encoding="utf-8")
        if composite is not None:
            (github / "actions" / "setup-workspace").mkdir(parents=True)
            (github / "actions" / "setup-workspace" / "action.yml").write_text(
                composite, encoding="utf-8"
            )

    def run(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        """Run the fill.

        Args:
            arguments: Its arguments.

        Returns:
            The finished process.
        """
        return subprocess.run(
            [host_bash(), "fill.sh", *arguments],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )

    def fetched(self) -> list[str]:
        """Every fetch the fill made, in order.

        Returns:
            ``<owner>/<repo> <sha>`` per fetch, none when it fetched nothing.
        """
        log = self.table / "fetched"
        return log.read_text(encoding="utf-8").splitlines() if log.exists() else []

    def archives(self) -> list[str]:
        """Every file in the cache, relative to it.

        Returns:
            The sorted ``<owner>_<repo>/<file>`` paths.
        """
        return sorted(
            p.relative_to(self.cache).as_posix() for p in self.cache.rglob("*") if p.is_file()
        )


def _tar_members(archive: pathlib.Path) -> dict[str, bytes]:
    """A tar.gz archive's files and their bytes.

    Args:
        archive: The archive.

    Returns:
        Each regular file's path to its bytes.
    """
    members: dict[str, bytes] = {}
    with tarfile.open(archive, "r:gz") as opened:
        for member in opened.getmembers():
            extracted = opened.extractfile(member)
            if member.isfile() and extracted is not None:
                members[member.name] = extracted.read()
    return members


def _summary(
    examined: int,
    workdirs: int,
    cached: int,
    held: int,
    failed: int,
    pruned: int,
) -> str:
    """The fill's last line, into the cases' ``cache``.

    Args:
        examined: References read.
        workdirs: Workdirs given.
        cached: Archives built.
        held: Archives already there.
        failed: References that resolved to nothing or were not built.
        pruned: Archives removed.

    Returns:
        The line, with its newline.
    """
    return (
        f"fleet-action-cache-fill: into cache: {examined} action reference(s) "
        f"in the checkouts of {workdirs} workdir(s): {cached} cached now, {held} already held, "
        f"{failed} failed; {pruned} unnamed for {runner_action_cache.PRUNE_DAYS} days removed\n"
    )


class TestTheFillRunsForReal:
    """One archive per action version, built once, kept, and pruned only when long unnamed."""

    def test_every_action_is_built_once_as_the_runner_reads_it_and_a_second_pass_fetches_nothing(
        self, tmp_path: pathlib.Path
    ) -> None:
        host = Fill(tmp_path)
        composite = "runs:\n  using: composite\n  steps:\n    - uses: actions/checkout@v7\n"
        host.checkout("one", "MCPs", host.workflow(), composite)
        host.checkout("two", "API", "steps:\n  - uses: actions/checkout@v7\n", None)
        first = host.run("cache", "one", "two")
        assert first.returncode == 0, first.stderr
        sha = host.sha
        # setup-node's v7 is an annotated tag: the archive is of the commit it
        # peels to, the sha the runner resolves, never the tag object's.
        assert host.fetched() == [
            f"actions/cache {sha['actions/cache']}",
            f"actions/checkout {sha['actions/checkout']}",
            f"actions/setup-node {sha['actions/setup-node']}",
            f"github/codeql-action {sha['github/codeql-action']}",
        ]
        assert host.archives() == sorted(
            f"{name.replace('/', '_')}/{commit}.tar.gz" for name, commit in sha.items()
        )
        checkout = sha["actions/checkout"]
        archive = host.cache / "actions_checkout" / f"{checkout}.tar.gz"
        # One top directory, which the runner unwraps, holding the action.
        assert _tar_members(archive) == {
            f"actions-checkout-{checkout[:12]}/action.yml": b"name: actions/checkout\n"
        }
        assert first.stdout.splitlines()[1] == (
            f"fleet-action-cache-fill: cached actions/checkout@v7 at {checkout}"
        )
        assert first.stdout.endswith(_summary(4, 2, 4, 0, 0, 0))
        second = host.run("cache", "one", "two")
        assert second.returncode == 0, second.stderr
        assert len(host.fetched()) == 4
        assert second.stdout == _summary(4, 2, 0, 4, 0, 0)

    def test_a_ref_naming_nothing_fails_the_pass_after_the_others_are_cached(
        self, tmp_path: pathlib.Path
    ) -> None:
        host = Fill(tmp_path)
        workflow = (
            "steps:\n"
            "  - uses: actions/checkout@v7\n"
            "  - uses: actions/setup-node@v99\n"
            "  - uses: nobody/gone@v1\n"
        )
        host.checkout("one", "MCPs", workflow, None)
        ran = host.run("cache", "one")
        assert ran.returncode == 1
        assert ran.stdout.splitlines()[1:] == [
            "fleet-action-cache-fill: actions/setup-node@v99 names no tag, branch or commit "
            "at https://github.com/actions/setup-node",
            "fleet-action-cache-fill: nobody/gone@v1 names no tag, branch or commit "
            "at https://github.com/nobody/gone",
            _summary(3, 1, 1, 0, 2, 0).rstrip("\n"),
        ]
        assert host.archives() == [f"actions_checkout/{host.sha['actions/checkout']}.tar.gz"]

    def test_a_refused_fetch_leaves_no_partial_archive_and_fails_the_pass(
        self, tmp_path: pathlib.Path
    ) -> None:
        host = Fill(tmp_path)
        host.checkout("one", "MCPs", "steps:\n  - uses: actions/checkout@v7\n", None)
        (host.table / "refuse").write_text("", encoding="utf-8")
        ran = host.run("cache", "one")
        assert ran.returncode == 1
        assert ran.stdout == (
            "fleet-action-cache-fill: could not fetch actions/checkout@v7 at "
            f"{host.sha['actions/checkout']} from https://github.com/actions/checkout\n"
            + _summary(1, 1, 0, 0, 1, 0)
        )
        assert host.archives() == []

    def test_only_an_archive_unnamed_past_the_prune_age_is_removed(
        self, tmp_path: pathlib.Path
    ) -> None:
        host = Fill(tmp_path)
        host.checkout("one", "MCPs", "steps:\n  - uses: actions/checkout@v7\n", None)
        checkout = host.sha["actions/checkout"]
        days = 86400
        old = time.time() - (runner_action_cache.PRUNE_DAYS + 10) * days
        named = host.cache / "actions_checkout" / f"{checkout}.tar.gz"
        stale = host.cache / "actions_checkout" / f"{'e' * 40}.tar.gz"
        recent = host.cache / "actions_setup-node" / f"{'f' * 40}.tar.gz"
        not_an_archive = host.cache / "actions_setup-node" / "notes.txt"
        for archive, when in (
            (named, old),
            (stale, old),
            (recent, time.time() - 5 * days),
            (not_an_archive, old),
        ):
            archive.parent.mkdir(parents=True, exist_ok=True)
            archive.write_text("held", encoding="utf-8")
            os.utime(archive, (when, when))
        ran = host.run("cache", "one")
        assert ran.returncode == 0, ran.stderr
        assert host.fetched() == []
        assert ran.stdout == _summary(1, 1, 0, 1, 0, 1)
        # Only an archive is the fill's to prune.
        assert host.archives() == [
            f"actions_checkout/{checkout}.tar.gz",
            f"actions_setup-node/{'f' * 40}.tar.gz",
            "actions_setup-node/notes.txt",
        ]
        # Named this pass, so its age starts again.
        assert named.stat().st_mtime > time.time() - days

    def test_a_host_with_no_checkouts_examines_nothing_and_removes_nothing_recent(
        self, tmp_path: pathlib.Path
    ) -> None:
        host = Fill(tmp_path)
        (tmp_path / "empty").mkdir()
        held = host.cache / "actions_checkout" / f"{host.sha['actions/checkout']}.tar.gz"
        held.parent.mkdir(parents=True)
        held.write_text("held", encoding="utf-8")
        ran = host.run("cache", "empty")
        assert ran.returncode == 0, ran.stderr
        assert ran.stdout == _summary(0, 1, 0, 0, 0, 0)
        assert host.archives() == [f"actions_checkout/{host.sha['actions/checkout']}.tar.gz"]

    @pytest.mark.parametrize("arguments", [(), ("cache",)])
    def test_a_fill_without_a_cache_and_a_workdir_is_refused_with_its_usage(
        self, tmp_path: pathlib.Path, arguments: tuple[str, ...]
    ) -> None:
        host = Fill(tmp_path)
        host.checkout("one", "MCPs", "steps:\n  - uses: actions/checkout@v7\n", None)
        ran = host.run(*arguments)
        assert ran.returncode == 2
        assert ran.stderr == (
            "fleet-action-cache-fill: usage: fleet-action-cache-fill CACHE WORKDIR...\n"
        )
        assert host.fetched() == []
