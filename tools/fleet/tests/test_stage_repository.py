"""The git commands that make a staged tree, or a companion, a repository."""

from __future__ import annotations

import hashlib
import io
import pathlib
import subprocess
import zipfile

import pytest

from fleet.core import names, stage_repository

#: The loose-object directory ``git gc --auto`` samples to estimate how many
#: loose objects a repository holds: past ``ceil(gc.auto / 256)`` objects in
#: it, gc packs and prunes them all.
SAMPLED_DIRECTORY = "17"


def test_a_staged_tree_is_initialised_kept_from_auto_gc_indexed_and_committed() -> None:
    """git init is what makes ruff honour .gitignore on the node, git add is
    what makes ``git ls-files`` answer as it does in a checkout (the first
    fleet verdict, fd5cabfa), the commit is what gives packages/db's migrator
    the HEAD its test admission reads (MCPs board task 6bbfd171), and gc.auto
    0 keeps that commit's loose objects where the build copies them (MCPs
    board task 939ec5c7)."""
    assert stage_repository.init_repository_commands("/s/run", "MCPs-packages-db-1790400000") == (
        ("git", "-C", "/s/run", "init", "--quiet"),
        ("git", "-C", "/s/run", "config", "gc.auto", "0"),
        ("git", "-C", "/s/run", "add", "--all", "--force"),
        (
            "git",
            "-C",
            "/s/run",
            "-c",
            f"user.name={stage_repository.EXPORT_AUTHOR_NAME}",
            "-c",
            f"user.email={stage_repository.EXPORT_AUTHOR_EMAIL}",
            "commit",
            "--quiet",
            "--message",
            "fleet export MCPs-packages-db-1790400000",
        ),
    )


def test_a_companion_is_cloned_from_its_bundle_onto_its_branch() -> None:
    """MCPs board task 2026dfbc: the bundle's ref is fetched into
    ``origin/<branch>``, the branch is checked out there, and HEAD is proved
    to be the commit the hub bundled; the fetch's own auto gc is off first
    (MCPs board task 939ec5c7)."""
    commands = stage_repository.companion_repository_commands(
        "/s/MCPs", "/s/MCPs.stage/tree.tgz", "a" * 40, "refs/heads/main"
    )

    assert commands == (
        ("git", "-C", "/s/MCPs", "init", "--quiet"),
        ("git", "-C", "/s/MCPs", "config", "gc.auto", "0"),
        (
            "git",
            "-C",
            "/s/MCPs",
            "fetch",
            "--quiet",
            "--no-tags",
            "/s/MCPs.stage/tree.tgz",
            "+refs/fleet/companion:refs/remotes/origin/main",
        ),
        ("git", "-C", "/s/MCPs", "checkout", "--quiet", "-B", "main", "refs/remotes/origin/main"),
        ("git", "-C", "/s/MCPs", "merge-base", "--is-ancestor", "HEAD", "a" * 40),
        ("git", "-C", "/s/MCPs", "merge-base", "--is-ancestor", "a" * 40, "HEAD"),
    )


def _git(*args: str) -> str:
    """Run git under a throwaway identity and answer its trimmed output.

    Args:
        *args: The arguments after ``git``.

    Returns:
        Its standard output, stripped.
    """
    ran = subprocess.run(
        ("git", "-c", "user.name=t", "-c", "user.email=t@t.invalid", *args),
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return ran.stdout.strip()


def _bundled_companion(tmp_path: pathlib.Path) -> tuple[pathlib.Path, str, str]:
    """A two-commit repository bundled at its companion ref, as the hub does.

    Args:
        tmp_path: Where the repository and its bundle are made.

    Returns:
        The bundle, the first commit and the tip.
    """
    source = tmp_path / "source"
    (source / "packages" / "maketools").mkdir(parents=True)
    _git("init", "--quiet", str(source))
    (source / "packages" / "maketools" / "run.py").write_bytes(b"print(1)\n")
    _git("-C", str(source), "add", "--all")
    _git("-C", str(source), "commit", "--quiet", "--message", "first")
    first = _git("-C", str(source), "rev-parse", "HEAD")
    (source / "packages" / "maketools" / "run.py").write_bytes(b"print(2)\n")
    _git("-C", str(source), "commit", "--quiet", "--all", "--message", "tip")
    tip = _git("-C", str(source), "rev-parse", "HEAD")
    _git("-C", str(source), "update-ref", "refs/fleet/companion", tip)
    bundle = tmp_path / "MCPs.stage" / names.ARCHIVE_NAME
    bundle.parent.mkdir()
    _git("-C", str(source), "bundle", "create", "--quiet", str(bundle), "refs/fleet/companion")
    return bundle, first, tip


@pytest.mark.parametrize("ref", ["main", "refs/heads/main"])
def test_a_staged_companion_is_its_ref_with_history_and_serves_origin_main(
    tmp_path: pathlib.Path, ref: str
) -> None:
    """The commands run here for real against a bundle shaped like the hub's.
    HEAD and ``origin/main`` are the bundled tip and its ancestor is
    readable, which is what corvis-stick's HookCommands suite needed of MCPs
    on serendipity (MCPs board task 2026dfbc). And MCPs board task a8ee9b21
    still holds: published maketools reads ``git --git-dir=../MCPs/.git
    archive origin/main``, and that call reads the staged file back."""
    bundle, first, tip = _bundled_companion(tmp_path)
    target = tmp_path / "MCPs"
    target.mkdir()
    for command in stage_repository.companion_repository_commands(
        target.as_posix(), bundle.as_posix(), tip, ref
    ):
        subprocess.run(command, check=True, capture_output=True, timeout=60)

    assert _git("-C", str(target), "rev-parse", "HEAD") == tip
    assert _git("-C", str(target), "rev-parse", "refs/remotes/origin/main") == tip
    assert _git("-C", str(target), "show", f"{first}:packages/maketools/run.py") == "print(1)"
    assert _git("-C", str(target), "config", "--local", "gc.auto") == "0"
    archived = subprocess.run(
        [
            "git",
            f"--git-dir={target / '.git'}",
            "archive",
            "--format=zip",
            "origin/main",
            "packages/maketools",
        ],
        check=True,
        capture_output=True,
        timeout=60,
    )

    # The node's own core.autocrlf decides the archive's line endings (CRLF
    # on a Windows node), which is the checkout's business, not the ref's.
    with zipfile.ZipFile(io.BytesIO(archived.stdout)) as archive:
        content = archive.read("packages/maketools/run.py")
    assert content.replace(b"\r\n", b"\n") == b"print(2)\n"


def test_a_bundle_whose_ref_is_not_the_bundled_commit_stops_the_stage(
    tmp_path: pathlib.Path,
) -> None:
    """The last two commands prove HEAD is the commit the hub resolved; asked
    for the bundle's ancestor instead, the first of them exits non-zero, so
    the checked script ends there and the run is not staged on a tree the
    feed would misname."""
    bundle, first, _tip = _bundled_companion(tmp_path)
    target = tmp_path / "MCPs"
    target.mkdir()
    *clone, head_in_sha, _sha_in_head = stage_repository.companion_repository_commands(
        target.as_posix(), bundle.as_posix(), first, "main"
    )
    for command in clone:
        subprocess.run(command, check=True, capture_output=True, timeout=60)

    assert subprocess.run(head_in_sha, capture_output=True, timeout=60).returncode == 1


def test_a_staged_export_tracks_every_file_its_commit_tracks(tmp_path: pathlib.Path) -> None:
    """The commands run here for real on the archive of a commit that tracks
    a file its own ``.gitignore`` ignores, which is what ``git add -f`` makes
    and what ``tools/hpc3``'s nine sweep documents are. Indexed without
    ``--force`` the staged HEAD lost them, and hpc3's audit of ``git archive
    HEAD`` failed on lavender-wsl at f75c4a94 (API board task 0b3591d7)."""
    source = tmp_path / "source"
    (source / "runs").mkdir(parents=True)
    _git("init", "--quiet", str(source))
    (source / ".gitignore").write_bytes(b"runs/*\n")
    (source / "runs" / "sweep.json").write_bytes(b"{}\n")
    (source / "kept.py").write_bytes(b"print(1)\n")
    _git("-C", str(source), "add", "--all")
    _git("-C", str(source), "add", "--force", "runs/sweep.json")
    _git("-C", str(source), "commit", "--quiet", "--message", "source")
    # Zip rather than the tar.gz the hub stages: what is under test is the
    # index the init commands build, not the extraction, and tarfile's
    # extractall(filter=) exists only from Python 3.11.4, which serendipity's
    # interpreter predates (FLEET-CHECK 691db3e6).
    archived = subprocess.run(
        ["git", "-C", str(source), "archive", "--format=zip", "HEAD"],
        check=True,
        capture_output=True,
        timeout=60,
    )
    target = tmp_path / "run"
    with zipfile.ZipFile(io.BytesIO(archived.stdout)) as archive:
        archive.extractall(target)

    for command in stage_repository.init_repository_commands(target.as_posix(), "tools-hpc3-1"):
        subprocess.run(command, check=True, capture_output=True, timeout=60)

    tracked = _git("-C", str(target), "ls-files")
    assert tracked.splitlines() == _git("-C", str(source), "ls-files").splitlines()
    assert "runs/sweep.json" in tracked.splitlines()


def _sampled_blobs(count: int) -> list[tuple[bytes, str]]:
    """File contents whose blobs land in :data:`SAMPLED_DIRECTORY`.

    Args:
        count: How many to find.

    Returns:
        Each content with its blob's object id.
    """
    found: list[tuple[bytes, str]] = []
    serial = 0
    while len(found) < count:
        content = f"{serial}\n".encode()
        blob = hashlib.sha1(b"blob %d\x00" % len(content) + content, usedforsecurity=False)
        if blob.hexdigest().startswith(SAMPLED_DIRECTORY):
            found.append((content, blob.hexdigest()))
        serial += 1
    return found


class TestNoAutoGc:
    """MCPs board task 939ec5c7: the API export's one commit crossed
    gc.auto, the detached gc packed and pruned its loose objects while the
    Linux execution build copied the stage, and the copy died at ``cp:
    cannot stat .../.git/objects/60/...`` on diphtheria (job 56e1aa54). Run
    for real in a repository made to gc eagerly on a three-file tree."""

    def _eager_tree(self, tmp_path: pathlib.Path) -> tuple[pathlib.Path, list[str]]:
        """A repository of three unindexed files, set to gc at once and in the foreground.

        The files' blobs all land in :data:`SAMPLED_DIRECTORY`. ``gc.auto``
        1 makes two loose objects there enough, and ``gc.autoDetach`` false
        makes the commit wait for the pack, so what took three seconds in
        the background on diphtheria is over when the command returns. The
        stage's own ``git init`` that runs next keeps both values, as it
        would keep a node's defaults, and its ``gc.auto 0`` overrides the
        first, as it overrides a node's 6700.

        Args:
            tmp_path: Where the tree is made.

        Returns:
            The tree and its blobs' object ids.
        """
        tree = tmp_path / "run"
        tree.mkdir()
        blobs = _sampled_blobs(3)
        for index, (content, _object_id) in enumerate(blobs):
            (tree / f"file{index}.txt").write_bytes(content)
        _git("init", "--quiet", str(tree))
        _git("-C", str(tree), "config", "gc.auto", "1")
        _git("-C", str(tree), "config", "gc.autoDetach", "false")
        return tree, [object_id for _content, object_id in blobs]

    def test_without_the_setting_the_commit_packs_and_prunes_the_loose_objects(
        self, tmp_path: pathlib.Path
    ) -> None:
        """The control: the same commands less the setting are what raced
        the copy, so the repository above does make gc run here."""
        tree, _object_ids = self._eager_tree(tmp_path)
        initialise, no_auto_gc, *index_and_commit = stage_repository.init_repository_commands(
            tree.as_posix(), "run-1"
        )
        assert no_auto_gc == ("git", "-C", tree.as_posix(), "config", "gc.auto", "0")
        for command in (initialise, *index_and_commit):
            subprocess.run(command, check=True, capture_output=True, timeout=60)

        objects = tree / ".git" / "objects"
        assert not (objects / SAMPLED_DIRECTORY).exists()
        assert list((objects / "pack").glob("*.pack")) != []

    def test_with_it_every_loose_object_stays_where_the_build_copies_it(
        self, tmp_path: pathlib.Path
    ) -> None:
        tree, object_ids = self._eager_tree(tmp_path)
        for command in stage_repository.init_repository_commands(tree.as_posix(), "run-1"):
            subprocess.run(command, check=True, capture_output=True, timeout=60)

        objects = tree / ".git" / "objects"
        for object_id in object_ids:
            assert (objects / object_id[:2] / object_id[2:]).is_file()
        assert list((objects / "pack").glob("*.pack")) == []
        assert _git("-C", str(tree), "rev-list", "--count", "HEAD") == "1"
