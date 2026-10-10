"""The align: every refusal by name, every dropped commit read path by path, the move."""

from __future__ import annotations

from collections.abc import Generator, Sequence
from pathlib import Path

import pytest
from platform_core.error_codes_tooling import MaketoolsErrorCode
from platform_core.errors import AppError

from maketools.align_checkout import (
    AlignReport,
    LocalCommit,
    PathReading,
    align_checkout,
    read_local_commits,
    read_path,
    render_report,
)
from maketools.commands import CommandResult
from tests.conftest import World, failed, ok, restore_defaults

TARGET = "b" * 40
HEAD = "a" * 40
LOCAL_ONE = "1" * 40
LOCAL_TWO = "2" * 40
ROOT = Path("/repo")
RESET = ("git", "reset", "--hard", TARGET)
READ_HEAD = ("git", "rev-parse", "HEAD")


class MovingWorld(World):
    """A world whose HEAD answers in sequence, since an align asks before and after the move.

    Attributes:
        sequences: Per argv, the answers given in order; once a sequence is
            spent the scripted single answer is given, as for any argv.
    """

    def __init__(self) -> None:
        """Start with no sequences."""
        super().__init__()
        self.sequences: dict[tuple[str, ...], list[CommandResult]] = {}

    def run_capturing(self, argv: Sequence[str], *, cwd: Path) -> CommandResult:
        key = tuple(argv)
        queued = self.sequences.get(key)
        if queued:
            self.captured.append(key)
            return queued.pop(0)
        return super().run_capturing(argv, cwd=cwd)


@pytest.fixture()
def world() -> Generator[MovingWorld, None, None]:
    """A moving world bound for the test's duration."""
    fake = MovingWorld()
    fake.bind()
    yield fake
    restore_defaults()


def blob_argv(revision: str, path: str) -> tuple[str, ...]:
    """The argv that reads a path's blob at a revision."""
    return ("git", "rev-parse", "--verify", "--quiet", f"{revision}:{path}")


def hash_argv(path: str) -> tuple[str, ...]:
    """The argv that hashes a working-tree file."""
    return ("git", "hash-object", "--", path)


def script_until_head(
    world: MovingWorld, *, head: str = HEAD, status: str = "", target_paths: str = ""
) -> None:
    """Script the questions every align asks before it reads local commits."""
    answers: dict[tuple[str, ...], CommandResult] = {
        ("git", "fetch", "origin"): ok(""),
        ("git", "rev-parse", "--verify", f"{TARGET}^{{commit}}"): ok(TARGET),
        ("git", "merge-base", "--is-ancestor", TARGET, "origin/main"): ok(""),
        ("git", "status", "--porcelain"): ok(status),
        ("git", "ls-tree", "-r", "--name-only", TARGET): ok(target_paths),
        READ_HEAD: ok(head),
    }
    world.capturing_answers.update(answers)


def script_blobs(world: MovingWorld, blobs: dict[tuple[str, str], str | None]) -> None:
    """Script what each (revision, path) reads as; None for an absent path."""
    for (revision, path), blob in blobs.items():
        world.capturing_answers[blob_argv(revision, path)] = (
            ok(blob) if blob is not None else failed(1, "")
        )


def script_local(
    world: MovingWorld,
    cherry: str,
    *,
    subjects: dict[str, str],
    touched: dict[str, str],
    blobs: dict[tuple[str, str], str | None],
) -> None:
    """Script the commits ``git cherry`` reports and what each path reads as."""
    world.capturing_answers[("git", "cherry", TARGET, HEAD)] = ok(cherry)
    for sha, subject in subjects.items():
        world.capturing_answers[("git", "log", "-1", "--format=%s", sha)] = ok(subject)
    for sha, names in touched.items():
        argv = ("git", "diff-tree", "--no-commit-id", "--name-only", "-r", sha)
        world.capturing_answers[argv] = ok(names)
    script_blobs(world, blobs)


def script_move(world: MovingWorld, *, reads_back: str = TARGET) -> None:
    """Script the move and what HEAD reads afterwards."""
    world.capturing_answers[RESET] = ok("HEAD is now at b")
    world.sequences[READ_HEAD] = [ok(HEAD), ok(reads_back)]


def test_a_failed_fetch_is_refused_with_its_stderr(world: MovingWorld) -> None:
    world.capturing_answers[("git", "fetch", "origin")] = failed(128, "could not resolve host")
    with pytest.raises(AppError, match=r"git fetch origin failed \(exit 128\): could not resolve"):
        align_checkout(ROOT, TARGET, [])


def test_a_commit_origin_main_does_not_contain_is_refused(world: MovingWorld) -> None:
    script_until_head(world)
    ancestry = ("git", "merge-base", "--is-ancestor", TARGET, "origin/main")
    world.capturing_answers[ancestry] = failed(1, "")
    with pytest.raises(AppError, match=rf"{TARGET[:12]} is not on origin/main.*push it first"):
        align_checkout(ROOT, TARGET, [])
    assert ("git", "status", "--porcelain") not in world.captured


def test_tracked_dirt_is_refused_by_path(world: MovingWorld) -> None:
    script_until_head(world, status=" M a/b.py\n D c.py\n?? notes/")
    with pytest.raises(AppError, match=r"2 uncommitted tracked path\(s\).*\na/b.py\nc.py") as got:
        align_checkout(ROOT, TARGET, [])
    assert "notes/" not in got.value.message


def test_an_untracked_file_differing_from_the_target_is_refused(world: MovingWorld) -> None:
    status = "?? libs/x/new.py\n?? runs/\n?? loose.txt\n?? a.py"
    target_paths = "libs/x/new.py\nruns/same.json\nruns/other.json\nruns/missing.json\na.py.bak"
    script_until_head(world, status=status, target_paths=target_paths)
    script_blobs(
        world,
        {
            (TARGET, "libs/x/new.py"): "theirs",
            (TARGET, "runs/same.json"): "same",
            (TARGET, "runs/other.json"): "theirs",
        },
    )
    world.capturing_answers[hash_argv("libs/x/new.py")] = ok("mine")
    world.capturing_answers[hash_argv("runs/same.json")] = ok("same")
    world.capturing_answers[hash_argv("runs/other.json")] = ok("mine")
    world.capturing_answers[hash_argv("runs/missing.json")] = failed(128, "no such file")
    expected = r"2 untracked file\(s\) differ from what .* tracks.*\nlibs/x/new.py\nruns/other.json"
    with pytest.raises(AppError, match=expected):
        align_checkout(ROOT, TARGET, [])
    assert hash_argv("a.py.bak") not in world.captured


def test_already_at_the_target_moves_nothing_and_carries_untracked_paths(
    world: MovingWorld,
) -> None:
    script_until_head(world, head=TARGET, status="?? loose.txt", target_paths="a.py")
    report = align_checkout(ROOT, TARGET, [])
    assert report == AlignReport(
        before=TARGET, after=TARGET, local=[], carried_untracked=["loose.txt"]
    )
    assert RESET not in world.captured
    assert render_report(report) == [
        f"align: already at {TARGET[:12]}; nothing moved",
        "  carried untracked loose.txt",
    ]


def test_a_fast_forward_drops_nothing_and_moves(world: MovingWorld) -> None:
    script_until_head(world)
    script_local(world, "", subjects={}, touched={}, blobs={})
    script_move(world)
    report = align_checkout(ROOT, TARGET, [])
    assert report["before"] == HEAD
    assert report["after"] == TARGET
    assert report["local"] == []
    assert world.captured[-2:] == [RESET, READ_HEAD]
    assert render_report(report) == [
        f"align: {HEAD[:12]} -> {TARGET[:12]}, 0 local commit(s) dropped"
    ]


def test_a_checkout_that_reads_back_elsewhere_is_refused(world: MovingWorld) -> None:
    script_until_head(world)
    script_local(world, "", subjects={}, touched={}, blobs={})
    script_move(world, reads_back="c" * 40)
    with pytest.raises(AppError, match=rf"reads back {'c' * 12}, not {TARGET[:12]}"):
        align_checkout(ROOT, TARGET, [])


def test_patch_equivalent_and_carried_commits_are_dropped_with_their_reading(
    world: MovingWorld,
) -> None:
    script_until_head(world)
    script_local(
        world,
        f"- {LOCAL_ONE}\n+ {LOCAL_TWO}",
        subjects={LOCAL_ONE: "re-commit of origin's row", LOCAL_TWO: "blobs origin carries"},
        touched={LOCAL_TWO: "x.py\ngone.py"},
        blobs={
            (TARGET, "x.py"): "blob-x",
            (LOCAL_TWO, "x.py"): "blob-x",
            (TARGET, "gone.py"): None,
            (LOCAL_TWO, "gone.py"): None,
        },
    )
    script_move(world)
    report = align_checkout(ROOT, TARGET, [])
    assert report["local"] == [
        LocalCommit(
            sha=LOCAL_ONE, subject="re-commit of origin's row", patch_equivalent=True, paths=[]
        ),
        LocalCommit(
            sha=LOCAL_TWO,
            subject="blobs origin carries",
            patch_equivalent=False,
            paths=[
                PathReading(path="x.py", fate="carried"),
                PathReading(path="gone.py", fate="carried"),
            ],
        ),
    ]
    assert render_report(report)[1:] == [
        f'  dropped {LOCAL_ONE[:12]} "re-commit of origin\'s row": '
        "patch-equivalent to a commit on the target",
        f"  dropped {LOCAL_TWO[:12]} 'blobs origin carries': x.py carried, gone.py carried",
    ]


def test_read_path_tells_carried_absent_and_evolved_apart(world: MovingWorld) -> None:
    script_blobs(
        world,
        {
            (TARGET, "same.py"): "new",
            (LOCAL_ONE, "same.py"): "new",
            (TARGET, "lost.py"): "old",
            (LOCAL_ONE, "lost.py"): "new",
            (f"{LOCAL_ONE}^", "lost.py"): "old",
            (TARGET, "moved.py"): "newer",
            (LOCAL_ONE, "moved.py"): "new",
            (f"{LOCAL_ONE}^", "moved.py"): "old",
        },
    )
    same = read_path(ROOT, LOCAL_ONE, TARGET, "same.py")
    lost = read_path(ROOT, LOCAL_ONE, TARGET, "lost.py")
    moved = read_path(ROOT, LOCAL_ONE, TARGET, "moved.py")
    assert same == PathReading(path="same.py", fate="carried")
    assert lost == PathReading(path="lost.py", fate="absent")
    assert moved == PathReading(path="moved.py", fate="evolved")


def test_an_uncarried_commit_is_refused_naming_each_path_and_the_drop_form(
    world: MovingWorld,
) -> None:
    script_until_head(world)
    script_local(
        world,
        f"+ {LOCAL_ONE}\n+ {LOCAL_TWO}",
        subjects={LOCAL_ONE: "a change origin lacks", LOCAL_TWO: "a change that moved on"},
        touched={LOCAL_ONE: "lost.py", LOCAL_TWO: "moved.py"},
        blobs={
            (TARGET, "lost.py"): "old",
            (LOCAL_ONE, "lost.py"): "new",
            (f"{LOCAL_ONE}^", "lost.py"): "old",
            (TARGET, "moved.py"): "newer",
            (LOCAL_TWO, "moved.py"): "new",
            (f"{LOCAL_TWO}^", "moved.py"): "old",
        },
    )
    with pytest.raises(AppError) as refusal:
        align_checkout(ROOT, TARGET, [])
    assert refusal.value.code is MaketoolsErrorCode.ALIGN
    assert refusal.value.message.endswith(
        f"{LOCAL_ONE[:12]} 'a change origin lacks': lost.py absent\n"
        f"{LOCAL_TWO[:12]} 'a change that moved on': moved.py evolved"
    )
    assert "name it in DROP=<sha>,<sha>" in refusal.value.message
    assert RESET not in world.captured


def test_drop_names_the_commits_the_caller_read_and_the_move_proceeds(
    world: MovingWorld,
) -> None:
    script_until_head(world, status="?? loose.txt")
    script_local(
        world,
        f"+ {LOCAL_ONE}",
        subjects={LOCAL_ONE: "a change that moved on"},
        touched={LOCAL_ONE: "moved.py"},
        blobs={
            (TARGET, "moved.py"): "newer",
            (LOCAL_ONE, "moved.py"): "new",
            (f"{LOCAL_ONE}^", "moved.py"): "old",
        },
    )
    script_move(world)
    report = align_checkout(ROOT, TARGET, ["", LOCAL_ONE[:8]])
    assert report["local"][0]["paths"] == [PathReading(path="moved.py", fate="evolved")]
    assert report["carried_untracked"] == ["loose.txt"]
    assert render_report(report) == [
        f"align: {HEAD[:12]} -> {TARGET[:12]}, 1 local commit(s) dropped",
        f"  dropped {LOCAL_ONE[:12]} 'a change that moved on': moved.py evolved",
        "  carried untracked loose.txt",
    ]


def test_read_local_commits_asks_cherry_then_each_commit(world: MovingWorld) -> None:
    script_local(world, f"- {LOCAL_ONE}", subjects={LOCAL_ONE: "equivalent"}, touched={}, blobs={})
    assert read_local_commits(ROOT, HEAD, TARGET) == [
        LocalCommit(sha=LOCAL_ONE, subject="equivalent", patch_equivalent=True, paths=[])
    ]
    assert world.captured == [
        ("git", "cherry", TARGET, HEAD),
        ("git", "log", "-1", "--format=%s", LOCAL_ONE),
    ]
