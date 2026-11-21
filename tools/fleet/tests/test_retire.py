"""Retiring a settled dispatch's directory (MCPs board task bfca20e6).

:func:`fleet.core.retire.attempt_retire_on_node` is asked which script it
sends where; the ``sh`` rendering is then executed for real under this host's
bash against a stage root laid out under ``tmp_path``, run twice to show a
retire that already happened is not an error.
"""

from __future__ import annotations

import pathlib
import stat
import subprocess

import pytest
from platform_core.errors import FleetErrorCode

from fleet.contracts.node import NodeConfig
from fleet.core import _test_hooks, dialect, names, retire
from fleet.core.dialect import PROLOGUE
from fleet.core.linux_isolated_build import EXEC_HOME_LINE
from fleet.core.retire_script import (
    SCAN_BLIND,
    RetireAnswer,
    parse_retire_answer,
    retire_script,
)
from tests._retire_answers import RETIRED_BYTES
from tests.conftest import DEMO_RUN_ID, FakeRun, failed, ok, retire_replies

#: The stage root, as the roster declares it.
STAGE_ROOT = "/home/corvis/fleet/stage"


def _node() -> NodeConfig:
    """A node with no rootless daemon.

    Returns:
        Its declaration.
    """
    return NodeConfig(
        host="node-1",
        stage_root=STAGE_ROOT,
        logical_cores=8,
        ram_gb=16.0,
        gpu=None,
        enabled=True,
        test_database=False,
        rust=None,
        cxx=None,
        docker=None,
        stack=None,
        netns=None,
        budget={
            "reserved_cores": 2,
            "reserved_ram_gb": 4.0,
            "worker_ram_gb": 1.1,
            "max_disk_gb": 20.0,
            "checks_at_once": None,
            "pinned_checks_only": False,
        },
        models=None,
    )


def _retire_script(*, isolated: bool) -> str:
    """The retire the node is sent for the demo run.

    Args:
        isolated: Whether the node runs docker projects as execdocker.

    Returns:
        The script's text.
    """
    return retire_script(
        stage_root=STAGE_ROOT,
        run_id=DEMO_RUN_ID,
        retained=names.retained_log_path(STAGE_ROOT, DEMO_RUN_ID),
        scripts=tuple(
            dialect.script_path(STAGE_ROOT, stem) for stem in names.root_script_stems(DEMO_RUN_ID)
        ),
        isolated=isolated,
    )


def test_the_names_are_one_spelling() -> None:
    assert names.retire_stem("libs-demo-1") == "retire-libs-demo-1"
    assert names.retained_log_path("/s", "libs-demo-1") == "/s/logs/libs-demo-1.log"
    assert names.root_script_stems("libs-demo-1") == (
        "mkdir-libs-demo-1",
        "mkdir-libs-demo-1.stage",
        "stop-libs-demo-1",
        "retire-libs-demo-1",
    )


def test_it_sends_the_retire_to_the_stage_root_runs_it_and_names_the_kept_transcript() -> None:
    runner = FakeRun(retire_replies()[:2])
    _test_hooks.run = runner

    outcome = retire.attempt_retire_on_node(_node(), run_id=DEMO_RUN_ID)

    assert outcome == retire.RetireAttempt(
        answer=RetireAnswer(holder=None, freed_bytes=RETIRED_BYTES), failure=None
    )
    kept = names.retained_log_path(STAGE_ROOT, DEMO_RUN_ID)
    assert kept == f"{STAGE_ROOT}/logs/{DEMO_RUN_ID}.log"
    # The script sent and run, and no generations' removal: each caller
    # sends that once its row is closed (MCPs board task 8993c306).
    assert len(runner.calls) == 2
    script_path = dialect.script_path(STAGE_ROOT, f"retire-{DEMO_RUN_ID}")
    assert runner.stdin[0] == _retire_script(isolated=False).encode("utf-8")
    assert f"{STAGE_ROOT}/stop-{DEMO_RUN_ID}.sh" in _retire_script(isolated=False)
    assert script_path in " ".join(runner.calls[0])
    assert runner.calls[1][-1].endswith(script_path)


def test_a_node_running_isolated_builds_retires_the_run_s_execdocker_copy_too() -> None:
    """A docker project's build removes its copy only when it runs to its
    end: measured 2026-10-18, lavender-wsl held eight stopped builds'
    copies, 11 GB (MCPs board task 346abc45)."""
    node = _node()
    node["docker"] = "29.8.1"
    runner = FakeRun(retire_replies()[:2])
    _test_hooks.run = runner

    assert retire.attempt_retire_on_node(node, run_id=DEMO_RUN_ID)["failure"] is None

    assert retire.runs_isolated(node)
    assert not retire.runs_isolated(_node())
    sent = _retire_script(isolated=True)
    assert runner.stdin[0] == sent.encode("utf-8")
    assert sent.endswith(
        f'{EXEC_HOME_LINE}\nsudo -n rm -rf "$exec_home/fleet/stage/{DEMO_RUN_ID}" '
        f'"$exec_home/fleet/stage/{DEMO_RUN_ID}.stage"\n'
        'echo "freed ${freed%%[[:space:]]*}"\n'
    )


@pytest.mark.parametrize(
    ("result", "code"),
    [
        (
            failed(255, "Connection timed out during banner exchange"),
            FleetErrorCode.NODE_UNREACHABLE,
        ),
        (failed(1, "rm: cannot remove: Permission denied"), FleetErrorCode.DISPATCH_FAILED),
    ],
)
def test_a_retire_that_fails_reports_its_transport_code(
    result: _test_hooks.CommandResult, code: FleetErrorCode
) -> None:
    """The retire record's writer tells a node that did not answer from one
    that answered and failed (fleet.core.retire_owed), so both are values."""
    _test_hooks.run = FakeRun([ok(""), result])

    attempt = retire.attempt_retire_on_node(_node(), run_id=DEMO_RUN_ID)

    failure = attempt["failure"]
    codes = [failure["code"]] if failure is not None else []
    assert codes == [code]
    assert attempt["answer"] == retire.UNANSWERED


@pytest.mark.parametrize(
    ("output", "answer"),
    [
        (
            f"in-use /proc/7/cwd {STAGE_ROOT}/r/r\n",
            RetireAnswer(holder=f"/proc/7/cwd {STAGE_ROOT}/r/r", freed_bytes=0),
        ),
        (
            "in-use /proc/7/fd/3 /s/r/log with space\n",
            RetireAnswer(holder="/proc/7/fd/3 /s/r/log with space", freed_bytes=0),
        ),
        ("moved\nfreed 0\n", RetireAnswer(holder=None, freed_bytes=0)),
        ("freed 1234\n", RetireAnswer(holder=None, freed_bytes=1234)),
        ("", None),
        ("freed\n", None),
        ("freed -3\n", None),
        ("freed 12 bytes\n", None),
        ("in-use /proc/7/cwd relative\n", None),
        ("in-use /proc/self/cwd /s\n", None),
        ("freed 1\nmoved\n", None),
    ],
)
def test_the_answer_is_the_last_line_and_nothing_else(
    output: str, answer: RetireAnswer | None
) -> None:
    assert parse_retire_answer(output) == answer


def test_an_answer_missing_is_a_dispatch_failure_naming_the_line() -> None:
    _test_hooks.run = FakeRun([ok(""), ok("half\n")])

    attempt = retire.attempt_retire_on_node(_node(), run_id=DEMO_RUN_ID)

    assert attempt["answer"] == retire.UNANSWERED
    assert attempt["failure"] == {
        "code": FleetErrorCode.DISPATCH_FAILED,
        "message": "node-1's retire ended without its answer: its last line was 'half', "
        "not 'in-use <link> <target>' or 'freed <bytes>'",
    }


def seal(release: pathlib.Path) -> None:
    """A release whose payload directory and checkout carry no write bit.

    Args:
        release: The checkout to make and lock.
    """
    payload = release / "scripts"
    payload.mkdir(parents=True)
    (payload / "release-kinds.json").write_text("{}\n", encoding="utf-8")
    for directory in (payload, release):
        directory.chmod(0o555)


class TestTheShScriptRunsForReal:
    """The ``sh`` retire against a real stage root, under this host's bash."""

    def outside_modes(self, root: pathlib.Path) -> list[int]:
        """The permission bits of the sealed release outside the run's trees.

        It gets its write bits back after they are read, so no case leaves
        a locked directory under ``tmp_path`` for the stage root this suite
        runs in to strand.

        Args:
            root: The stage root.

        Returns:
            The release's and its payload directory's bits, in that order.
        """
        release = root / "cache" / "MCPs"
        locked = (release, release / "scripts")
        modes = [stat.S_IMODE(path.stat().st_mode) for path in locked]
        for path in locked:
            path.chmod(0o755)
        return modes

    def lay_out(self, root: pathlib.Path, *, transcript: bool) -> pathlib.Path:
        """A finished run's directory and its root scripts, as a node leaves them.

        Args:
            root: The stage root.
            transcript: Whether the build wrote its transcript.

        Returns:
            The run's directory.
        """
        target = pathlib.Path(names.dispatch_directory(root.as_posix(), DEMO_RUN_ID))
        objects = target / ".git" / "objects"
        objects.mkdir(parents=True)
        loose = objects / "ab"
        loose.write_text("loose object", encoding="utf-8")
        loose.chmod(0o444)
        (target / names.RESULT_NAME).write_text("0\n", encoding="utf-8")
        if transcript:
            (target / f"{names.RESULT_NAME}.log").write_text("887 passed\n", encoding="utf-8")
        for stem in names.root_script_stems(DEMO_RUN_ID):
            (root / f"{stem}.sh").write_text(PROLOGUE, encoding="utf-8")
        # The staging directory beside the export in the run's enclosure: the
        # archive and the scripts that staged it, none of them inside the tree.
        staged = pathlib.Path(names.staging_directory(target.as_posix()))
        staged.mkdir()
        (staged / names.ARCHIVE_NAME).write_bytes(b"\x1f\x8b")
        (staged / "extract.sh").write_text(PROLOGUE, encoding="utf-8")
        (root / "cache").mkdir()
        (root / "libs-other-1757000001").mkdir()
        # A companion in the stage root, linked into the enclosure as
        # staging.link_companions links it: the run's to read, not to remove.
        companion = root / "MCPs"
        (companion / "wiki").mkdir(parents=True)
        (target.parent / "MCPs").symlink_to(companion, target_is_directory=True)
        # Releases sealed read-only, the way MCPs f27b4822e seals them: one
        # where a maketools run's tests leave their fixtures, one staged, and
        # one outside both trees that a link inside the run names.
        seal(target / "packages" / "maketools" / ".pytest-tmp" / "store" / "MCPs")
        seal(staged / "MCPs")
        seal(root / "cache" / "MCPs")
        (target / "linked").symlink_to(root / "cache" / "MCPs", target_is_directory=True)
        # The two modes loki's stranded run held under its .pytest-tmp (MCPs
        # board task 1526ece1): 0355 can be written and searched but not
        # listed, so neither find nor rm reads its entries, and 0655 can be
        # listed but not searched, so its entries cannot be reached.
        basetemp = target / "packages" / "maketools" / ".pytest-tmp" / "basetemp"
        for name, mode in (("repository", 0o355), ("candidate", 0o655)):
            nested = basetemp / name / "objects"
            nested.mkdir(parents=True)
            (nested / "pack").write_text("pack\n", encoding="utf-8")
            (basetemp / name).chmod(mode)
        return target

    def run_retire(
        self, root: pathlib.Path, *, home: pathlib.Path | None = None
    ) -> subprocess.CompletedProcess[str]:
        """Render the retire for ``root`` and run it by its own path, as a node does.

        Args:
            root: The stage root.
            home: A ``HOME`` for the script, whose ``.local/bin`` the
                prologue puts first on ``PATH``; this process's when None.

        Returns:
            What bash did.
        """
        stage_root = root.as_posix()
        script = pathlib.Path(root / f"{names.retire_stem(DEMO_RUN_ID)}.sh")
        script.write_text(
            retire_script(
                stage_root=stage_root,
                run_id=DEMO_RUN_ID,
                retained=names.retained_log_path(stage_root, DEMO_RUN_ID),
                scripts=tuple(
                    dialect.script_path(stage_root, stem)
                    for stem in names.root_script_stems(DEMO_RUN_ID)
                ),
                isolated=False,
            ),
            encoding="utf-8",
        )
        homed: tuple[str, ...] = () if home is None else ("env", f"HOME={home.as_posix()}")
        return subprocess.run(
            [*homed, "bash", script.as_posix()],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )

    def retire(self, root: pathlib.Path) -> str:
        """Run the retire for ``root``, which must succeed.

        Args:
            root: The stage root.

        Returns:
            Its standard output.
        """
        ran = self.run_retire(root)
        assert ran.returncode == 0, ran.stderr
        return ran.stdout

    def test_a_process_in_the_tree_keeps_it_whole_until_it_ends(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A shell left in the run's tree: nothing is moved or removed while it lives."""
        target = self.lay_out(tmp_path, transcript=True)
        process = subprocess.Popen(["sleep", "300"], cwd=target / "packages")
        answer = parse_retire_answer(self.retire(tmp_path))
        process.kill()
        process.wait(timeout=10)

        assert answer == RetireAnswer(
            holder=f"/proc/{process.pid}/cwd {(target / 'packages').as_posix()}", freed_bytes=0
        )
        assert (target / f"{names.RESULT_NAME}.log").is_file()
        assert (tmp_path / f"{names.retire_stem(DEMO_RUN_ID)}.sh").is_file()
        assert not (tmp_path / "logs").exists()

        freed = parse_retire_answer(self.retire(tmp_path))

        assert freed is not None and freed["holder"] is None and freed["freed_bytes"] > 0
        assert not target.exists()
        assert self.outside_modes(tmp_path) == [0o555, 0o555]

    def test_an_open_file_in_the_tree_keeps_it_whole(self, tmp_path: pathlib.Path) -> None:
        target = self.lay_out(tmp_path, transcript=False)
        held = target / names.RESULT_NAME
        with held.open("rb") as handle:
            process = subprocess.Popen(["sleep", "300"], stdin=handle)
        answer = parse_retire_answer(self.retire(tmp_path))
        process.kill()
        process.wait(timeout=10)

        assert answer == RetireAnswer(
            holder=f"/proc/{process.pid}/fd/0 {held.as_posix()}", freed_bytes=0
        )
        assert held.is_file()
        self.retire(tmp_path)
        assert not target.exists()
        assert self.outside_modes(tmp_path) == [0o555, 0o555]

    def test_a_find_that_cannot_read_proc_links_fails_the_retire_and_touches_nothing(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A ``find`` without ``-printf``, standing first on ``PATH``, reads
        no link: the retire stops by name rather than answering that nothing
        uses the run."""
        target = self.lay_out(tmp_path, transcript=True)
        home = tmp_path / "home"
        shim = home / ".local" / "bin" / "find"
        shim.parent.mkdir(parents=True)
        shim.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        shim.chmod(0o755)

        ran = self.run_retire(tmp_path, home=home)

        assert ran.returncode == 1
        assert ran.stderr.startswith(f"{SCAN_BLIND}: find read /proc/")
        assert ran.stdout == ""
        assert (target / f"{names.RESULT_NAME}.log").is_file()
        assert self.outside_modes(tmp_path) == [0o555, 0o555]
        self.retire(tmp_path)

    def test_it_keeps_the_transcript_and_removes_the_tree_and_the_run_s_root_scripts(
        self, tmp_path: pathlib.Path
    ) -> None:
        self.lay_out(tmp_path, transcript=True)

        self.retire(tmp_path)

        assert (tmp_path / "logs" / f"{DEMO_RUN_ID}.log").read_text(encoding="utf-8") == (
            "887 passed\n"
        )
        # The cache, the companion and another run's directory are not this
        # run's to remove: its enclosure goes whole, its links with it, and
        # what the links named is left whole and still locked.
        assert sorted(entry.name for entry in tmp_path.iterdir()) == [
            "MCPs",
            "cache",
            "libs-other-1757000001",
            "logs",
        ]
        assert self.outside_modes(tmp_path) == [0o555, 0o555]
        assert (tmp_path / "cache" / "MCPs" / "scripts" / "release-kinds.json").is_file()
        assert (tmp_path / "MCPs" / "wiki").is_dir()

    def test_the_check_s_results_are_kept_beside_the_transcript_once_however_often_retired(
        self, tmp_path: pathlib.Path
    ) -> None:
        """MCPs reads a run's cases from its JUnit XML (board task 25ab85cf),
        so the copy the build made beside the transcript outlives the tree."""
        target = self.lay_out(tmp_path, transcript=True)
        results = target / "check-results"
        results.mkdir()
        (results / "pkg.vitest.junit.xml").write_text("<testsuites/>\n", encoding="utf-8")

        self.retire(tmp_path)
        self.retire(tmp_path)

        kept = tmp_path / "logs" / f"{DEMO_RUN_ID}.results"
        assert [entry.name for entry in kept.iterdir()] == ["pkg.vitest.junit.xml"]
        assert (kept / "pkg.vitest.junit.xml").read_text(encoding="utf-8") == "<testsuites/>\n"
        assert sorted(entry.name for entry in (tmp_path / "logs").iterdir()) == [
            f"{DEMO_RUN_ID}.log",
            f"{DEMO_RUN_ID}.results",
        ]
        assert self.outside_modes(tmp_path) == [0o555, 0o555]

    def test_a_run_that_wrote_no_transcript_is_retired_all_the_same_and_twice(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A run cancelled before its build began wrote no transcript, and a
        retire that failed part way is run again by the next tick."""
        self.lay_out(tmp_path, transcript=False)

        self.retire(tmp_path)
        self.retire(tmp_path)

        assert list((tmp_path / "logs").iterdir()) == []
        assert not (tmp_path / DEMO_RUN_ID).exists()
        assert self.outside_modes(tmp_path) == [0o555, 0o555]
