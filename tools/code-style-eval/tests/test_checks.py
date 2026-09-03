"""Running the repository's checkers over one generated file.

The composition of each checker's argv is asserted separately from the
running of it: the argv is where the guards' scoping lives, and scoping them
wrongly is silent -- every item would simply share one verdict.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.config import config_test_hooks

from code_style_eval.core import _test_hooks as core_hooks
from code_style_eval.core.checks import (
    checker_command,
    checker_environment,
    package_source_roots,
    run_check,
    score_item,
)
from tests.conftest import _Finished, _Recorder


class TestComposingCheckerCommands:
    """Every checker is invoked through a named interpreter."""

    def test_ruff_checks_the_file(self) -> None:
        """A bare name would resolve against PATH to another version."""
        command = checker_command("ruff", "py", pathlib.Path("a.py"), pathlib.Path("root"))

        assert command == ("py", "-m", "ruff", "check", "a.py")

    def test_mypy_checks_the_file(self) -> None:
        """Same reasoning, same shape."""
        command = checker_command("mypy", "py", pathlib.Path("a.py"), pathlib.Path("root"))

        assert command == ("py", "-m", "mypy", "a.py")

    def test_the_guards_are_scoped_to_the_items_own_root(self) -> None:
        """Without --root they would score the whole sweep identically.

        The guards are scoped to a tree. An unscoped run reads whatever
        package the process is in, which is one verdict shared by every item,
        so the guards column would carry no per-item information at all.
        """
        command = checker_command("guards", "py", pathlib.Path("a.py"), pathlib.Path("root"))

        assert command == ("py", "-m", "scripts.guard", "--root", "root")

    def test_the_guards_root_is_not_the_file(self) -> None:
        """The file is what ruff and mypy read; the guards read the tree."""
        command = checker_command(
            "guards", "py", pathlib.Path("root/src/a.py"), pathlib.Path("root")
        )

        assert str(pathlib.Path("root/src/a.py")) not in command

    def test_an_unknown_checker_is_refused(self) -> None:
        """The set is closed at the composition point too."""
        with pytest.raises(ValueError, match="unknown checker"):
            _ = checker_command("pylint", "py", pathlib.Path("a.py"), pathlib.Path("root"))


class TestResolvingImportsForASandboxedFile:
    """A file alone in a throwaway tree can still find its own package.

    The per-item isolation the guards need is exactly what starves mypy: in
    the real repository a generated file would sit inside its package and
    resolve its imports against that package's ``src``.
    """

    def _package(self, repo: pathlib.Path, category: str, name: str) -> pathlib.Path:
        """Create one package with a source root.

        Args:
            repo: Repository root.
            category: Category directory, e.g. ``libs``.
            name: Package directory name.

        Returns:
            The package's ``src`` directory.
        """
        source = repo / category / name / "src"
        source.mkdir(parents=True)
        return source

    def test_every_package_source_root_is_found(self, tmp_path: pathlib.Path) -> None:
        """Missing one means the imports it holds report as missing stubs."""
        (tmp_path / "libs").mkdir()
        first = self._package(tmp_path, "libs", "platform_core")
        second = self._package(tmp_path, "clients", "DiscordBot")
        caller = tmp_path / "tools" / "code-style-eval"
        caller.mkdir(parents=True)

        assert package_source_roots(caller) == tuple(sorted((second, first)))

    def test_a_directory_without_a_source_root_is_not_offered(self, tmp_path: pathlib.Path) -> None:
        """A path that does not exist on disk would only pad MYPYPATH."""
        (tmp_path / "libs").mkdir()
        (tmp_path / "libs" / "no_src").mkdir()
        caller = tmp_path / "tools" / "code-style-eval"
        caller.mkdir(parents=True)

        assert package_source_roots(caller) == ()

    def test_a_package_outside_the_layout_is_refused(self, tmp_path: pathlib.Path) -> None:
        """Silently returning nothing would look like a typing verdict.

        With an empty MYPYPATH every import reports a missing stub, and the
        sweep would record that as the generated code failing to type-check
        rather than as the instrument being pointed at the wrong tree.
        """
        caller = tmp_path / "tools" / "code-style-eval"
        caller.mkdir(parents=True)

        with pytest.raises(RuntimeError, match="expected a libs directory"):
            _ = package_source_roots(caller)

    def test_the_environment_carries_the_roots_and_keeps_the_rest(
        self, tmp_path: pathlib.Path
    ) -> None:
        """MYPYPATH is added; the inherited environment is not discarded."""
        (tmp_path / "libs").mkdir()
        source = self._package(tmp_path, "libs", "platform_core")
        caller = tmp_path / "tools" / "code-style-eval"
        caller.mkdir(parents=True)

        environment = checker_environment(caller, {"PATH": "/usr/bin"})

        assert environment["MYPYPATH"] == str(source)
        assert environment["PATH"] == "/usr/bin"


class TestRunningOneCheck:
    """A checker's exit status is the verdict."""

    def test_a_zero_exit_passes_and_carries_no_detail(self) -> None:
        """A clean run has nothing to say."""
        core_hooks.Hooks.run_checker = _Recorder({"ruff": _Finished(0, "all good")})

        outcome = run_check(
            "ruff", "py", pathlib.Path("a.py"), pathlib.Path("root"), pathlib.Path("."), {}
        )

        assert outcome["passed"] is True
        assert outcome["exit_code"] == 0
        assert outcome["detail"] == ""

    def test_a_failure_carries_the_first_output_line(self) -> None:
        """An index into the logs, not a replacement for them."""
        core_hooks.Hooks.run_checker = _Recorder(
            {"ruff": _Finished(1, "\n  a.py:1:1 E501 line too long\nmore\n")}
        )

        outcome = run_check(
            "ruff", "py", pathlib.Path("a.py"), pathlib.Path("root"), pathlib.Path("."), {}
        )

        assert outcome["passed"] is False
        assert outcome["exit_code"] == 1
        assert outcome["detail"] == "a.py:1:1 E501 line too long"

    def test_stderr_is_used_when_stdout_is_empty(self) -> None:
        """mypy crashes report on stderr, and a crash is not a pass."""
        core_hooks.Hooks.run_checker = _Recorder({"mypy": _Finished(2, "", "internal error")})

        outcome = run_check(
            "mypy", "py", pathlib.Path("a.py"), pathlib.Path("root"), pathlib.Path("."), {}
        )

        assert outcome["exit_code"] == 2
        assert outcome["detail"] == "internal error"

    def test_stderr_wins_over_stdout_when_both_speak(self) -> None:
        """The guards banner must not shadow the violation it precedes.

        ``scripts.guard`` always writes a rule-count summary to stdout and
        writes the violations themselves to stderr. Reading stdout first gave
        every guard failure in a sweep the identical detail "Guard rule
        summary:", so the field that is supposed to index the run's logs
        pointed at nothing for the one checker this instrument exists to
        report.
        """
        core_hooks.Hooks.run_checker = _Recorder(
            {
                "scripts.guard": _Finished(
                    2,
                    "Guard rule summary:\n  typing: 1 violations\n",
                    "Guard checks failed:\n  a.py:8: kind=any-usage text=\n",
                )
            }
        )

        outcome = run_check(
            "guards", "py", pathlib.Path("a.py"), pathlib.Path("root"), pathlib.Path("."), {}
        )

        assert outcome["detail"] == "a.py:8: kind=any-usage text="

    def test_a_detail_of_nothing_but_a_header_is_still_reported(self) -> None:
        """Skipping introducers must not turn a failure into silence."""
        core_hooks.Hooks.run_checker = _Recorder(
            {"scripts.guard": _Finished(2, "", "Guard checks failed:\n")}
        )

        outcome = run_check(
            "guards", "py", pathlib.Path("a.py"), pathlib.Path("root"), pathlib.Path("."), {}
        )

        assert outcome["detail"] == "Guard checks failed:"

    def test_a_failure_with_no_output_at_all_has_no_detail(self) -> None:
        """There is nothing to index, and inventing a line would mislead."""
        core_hooks.Hooks.run_checker = _Recorder({"scripts.guard": _Finished(2, "", "")})

        outcome = run_check(
            "guards", "py", pathlib.Path("a.py"), pathlib.Path("root"), pathlib.Path("."), {}
        )

        assert outcome["passed"] is False
        assert outcome["detail"] == ""

    def test_a_failure_with_no_output_still_records_the_code(self) -> None:
        """Silence plus non-zero is still a failure."""
        core_hooks.Hooks.run_checker = _Recorder({"ruff": _Finished(3, "", "")})

        outcome = run_check(
            "ruff", "py", pathlib.Path("a.py"), pathlib.Path("root"), pathlib.Path("."), {}
        )

        assert outcome["passed"] is False
        assert outcome["detail"] == ""


class TestScoringOneItem:
    """All three checkers run, regardless of earlier failures."""

    def test_every_checker_runs_even_after_one_fails(self) -> None:
        """Stopping early would make the rates depend on order."""
        recorder = _Recorder({"ruff": _Finished(1, "bad")})
        core_hooks.Hooks.run_checker = recorder

        outcome = score_item(
            item_id="a.py",
            arm="base",
            interpreter="py",
            target=pathlib.Path("a.py"),
            root=pathlib.Path("root"),
            cwd=pathlib.Path("."),
            env={},
        )

        assert len(recorder.calls) == 3
        assert [c["checker"] for c in outcome["checks"]] == ["ruff", "mypy", "guards"]
        assert outcome["all_passed"] is False

    def test_all_passing_sets_the_summary(self) -> None:
        """The summary is derived, never asserted independently."""
        core_hooks.Hooks.run_checker = _Recorder({})

        outcome = score_item(
            item_id="a.py",
            arm="base",
            interpreter="py",
            target=pathlib.Path("a.py"),
            root=pathlib.Path("root"),
            cwd=pathlib.Path("."),
            env={},
        )

        assert outcome["all_passed"] is True


class TestTheProductionRunner:
    """The default hook actually starts a process."""

    def test_the_production_runner_actually_runs_a_checker(self, tmp_path: pathlib.Path) -> None:
        """Runs real ruff on a file that genuinely violates a rule.

        A runner asserted only against a fake would pass while invoking
        nothing, which is the failure mode this whole package exists to
        avoid measuring.
        """
        offender = tmp_path / "offender.py"
        offender.write_text("import os\n", encoding="utf-8")
        clean = tmp_path / "clean.py"
        clean.write_text("x = 1\n", encoding="utf-8")

        def _ruff(target: pathlib.Path) -> int:
            """Run real ruff over one file and return its exit status.

            Args:
                target: File to check.

            Returns:
                The exit status.
            """
            return core_hooks.Hooks.run_checker(
                ("python", "-m", "ruff", "check", "--select", "F401", str(target)),
                tmp_path,
                config_test_hooks.get_environment(),
            ).returncode

        # Asserted as a PAIR rather than on output text: the exit statuses
        # must DIFFER between a violating file and a clean one, which is the
        # property the instrument depends on and is stable across versions.
        assert _ruff(offender) == 1
        assert _ruff(clean) == 0

    def test_output_that_the_locale_codec_cannot_decode_still_arrives(
        self, tmp_path: pathlib.Path
    ) -> None:
        """A real process writing byte 0x81, which cp1252 cannot map.

        This is the byte that ended a real sweep. Under ``text=True`` the
        decode happens on subprocess's reader thread, so the failure does not
        surface as an exception here: the thread dies, ``communicate``
        returns None for that stream, and the caller holds a ``stdout`` that
        the protocol in this package promises is a ``str``.

        Asserted against the exact decoded text rather than against a type or
        a truthiness, so it pins WHAT arrives: 0x81 is not valid UTF-8, and
        ``errors="replace"`` renders it as U+FFFD. A None would fail this,
        and so would a silent switch back to the locale codec.

        Args:
            tmp_path: Directory to run the process in.
        """
        finished = core_hooks.Hooks.run_checker(
            (
                "python",
                "-c",
                "import sys; sys.stdout.buffer.write(bytes([0x81])); "
                "sys.stderr.buffer.write(bytes([0x81]))",
            ),
            tmp_path,
            config_test_hooks.get_environment(),
        )

        assert finished.stdout == "�"
        assert finished.stderr == "�"
        assert finished.returncode == 0


class TestResettingTheHookContainer:
    """The container's reset() is the seam an autouse fixture names."""

    def test_reset_restores_the_production_runner(self) -> None:
        """Assigned away, then restored, and the restoration is asserted.

        Not a smoke test: a reset() that quietly did nothing would leave a
        fake checker installed for every test that ran afterwards, and the
        sweep would score whatever that fake said.
        """
        replacement = _Recorder({})
        core_hooks.Hooks.run_checker = replacement
        assert core_hooks.Hooks.run_checker is replacement

        core_hooks.Hooks.reset()

        assert core_hooks.Hooks.run_checker is core_hooks._default_run_checker
