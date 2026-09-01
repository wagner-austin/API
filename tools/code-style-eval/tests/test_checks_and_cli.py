"""Running the checkers, and the CLI that drives a sweep.

The one test that runs a REAL checker is
``test_the_production_runner_actually_runs_a_checker``. It calls ruff on a
file that genuinely violates a rule, because a runner asserted only against
a fake would pass while invoking nothing.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Generator, Sequence

import pytest
from platform_core.json_utils import dump_json_str, load_json_str, narrow_json_to_dict

from code_style_eval.cli import _test_hooks as cli_hooks
from code_style_eval.cli.evaluate import (
    entrypoint,
    generated_path,
    main,
    parse_arguments,
)
from code_style_eval.contracts.outcomes import decode_item_outcome
from code_style_eval.core import _test_hooks as core_hooks
from code_style_eval.core.checks import checker_command, run_check, score_item

_SOURCE = "".join(f"line{i}\n" for i in range(10))


class _Finished:
    """A finished process with the three fields the package reads."""

    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        """Store the result.

        Args:
            returncode: Exit status.
            stdout: Captured stdout.
            stderr: Captured stderr.
        """
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _Recorder:
    """Checker runner that records its calls and replays scripted results."""

    def __init__(self, results: dict[str, _Finished]) -> None:
        """Store the scripted results.

        Args:
            results: Result per checker module name.
        """
        self.results = results
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, command: tuple[str, ...], cwd: pathlib.Path) -> _Finished:
        """Record a call and return its scripted result.

        Args:
            command: The composed argv.
            cwd: Directory the checker would run in.

        Returns:
            The scripted result.
        """
        self.calls.append(command)
        for name, finished in self.results.items():
            if name in command:
                return finished
        return _Finished(0)


@pytest.fixture(autouse=True)
def _reset() -> None:
    """Restore both hook containers around every test."""
    core_hooks.reset_hooks()
    cli_hooks.reset_hooks()


class TestComposingCheckerCommands:
    """Every checker is invoked through a named interpreter."""

    def test_ruff_checks_the_target(self) -> None:
        """A bare name would resolve against PATH to another version."""
        command = checker_command("ruff", "py", pathlib.Path("a.py"))

        assert command == ("py", "-m", "ruff", "check", "a.py")

    def test_mypy_checks_the_target(self) -> None:
        """Same reasoning, same shape."""
        command = checker_command("mypy", "py", pathlib.Path("a.py"))

        assert command == ("py", "-m", "mypy", "a.py")

    def test_the_guards_run_over_their_package(self) -> None:
        """scripts.guard takes no target; it reads the package it runs in."""
        command = checker_command("guards", "py", pathlib.Path("a.py"))

        assert command == ("py", "-m", "scripts.guard")

    def test_an_unknown_checker_is_refused(self) -> None:
        """The set is closed at the composition point too."""
        with pytest.raises(ValueError, match="unknown checker"):
            _ = checker_command("pylint", "py", pathlib.Path("a.py"))


class TestRunningOneCheck:
    """A checker's exit status is the verdict."""

    def test_a_zero_exit_passes_and_carries_no_detail(self) -> None:
        """A clean run has nothing to say."""
        core_hooks.Hooks.run_checker = _Recorder({"ruff": _Finished(0, "all good")})

        outcome = run_check("ruff", "py", pathlib.Path("a.py"), pathlib.Path("."))

        assert outcome["passed"] is True
        assert outcome["exit_code"] == 0
        assert outcome["detail"] == ""

    def test_a_failure_carries_the_first_output_line(self) -> None:
        """An index into the logs, not a replacement for them."""
        core_hooks.Hooks.run_checker = _Recorder(
            {"ruff": _Finished(1, "\n  a.py:1:1 E501 line too long\nmore\n")}
        )

        outcome = run_check("ruff", "py", pathlib.Path("a.py"), pathlib.Path("."))

        assert outcome["passed"] is False
        assert outcome["exit_code"] == 1
        assert outcome["detail"] == "a.py:1:1 E501 line too long"

    def test_stderr_is_used_when_stdout_is_empty(self) -> None:
        """mypy crashes report on stderr, and a crash is not a pass."""
        core_hooks.Hooks.run_checker = _Recorder({"mypy": _Finished(2, "", "internal error")})

        outcome = run_check("mypy", "py", pathlib.Path("a.py"), pathlib.Path("."))

        assert outcome["exit_code"] == 2
        assert outcome["detail"] == "internal error"

    def test_a_failure_with_no_output_still_records_the_code(self) -> None:
        """Silence plus non-zero is still a failure."""
        core_hooks.Hooks.run_checker = _Recorder({"ruff": _Finished(3, "", "")})

        outcome = run_check("ruff", "py", pathlib.Path("a.py"), pathlib.Path("."))

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
            cwd=pathlib.Path("."),
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
            cwd=pathlib.Path("."),
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
            ).returncode

        # Asserted as a PAIR rather than on output text: the exit statuses
        # must DIFFER between a violating file and a clean one, which is the
        # property the instrument depends on and is stable across versions.
        assert _ruff(offender) == 1
        assert _ruff(clean) == 0


class TestParsingArguments:
    """Every required flag is required, and the optional ones default."""

    def _base(self, tmp_path: pathlib.Path) -> list[str]:
        """Minimal valid argv.

        Args:
            tmp_path: Directory for the paths.

        Returns:
            The argv tokens.
        """
        return [
            "--holdout",
            str(tmp_path / "h.jsonl"),
            "--generated-dir",
            str(tmp_path / "gen"),
            "--interpreter",
            "py",
            "--arm",
            "candidate",
            "--out",
            str(tmp_path / "out.jsonl"),
        ]

    def test_a_minimal_command_line_parses(self, tmp_path: pathlib.Path) -> None:
        """Prompt length and check directory default."""
        arguments = parse_arguments(self._base(tmp_path))

        assert arguments.arm == "candidate"
        assert arguments.prompt_lines == 20
        assert arguments.check_cwd == tmp_path / "gen"

    def test_the_check_directory_can_be_overridden(self, tmp_path: pathlib.Path) -> None:
        """The guards must run where the package they check lives."""
        arguments = parse_arguments([*self._base(tmp_path), "--check-cwd", str(tmp_path / "pkg")])

        assert arguments.check_cwd == tmp_path / "pkg"

    @pytest.mark.parametrize(
        "missing",
        ["--holdout", "--generated-dir", "--interpreter", "--arm", "--out"],
    )
    def test_each_required_flag_is_required(self, missing: str, tmp_path: pathlib.Path) -> None:
        """Parametrised so a sixth required flag cannot be added untested.

        Args:
            missing: The flag to drop.
            tmp_path: Directory for the paths.
        """
        tokens = self._base(tmp_path)
        index = tokens.index(missing)
        del tokens[index : index + 2]

        with pytest.raises(ValueError, match=f"{missing} is required"):
            _ = parse_arguments(tokens)

    def test_an_unknown_flag_is_refused(self, tmp_path: pathlib.Path) -> None:
        """A typo must not be silently ignored."""
        with pytest.raises(ValueError, match="unknown argument"):
            _ = parse_arguments([*self._base(tmp_path), "--nope", "x"])

    def test_a_flag_without_a_value_is_refused(self, tmp_path: pathlib.Path) -> None:
        """A trailing flag would otherwise read past the end."""
        with pytest.raises(ValueError, match="requires a value"):
            _ = parse_arguments([*self._base(tmp_path), "--prompt-lines"])

    @pytest.mark.parametrize("bad", ["0", "-3", "many"])
    def test_a_non_positive_prompt_length_is_refused(
        self, bad: str, tmp_path: pathlib.Path
    ) -> None:
        """Zero lines asks the model to write a file from nothing.

        Args:
            bad: The rejected value.
            tmp_path: Directory for the paths.
        """
        with pytest.raises(ValueError, match="positive integer"):
            _ = parse_arguments([*self._base(tmp_path), "--prompt-lines", bad])


class TestLocatingGenerations:
    """Item ids are repository paths, so they are flattened not joined."""

    def test_a_nested_path_is_flattened(self, tmp_path: pathlib.Path) -> None:
        """Joining would let '..' escape the directory."""
        located = generated_path(tmp_path, "src/pkg/mod.py")

        assert located == tmp_path / "src__pkg__mod.py.py"

    def test_a_windows_separator_is_flattened_too(self, tmp_path: pathlib.Path) -> None:
        """The corpus is emitted on Windows as well as read there."""
        located = generated_path(tmp_path, "src\\pkg\\mod.py")

        assert located == tmp_path / "src__pkg__mod.py.py"


class TestTheSweep:
    """End to end over real files on disk."""

    def _holdout(self, tmp_path: pathlib.Path, paths: Sequence[str]) -> pathlib.Path:
        """Write a holdout corpus.

        Args:
            tmp_path: Directory to write into.
            paths: Item paths to include.

        Returns:
            The holdout file.
        """
        holdout = tmp_path / "h.jsonl"
        holdout.write_text(
            "".join(
                dump_json_str({"repo": "api", "path": p, "text": _SOURCE}) + "\n" for p in paths
            ),
            encoding="utf-8",
        )
        return holdout

    def test_a_sweep_writes_one_outcome_per_generated_item(self, tmp_path: pathlib.Path) -> None:
        """Outcomes are one JSON object per line, decodable by the codec."""
        holdout = self._holdout(tmp_path, ["a.py", "b.py"])
        generated = tmp_path / "gen"
        generated.mkdir()
        for name in ("a.py", "b.py"):
            generated_path(generated, name).write_text("x = 1\n", encoding="utf-8")
        core_hooks.Hooks.run_checker = _Recorder({})
        out = tmp_path / "out.jsonl"

        code = main(
            [
                "--holdout",
                str(holdout),
                "--generated-dir",
                str(generated),
                "--interpreter",
                "py",
                "--arm",
                "candidate",
                "--out",
                str(out),
                "--prompt-lines",
                "3",
            ]
        )

        assert code == 0
        lines = out.read_text(encoding="utf-8").splitlines()
        outcomes = [decode_item_outcome(narrow_json_to_dict(load_json_str(x))) for x in lines]
        assert [o["item_id"] for o in outcomes] == ["a.py", "b.py"]
        assert all(o["arm"] == "candidate" for o in outcomes)

    def test_an_item_with_no_generation_is_skipped_not_failed(self, tmp_path: pathlib.Path) -> None:
        """A missing generation is a fact about the run, not the model.

        Recording it as a failure would let a crashed generation masquerade
        as a style result.
        """
        holdout = self._holdout(tmp_path, ["present.py", "absent.py"])
        generated = tmp_path / "gen"
        generated.mkdir()
        generated_path(generated, "present.py").write_text("x = 1\n", encoding="utf-8")
        core_hooks.Hooks.run_checker = _Recorder({})
        out = tmp_path / "out.jsonl"
        emitted: list[str] = []
        cli_hooks.emit = emitted.append

        _ = main(
            [
                "--holdout",
                str(holdout),
                "--generated-dir",
                str(generated),
                "--interpreter",
                "py",
                "--arm",
                "candidate",
                "--out",
                str(out),
                "--prompt-lines",
                "3",
            ]
        )

        assert len(out.read_text(encoding="utf-8").splitlines()) == 1
        assert emitted == ["arm candidate: scored 1 of 2 prompt(s), 1 passed every checker"]

    def test_the_output_directory_is_created(self, tmp_path: pathlib.Path) -> None:
        """A sweep should not fail on a missing runs/ directory."""
        holdout = self._holdout(tmp_path, ["a.py"])
        generated = tmp_path / "gen"
        generated.mkdir()
        generated_path(generated, "a.py").write_text("x = 1\n", encoding="utf-8")
        core_hooks.Hooks.run_checker = _Recorder({})
        out = tmp_path / "nested" / "deep" / "out.jsonl"

        _ = main(
            [
                "--holdout",
                str(holdout),
                "--generated-dir",
                str(generated),
                "--interpreter",
                "py",
                "--arm",
                "candidate",
                "--out",
                str(out),
                "--prompt-lines",
                "3",
            ]
        )

        assert out.is_file()


def _make_argv() -> Generator[list[str], None, None]:
    """Give a test control of ``sys.argv`` and restore it afterwards.

    Yields:
        The live argument list, for the test to replace in place.
    """
    original = list(sys.argv)
    yield sys.argv
    sys.argv[:] = original


# The call form resolves pytest's overloaded decorator to a concrete type;
# the bare @pytest.fixture expression carries Any under disallow_any_expr.
argv = pytest.fixture(_make_argv)


class TestTheEntryPoint:
    """The console script reads the process arguments and carries the code.

    Run for real rather than excluded from coverage. A line excluded because
    it is awkward to reach is a line nobody has ever run.
    """

    def test_the_entry_point_reads_the_process_arguments(
        self, tmp_path: pathlib.Path, argv: list[str]
    ) -> None:
        """A real sweep driven entirely through sys.argv.

        Args:
            tmp_path: Directory for the corpus and outputs.
            argv: The live process arguments, replaced in place.
        """
        holdout = tmp_path / "h.jsonl"
        holdout.write_text(
            dump_json_str({"repo": "api", "path": "a.py", "text": _SOURCE}) + "\n",
            encoding="utf-8",
        )
        generated = tmp_path / "gen"
        generated.mkdir()
        generated_path(generated, "a.py").write_text("x = 1\n", encoding="utf-8")
        core_hooks.Hooks.run_checker = _Recorder({})
        out = tmp_path / "out.jsonl"
        argv[:] = [
            "prog",
            "--holdout",
            str(holdout),
            "--generated-dir",
            str(generated),
            "--interpreter",
            "py",
            "--arm",
            "candidate",
            "--out",
            str(out),
            "--prompt-lines",
            "3",
        ]

        with pytest.raises(SystemExit) as raised:
            entrypoint()

        assert raised.value.code == 0
        assert out.is_file()


class TestTheEmitHook:
    """The CLI's one impure act is injectable."""

    def test_the_default_emit_writes_a_line(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Asserted once so the production path is covered."""
        cli_hooks.reset_hooks()

        cli_hooks.emit("hello")

        assert capsys.readouterr().out == "hello\n"


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
