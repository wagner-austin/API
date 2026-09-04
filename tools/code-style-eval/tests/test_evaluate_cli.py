"""The CLI that drives a sweep, and where it expects generations to live.

The layout is part of the contract rather than an implementation detail: the
guards are scoped to a directory, so where a generated file sits decides
whether they read it at all.
"""

from __future__ import annotations

import importlib.metadata
import pathlib
import sys
from collections.abc import Generator, Sequence

import pytest
from platform_core.json_utils import dump_json_str, load_json_str, narrow_json_to_dict

from code_style_eval.cli import _test_hooks as cli_hooks
from code_style_eval.cli.evaluate import entrypoint, main, parse_arguments
from code_style_eval.contracts.outcomes import decode_item_outcome
from code_style_eval.core import _test_hooks as core_hooks
from tests.conftest import _Recorder, _write_generation

_SOURCE = "".join(f"line{i}" + chr(10) for i in range(10))


def _check_cwd(tmp_path: pathlib.Path) -> pathlib.Path:
    """Build a directory shaped like a package inside the monorepo.

    The checkers are invoked from a package at ``<repo>/<category>/<package>``,
    and the instrument reads the repository root from that shape to find every
    package source root for MYPYPATH. A bare temporary directory is not that
    shape, and is refused rather than silently yielding an empty path list.

    Args:
        tmp_path: The test's directory, standing in for a repository root.

    Returns:
        The package directory to pass as --check-cwd.
    """
    (tmp_path / "libs").mkdir(exist_ok=True)
    package = tmp_path / "tools" / "pkg"
    package.mkdir(parents=True, exist_ok=True)
    return package


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
            "--check-cwd",
            str(tmp_path / "pkg"),
        ]

    def test_a_minimal_command_line_parses(self, tmp_path: pathlib.Path) -> None:
        """Only the prompt length defaults."""
        arguments = parse_arguments(self._base(tmp_path))

        assert arguments.arm == "candidate"
        assert arguments.prompt_lines == 20
        assert arguments.check_cwd == tmp_path / "pkg"

    def test_the_check_directory_has_no_default(self, tmp_path: pathlib.Path) -> None:
        """It used to default to the generated directory, which is wrong.

        The checkers are invoked from a package whose ``scripts/guard.py``
        makes ``python -m scripts.guard`` resolve. The generated directory
        holds no such file, so the old default could only ever have produced
        an import failure scored as a guard verdict. There is no sensible
        default, so the flag is required.
        """
        tokens = self._base(tmp_path)
        index = tokens.index("--check-cwd")
        del tokens[index : index + 2]

        with pytest.raises(ValueError, match="--check-cwd is required"):
            _ = parse_arguments(tokens)

    @pytest.mark.parametrize(
        "missing",
        ["--holdout", "--generated-dir", "--interpreter", "--arm", "--out", "--check-cwd"],
    )
    def test_each_required_flag_is_required(self, missing: str, tmp_path: pathlib.Path) -> None:
        """Parametrised so a seventh required flag cannot be added untested.

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
            _write_generation(generated, name)
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
                "--check-cwd",
                str(_check_cwd(tmp_path)),
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
        _write_generation(generated, "present.py")
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
                "--check-cwd",
                str(_check_cwd(tmp_path)),
            ]
        )

        assert len(out.read_text(encoding="utf-8").splitlines()) == 1
        assert emitted == ["arm candidate: scored 1 of 2 prompt(s), 1 passed every checker"]

    def test_the_output_directory_is_created(self, tmp_path: pathlib.Path) -> None:
        """A sweep should not fail on a missing runs/ directory."""
        holdout = _write_holdout(tmp_path, ["a.py"])
        generated = tmp_path / "gen"
        generated.mkdir()
        _write_generation(generated, "a.py")
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
                "--check-cwd",
                str(_check_cwd(tmp_path)),
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


def _write_holdout(tmp_path: pathlib.Path, paths: Sequence[str]) -> pathlib.Path:
    """Write a holdout corpus.

    Module level rather than a method, because two classes need it and a
    method on one of them is not reachable from the other.

    Args:
        tmp_path: Directory to write into.
        paths: Item paths to include.

    Returns:
        The holdout file.
    """
    holdout = tmp_path / "h.jsonl"
    holdout.write_text(
        "".join(
            dump_json_str({"repo": "api", "path": p, "text": _SOURCE}) + chr(10) for p in paths
        ),
        encoding="utf-8",
    )
    return holdout


class TestRefusingAWrongInstrument:
    """Scoring writes no fingerprint, so it must check one before it starts."""

    def test_a_missing_distribution_stops_the_run(self, tmp_path: pathlib.Path) -> None:
        """The failure this prevents is a run that SUCCEEDS and is wrong.

        ``poetry sync --with dev`` -- which ``make check`` runs every time --
        removes the optional corpus group, and scoring without it quietly
        produces outcomes full of missing-stub verdicts about the sandbox
        rather than about the generated code. Nothing downstream distinguishes
        those from good ones until the comparison refuses, long after.

        Args:
            tmp_path: Temporary directory.
        """
        holdout = _write_holdout(tmp_path, ["a.py"])
        generated = tmp_path / "gen"
        generated.mkdir()
        _write_generation(generated, "a.py")
        core_hooks.Hooks.run_checker = _Recorder({})
        cli_hooks.record_distributions = ("no-such-distribution-exists",)
        out = tmp_path / "out.jsonl"

        with pytest.raises(importlib.metadata.PackageNotFoundError):
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
                    "--check-cwd",
                    str(tmp_path),
                ]
            )

        assert not out.exists()

    def test_the_refusal_precedes_every_checker(self, tmp_path: pathlib.Path) -> None:
        """A refusal after the work has run is a refusal that saved nothing.

        Args:
            tmp_path: Temporary directory.
        """
        holdout = _write_holdout(tmp_path, ["a.py"])
        generated = tmp_path / "gen"
        generated.mkdir()
        _write_generation(generated, "a.py")
        recorder = _Recorder({})
        core_hooks.Hooks.run_checker = recorder
        cli_hooks.record_distributions = ("no-such-distribution-exists",)

        with pytest.raises(importlib.metadata.PackageNotFoundError):
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
                    str(tmp_path / "out.jsonl"),
                    "--check-cwd",
                    str(tmp_path),
                ]
            )

        assert recorder.calls == []


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
        _write_generation(generated, "a.py")
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
            "--check-cwd",
            str(_check_cwd(tmp_path)),
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
