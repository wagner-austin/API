"""A world the lifted mutation step's tests control (MCPs board task e52e5148).

MCPs' own world for the same core (``mcp-shared-py/tests/_mutation_world.py``,
board task fec5851c), bound to API's boundary: :class:`World` builds a
:class:`~maketools.mutation.context.MutationHooks` whose ``run`` answers each
command from a function the test gives, and records every command with its
environment; its files live in a dict, its clock reads the times the test
lists, its output is a list of lines, and the spans the step records land in
:attr:`World.spans`. The step reads JSON with API's codec
(:data:`maketools.mutation.launch.API_CODEC`). ``test_mutation_launch.py``
runs the real bundle these stand in for.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from platform_core.json_utils import JSONValue

from maketools.mutation.context import MutationHooks, Ran, StepContext
from maketools.mutation.launch import API_CODEC

PACKAGE = Path("/pkg")
RUN_DIR = Path("/scratch/run")
PYTHON = "/pkg/.venv/bin/python"
COSMIC_RAY = "/pkg/.venv/bin/cosmic-ray"
ENVIRONMENT = {"PATH": "/usr/bin", "CORVIS_CHECK_SPANS": "/tmp/spans"}
RUN_ENVIRONMENT = {"PATH": "/usr/bin", "PYTHONDONTWRITEBYTECODE": "1"}


def ok(stdout: str = "") -> Ran:
    """A command that succeeded.

    Args:
        stdout: What it printed.

    Returns:
        Its ending.
    """
    return Ran(exit_code=0, stdout=stdout, stderr="")


def failed(exit_code: int, stderr: str) -> Ran:
    """A command that failed.

    Args:
        exit_code: Its nonzero exit.
        stderr: What it said.

    Returns:
        Its ending.
    """
    return Ran(exit_code=exit_code, stdout="", stderr=stderr)


class World:
    """Fake hooks and the record of what the step did through them."""

    def __init__(self, answer: Callable[[list[str]], Ran], times: Sequence[float]) -> None:
        """Make the world.

        Args:
            answer: The ending of each command, given its argv.
            times: What the clock reads, one value per call, in order.
        """
        self.answer = answer
        self.times = [*times]
        self.calls: list[tuple[list[str], dict[str, str]]] = []
        self.files: dict[Path, str] = {}
        self.lines: list[str] = []
        self.removed: list[Path] = []
        self.spans: list[int] = []

    def _run(self, argv: Sequence[str], env: Mapping[str, str]) -> Ran:
        self.calls.append(([*argv], {**env}))
        return self.answer([*argv])

    def _now(self) -> float:
        return self.times.pop(0)

    def _write_text(self, path: Path, text: str) -> None:
        self.files[path] = text

    def _read_text(self, path: Path) -> str:
        return self.files[path]

    def hooks(self) -> MutationHooks:
        """The bundle the step calls through.

        Returns:
            Hooks bound to this world.
        """
        return MutationHooks(
            run=self._run,
            now=self._now,
            make_run_directory=lambda: RUN_DIR,
            remove_tree=self.removed.append,
            write_text=self._write_text,
            read_text=self._read_text,
            write=self.lines.append,
        )

    def context(self, prepare: str = "") -> StepContext[JSONValue]:
        """A run of the step bound to this world, with API's codec.

        Args:
            prepare: The package's prepare command, or empty.

        Returns:
            The context, recording its spans into :attr:`spans`.
        """
        return StepContext(
            hooks=self.hooks(),
            codec=API_CODEC,
            package_dir=PACKAGE,
            python=PYTHON,
            prepare=prepare,
            environment=ENVIRONMENT,
            run_dir=RUN_DIR,
            record_span=self.spans.append,
        )

    def argvs(self) -> list[list[str]]:
        """Every command run, in order.

        Returns:
            Their argv lists.
        """
        return [argv for argv, _ in self.calls]
