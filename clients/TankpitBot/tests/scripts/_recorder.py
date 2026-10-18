"""A command runner that answers from a script and records every argv.

Shared by the tests of the scripts that operate the game host
(``scripts.fleet_host``, ``scripts.fleet_release`` and ``scripts.sim_host``): every command they run
goes through ``scripts._test_hooks.run_command``, which these tests replace
with a :class:`Recorder`, so each test asserts the exact commands sent.
"""

from __future__ import annotations

from pathlib import Path

from scripts import _test_hooks as script_hooks


class Recorder:
    """A command runner that answers from a script and records every argv."""

    def __init__(
        self,
        answers: dict[str, script_hooks.CommandResult | list[script_hooks.CommandResult]],
    ) -> None:
        """Hold the answers.

        Args:
            answers: Keyed by a word the command line contains; the first
                key found in the joined argv answers. A list is answered
                in order, its last answer repeating. Unmatched commands
                succeed with empty output.
        """
        self.answers = answers
        self.calls: list[list[str]] = []
        self.staged: dict[str, str] = {}

    def __call__(self, argv: list[str], cwd: Path) -> script_hooks.CommandResult:
        """Record one command and answer it.

        Args:
            argv: The command.
            cwd: Ignored; every command runs from the package directory.

        Returns:
            The scripted answer.
        """
        self.calls.append(argv)
        if argv[0] == "scp":
            for source in argv[2:-1]:
                self.staged[Path(source).name] = Path(source).read_text(encoding="utf-8")
        line = " ".join(argv)
        for word, answer in self.answers.items():
            if word not in line:
                continue
            if not isinstance(answer, list):
                return answer
            return answer.pop(0) if len(answer) > 1 else answer[0]
        return script_hooks.CommandResult(returncode=0, stdout="", stderr="")


def ok(stdout: str) -> script_hooks.CommandResult:
    """A successful command with ``stdout``.

    Args:
        stdout: Its standard output.

    Returns:
        The command result.
    """
    return script_hooks.CommandResult(returncode=0, stdout=stdout, stderr="")


def fail(stderr: str) -> script_hooks.CommandResult:
    """A failed command with ``stderr``.

    Args:
        stderr: Its standard error.

    Returns:
        The command result, exit 1.
    """
    return script_hooks.CommandResult(returncode=1, stdout="", stderr=stderr)
