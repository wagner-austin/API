"""Shared fakes and the hook reset every test in this package relies on.

The checker runner is the one impure thing the package does, so the fake for
it lives here rather than in one test module: two modules exercise it, and a
second copy would be free to drift into agreeing with a stale contract.

Hooks are reset around every test so a rebinding made by one cannot leak into
another. Leakage here is hard to diagnose, because the symptom is a test that
fails only when it runs after a specific other test, which ``-n auto``
reorders.
"""

from __future__ import annotations

import pathlib
from collections.abc import Mapping

import pytest

from code_style_eval.cli import _test_hooks as cli_hooks
from code_style_eval.cli.evaluate import generated_path
from code_style_eval.core import _test_hooks as core_hooks


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
        self.environments: list[Mapping[str, str]] = []

    def __call__(
        self, command: tuple[str, ...], cwd: pathlib.Path, env: Mapping[str, str]
    ) -> _Finished:
        """Record a call and return its scripted result.

        Args:
            command: The composed argv.
            cwd: Directory the checker would run in.
            env: Environment the checker would run under.

        Returns:
            The scripted result.
        """
        self.calls.append(command)
        self.environments.append(env)
        for name, finished in self.results.items():
            if name in command:
                return finished
        return _Finished(0)


def _write_generation(generated_dir: pathlib.Path, item_id: str) -> pathlib.Path:
    """Write one generation where the instrument expects to find it.

    Args:
        generated_dir: The arm's directory.
        item_id: The item.

    Returns:
        The file written.
    """
    target = generated_path(generated_dir, item_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("x = 1\n", encoding="utf-8")
    return target


#: Distributions the tests fingerprint against: two the dev group guarantees.
#:
#: Production names those plus the corpus group, which ``poetry sync --with
#: dev`` removes before every ``make check``. Binding the seam here keeps the
#: suite from depending on a ~2 GB optional install to reach one line of the
#: compare CLI; that the production default is the wider set is asserted
#: structurally in ``test_provenance.py`` instead of by capturing it.
TEST_DISTRIBUTIONS: tuple[str, ...] = ("ruff", "mypy")


@pytest.fixture(autouse=True)
def _reset() -> None:
    """Restore both hook containers around every test, then narrow one seam."""
    core_hooks.reset_hooks()
    cli_hooks.reset_hooks()
    cli_hooks.record_distributions = TEST_DISTRIBUTIONS
