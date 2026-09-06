"""Fakes and the hook reset that keeps tests independent.

Everything here is a FAKE, not a mock. Each implements the same Protocol the
production implementation does and records what it was asked for, so an
assertion is about the arguments this package builds rather than about a
patching library's call-recording API. Nothing patches anything: the hooks in
:mod:`commit_scope._test_hooks` are module-level names and a test rebinds
them.

HOOKS ARE RESET BEFORE AND AFTER EVERY TEST. A rebinding that leaked would
produce a test that fails only after a specific other one, and ``-n auto``
reorders freely, so the cause would be invisible in the failing test.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from platform_core.config import config_test_hooks

from commit_scope import _test_hooks


class FakeGit:
    """A git that answers from a script, and records what it was asked.

    Attributes:
        answers: Output to return, keyed by the arguments tuple.
        calls: Every argument tuple received, in order.
    """

    def __init__(self, answers: dict[tuple[str, ...], str]) -> None:
        """Bind the scripted answers.

        Args:
            answers: Output per argument tuple. A tuple absent from this
                mapping is a test asking a question it did not intend to,
                which raises rather than returning an empty string.
        """
        self.answers = answers
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, arguments: tuple[str, ...]) -> str:
        """Answer one scripted git call.

        Args:
            arguments: Arguments after the program name.

        Returns:
            The scripted output.

        Raises:
            KeyError: When the test did not script this call. Deliberately
                unguarded: a fake that invented an answer would let a change
                in which git commands run pass unnoticed.
        """
        self.calls.append(arguments)
        return self.answers[arguments]


class FakeEnv:
    """An environment holding one declaration.

    Attributes:
        values: Variable values, absent keys reading as None.
        names: Every variable name requested, in order.
    """

    def __init__(self, values: dict[str, str]) -> None:
        """Bind the variables.

        Args:
            values: The variables that are set.
        """
        self.values = values
        self.names: list[str] = []

    def __call__(self, name: str) -> str | None:
        """Read one variable.

        Args:
            name: The variable name.

        Returns:
            Its value, or None when unset -- matching the real reader, which
            normalises a whitespace-only value to None.
        """
        self.names.append(name)
        return self.values.get(name)


class FakeEmit:
    """A sink that keeps every line.

    Attributes:
        lines: The lines written, in order.
    """

    def __init__(self) -> None:
        """Start with nothing written."""
        self.lines: list[str] = []

    def __call__(self, line: str) -> None:
        """Record one line.

        Args:
            line: The line, without a trailing newline.
        """
        self.lines.append(line)

    @property
    def text(self) -> str:
        """The whole report as one string.

        Returns:
            Every line joined by newlines.
        """
        return "\n".join(self.lines)


@pytest.fixture(autouse=True)
def _restore_hooks() -> Generator[None, None, None]:
    """Restore every hook after each test.

    ``config_test_hooks`` is included even though it belongs to
    ``platform_core``: :mod:`tests.test_hook_defaults` rebinds it to exercise
    the delegation, and a leak there would silently change what
    :func:`commit_scope._test_hooks.env` reads for every later test in the
    same worker process.

    Yields:
        None, once, with the production bindings in place.
    """
    original_run_git = _test_hooks.run_git
    original_env = _test_hooks.env
    original_emit = _test_hooks.emit
    original_get_env = config_test_hooks.get_env
    yield
    _test_hooks.run_git = original_run_git
    _test_hooks.env = original_env
    _test_hooks.emit = original_emit
    config_test_hooks.get_env = original_get_env
