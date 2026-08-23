"""Tests for the one place a refusal becomes a message.

The property that matters most here is the negative one: an exception this
package did not anticipate must still reach the operator as a traceback. A
translator that turned every failure into a tidy line would disguise a defect
as a decision, and the defect would never get debugged.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONTypeError

from hpc3.cli import _fatal
from hpc3.cli._fatal import EXIT_NEGATIVE, EXIT_OK, EXIT_REFUSED


def _raising(error: BaseException) -> Callable[[Sequence[str] | None], int]:
    """Build a main-like callable that raises.

    Args:
        error: What it should raise.

    Returns:
        A callable matching the entry-function shape.
    """

    def _main(argv: Sequence[str] | None) -> int:
        raise error

    return _main


def _returning(status: int) -> Callable[[Sequence[str] | None], int]:
    """Build a main-like callable that returns a status.

    Args:
        status: What it should return.

    Returns:
        A callable matching the entry-function shape.
    """

    def _main(argv: Sequence[str] | None) -> int:
        return status

    return _main


class TestStatusPassesThrough:
    def test_success_is_reported_as_success(self, errors: list[str]) -> None:
        assert _fatal.run(_returning(EXIT_OK)) == EXIT_OK
        assert errors == []

    def test_a_negative_answer_is_not_a_refusal(self, errors: list[str]) -> None:
        """triage finding something, or trace matching nothing, is status 1."""
        assert _fatal.run(_returning(EXIT_NEGATIVE)) == EXIT_NEGATIVE
        assert errors == []

    def test_the_three_statuses_are_distinct(self) -> None:
        """A caller scripting these has to be able to tell them apart."""
        assert sorted({EXIT_OK, EXIT_NEGATIVE, EXIT_REFUSED}) == [0, 1, 2]


class TestRefusalsBecomeMessages:
    def test_a_rule_refusal_prints_its_code_and_message(self, errors: list[str]) -> None:
        refusal = AppError(
            Hpc3ErrorCode.ENV_PACKAGE_MISMATCH,
            "/pub/envs/abl has torch==2.11.0+cu128, but this project pins 2.6.0+cu124.",
        )
        assert _fatal.run(_raising(refusal)) == EXIT_REFUSED
        assert errors == [
            "ENV_PACKAGE_MISMATCH: /pub/envs/abl has torch==2.11.0+cu128, "
            "but this project pins 2.6.0+cu124."
        ]

    def test_the_code_is_the_bare_name_not_an_enum_repr(self, errors: list[str]) -> None:
        """'Hpc3ErrorCode.X' is not a thing anyone can search for."""
        _fatal.run(_raising(AppError(Hpc3ErrorCode.CLUSTER_UNKNOWN, "no such cluster")))
        assert errors[0].startswith("CLUSTER_UNKNOWN: ")

    def test_a_malformed_document_is_named_as_one(self, errors: list[str]) -> None:
        assert _fatal.run(_raising(JSONTypeError("Field 'gpu' must be a string"))) == EXIT_REFUSED
        assert errors == ["invalid document: Field 'gpu' must be a string"]

    def test_a_bad_command_line_is_named_as_usage(self, errors: list[str]) -> None:
        assert _fatal.run(_raising(ValueError("--config is required"))) == EXIT_REFUSED
        assert errors == ["usage: --config is required"]

    def test_nothing_is_written_to_the_report_stream(self, emitted: list[str]) -> None:
        """A refusal on stdout would land in whatever the report was piped to."""
        _fatal.run(_raising(ValueError("--config is required")))
        assert emitted == []


class TestUnexpectedFailuresStillRaise:
    """The negative property, and the reason this is a translator rather than
    an ``except Exception``. Each of these is a bug in this package, not a
    refusal by it, and a bug that prints one tidy line is a bug nobody fixes.
    """

    def test_a_programming_error_propagates(self) -> None:
        with pytest.raises(AttributeError):
            _fatal.run(_raising(AttributeError("'NoneType' has no attribute 'name'")))

    def test_a_missing_key_propagates(self) -> None:
        with pytest.raises(KeyError):
            _fatal.run(_raising(KeyError("partition")))

    def test_an_unreadable_file_propagates(self) -> None:
        """The tool did not refuse; the machine did, and that is a stack worth
        seeing because the fix is not in the caller's document."""
        with pytest.raises(OSError):
            _fatal.run(_raising(OSError("permission denied")))

    def test_a_keyboard_interrupt_is_not_swallowed(self) -> None:
        with pytest.raises(KeyboardInterrupt):
            _fatal.run(_raising(KeyboardInterrupt()))
