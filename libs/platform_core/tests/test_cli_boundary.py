"""The one place a command-line refusal becomes a message.

The property that matters most here is the negative one: an exception a
command did not anticipate must still reach the operator as a traceback. A
translator that turned every failure into a tidy line would disguise a defect
as a decision, and the defect would never get debugged.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest

from platform_core.cli_args import HelpRequestedError
from platform_core.cli_boundary import EXIT_NEGATIVE, EXIT_OK, EXIT_REFUSED, run_cli
from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONTypeError


class _Streams:
    """The two streams :func:`run_cli` writes to, captured as lists."""

    def __init__(self) -> None:
        """Start both empty."""
        self.emitted: list[str] = []
        self.errors: list[str] = []

    def run(self, main: Callable[[Sequence[str] | None], int]) -> int:
        """Run ``main`` through the boundary with both streams captured.

        Args:
            main: The entry function.

        Returns:
            The boundary's status.
        """
        return run_cli(main, emit=self.emitted.append, emit_error=self.errors.append)


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
    """Build a main-like callable that returns a status, having read argv.

    Args:
        status: What it should return.

    Returns:
        A callable matching the entry-function shape, which refuses any argv
        but None: the boundary hands main the process arguments to read.
    """

    def _main(argv: Sequence[str] | None) -> int:
        if argv is not None:
            raise AssertionError(f"main was handed {argv!r}, not None")
        return status

    return _main


class TestStatusPassesThrough:
    def test_success_is_reported_as_success(self) -> None:
        streams = _Streams()
        assert streams.run(_returning(EXIT_OK)) == EXIT_OK
        assert (streams.emitted, streams.errors) == ([], [])

    def test_a_negative_answer_is_not_a_refusal(self) -> None:
        """An audit finding drift, or a trace matching nothing, is status 1."""
        streams = _Streams()
        assert streams.run(_returning(EXIT_NEGATIVE)) == EXIT_NEGATIVE
        assert (streams.emitted, streams.errors) == ([], [])

    def test_the_three_statuses_are_distinct(self) -> None:
        """A caller scripting these has to be able to tell them apart."""
        assert sorted({EXIT_OK, EXIT_NEGATIVE, EXIT_REFUSED}) == [0, 1, 2]


class TestHelpIsNotARefusal:
    """``--help`` is the one thing a caller who does not know the flags can
    type, and it used to end in a traceback in every package but hpc3.
    """

    def test_it_prints_usage_on_the_report_stream_and_succeeds(self) -> None:
        streams = _Streams()
        status = streams.run(_raising(HelpRequestedError(("--config", "--run"))))
        assert status == EXIT_OK
        assert streams.emitted == ["usage: --config <value> --run <value>"]
        assert streams.errors == []

    def test_a_bad_command_line_is_still_a_refusal(self) -> None:
        """The neighbouring case, kept beside it so the two cannot be
        confused. A mistyped flag is wrong and refuses at status 2; asking
        what the flags are is a question and succeeds at status 0.
        """
        streams = _Streams()
        status = streams.run(_raising(ValueError("unknown argument '--turbo'")))
        assert (status, streams.emitted) == (EXIT_REFUSED, [])
        assert streams.errors == ["usage: unknown argument '--turbo'"]


class TestRefusalsBecomeMessages:
    def test_a_rule_refusal_prints_its_code_and_message(self) -> None:
        streams = _Streams()
        refusal = AppError(
            Hpc3ErrorCode.ENV_PACKAGE_MISMATCH,
            "/pub/envs/abl has torch==2.11.0+cu128, but this project pins 2.6.0+cu124.",
        )
        assert streams.run(_raising(refusal)) == EXIT_REFUSED
        assert streams.errors == [
            "ENV_PACKAGE_MISMATCH: /pub/envs/abl has torch==2.11.0+cu128, "
            "but this project pins 2.6.0+cu124."
        ]

    def test_the_code_is_the_bare_name_not_an_enum_repr(self) -> None:
        """'Hpc3ErrorCode.X' is not a thing anyone can search for."""
        streams = _Streams()
        streams.run(_raising(AppError(Hpc3ErrorCode.CLUSTER_UNKNOWN, "no such cluster")))
        assert streams.errors == ["CLUSTER_UNKNOWN: no such cluster"]

    def test_a_malformed_document_is_named_as_one(self) -> None:
        streams = _Streams()
        status = streams.run(_raising(JSONTypeError("Field 'gpu' must be a string")))
        assert status == EXIT_REFUSED
        assert streams.errors == ["invalid document: Field 'gpu' must be a string"]

    def test_a_bad_command_line_is_named_as_usage(self) -> None:
        streams = _Streams()
        assert streams.run(_raising(ValueError("--config is required"))) == EXIT_REFUSED
        assert streams.errors == ["usage: --config is required"]

    def test_nothing_is_written_to_the_report_stream(self) -> None:
        """A refusal on stdout would land in whatever the report was piped to."""
        streams = _Streams()
        streams.run(_raising(ValueError("--config is required")))
        assert streams.emitted == []


class TestUnexpectedFailuresStillRaise:
    """The negative property, and the reason this is a translator rather than
    an ``except Exception``. Each of these is a defect in a command, not a
    refusal by it, and a defect that prints one tidy line is one nobody fixes.
    """

    def test_a_programming_error_propagates(self) -> None:
        with pytest.raises(AttributeError):
            _Streams().run(_raising(AttributeError("'NoneType' has no attribute 'name'")))

    def test_a_missing_key_propagates(self) -> None:
        with pytest.raises(KeyError):
            _Streams().run(_raising(KeyError("partition")))

    def test_an_unreadable_file_propagates(self) -> None:
        """The command did not refuse; the machine did, and that is a stack
        worth seeing because the fix is not in the caller's document."""
        with pytest.raises(OSError):
            _Streams().run(_raising(OSError("permission denied")))

    def test_a_keyboard_interrupt_is_not_swallowed(self) -> None:
        with pytest.raises(KeyboardInterrupt):
            _Streams().run(_raising(KeyboardInterrupt()))
