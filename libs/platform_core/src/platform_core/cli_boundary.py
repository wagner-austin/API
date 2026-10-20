"""The one place a command-line refusal becomes a message instead of a traceback.

Lifted from ``tools/hpc3``'s ``cli/_fatal.py`` on 2026-10-20 (MCPs board
task 1e9ab738), when ``fleet-runners --help`` in ``tools/fleet`` ended in an
uncaught :class:`~platform_core.cli_args.HelpRequestedError` traceback:
hpc3 was the one package of the ten calling
:func:`~platform_core.cli_args.parse_single_flags` that translated it.

Every rule a command enforces is enforced by raising. That is deliberate and
it stays: a job that would not run must not preflight clean, and the way to
guarantee that is for nothing in between to catch. But the operator standing
in front of a terminal is not helped by eight lines of stack above the line
that matters, and an exit status of 1 that means "the tool refused" is
indistinguishable from an exit status of 1 that means "the audit found
drift".

So exactly one place translates, it is the process boundary, and it is typed.
Three exception types are translated, because each names a thing the caller
did that the command declined to do:

* ``AppError`` -- a rule refused, and it carries a stable code saying which.
* ``JSONTypeError`` -- a document is not the shape the contract requires.
* ``ValueError`` -- the command line itself is wrong.

Anything else propagates and prints a traceback, because anything else is a
defect in the command rather than a refusal by it, and a defect that prints
one tidy line is a defect nobody debugs. A help request is not a refusal at
all: it is answered with the usage line on the report stream and status 0.

===  ===========================================================
  0  the command did what it was asked
  1  the command ran and the answer is negative: an audit found
     drift, a trace matched nothing
  2  the command refused; nothing was submitted, staged or run
===  ===========================================================
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from platform_core.cli_args import HelpRequestedError, usage_text
from platform_core.errors import AppError
from platform_core.json_utils import JSONTypeError
from platform_core.report_line import EmitProtocol

EXIT_OK = 0
EXIT_NEGATIVE = 1
EXIT_REFUSED = 2


def run_cli(
    main: Callable[[Sequence[str] | None], int],
    *,
    emit: EmitProtocol,
    emit_error: EmitProtocol,
) -> int:
    """Run a command's ``main`` and turn its typed refusals into messages.

    Args:
        main: The command's entry function, which reads the process arguments
            when given None.
        emit: Where the usage line of a help request goes: the command's
            report stream.
        emit_error: Where a refusal's one line goes: the command's error
            stream, so a refusal never lands in a file something else will
            parse as the report.

    Returns:
        ``main``'s own status when it completed, :data:`EXIT_OK` for a help
        request, or :data:`EXIT_REFUSED` when it refused.

    Raises:
        Exception: Anything that is not a recognised refusal, unchanged and
            with its traceback. An unexpected exception is a defect in the
            command; presenting it as a tidy one-line message would disguise
            a defect as a decision.
    """
    try:
        return main(None)
    except HelpRequestedError as request:
        # Asking what the flags are is not a refusal, so it goes to the report
        # stream and exits zero: `cmd --help` in a script must not read as the
        # command having declined to run.
        emit(usage_text(request.allowed))
        return EXIT_OK
    except AppError as refusal:
        # ErrorCodeBase subclasses str, so a member IS its own string value.
        # Annotating rather than reading `.value` keeps the expression typed:
        # Enum.value is Any. CONCATENATED, not interpolated: an f-string calls
        # Enum.__format__, which renders the qualified "Hpc3ErrorCode.X" that
        # nobody can grep for, where str.__add__ uses the bare value.
        code: str = refusal.code
        emit_error(code + ": " + refusal.message)
        return EXIT_REFUSED
    except JSONTypeError as invalid:
        emit_error(f"invalid document: {invalid}")
        return EXIT_REFUSED
    except ValueError as usage:
        emit_error(f"usage: {usage}")
        return EXIT_REFUSED


__all__ = ["EXIT_NEGATIVE", "EXIT_OK", "EXIT_REFUSED", "run_cli"]
