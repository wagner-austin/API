"""The one place a refusal becomes a message instead of a traceback.

Every rule in this package is enforced by raising. That is deliberate and it
stays: a job that would not run must not preflight clean, and the way to
guarantee that is for nothing in between to catch. But the operator standing
in front of a terminal is not helped by eight lines of stack above the line
that matters, and an exit status of 1 that means "the tool refused" is
indistinguishable from an exit status of 1 that means "triage found something".

So exactly one place translates, it is the process boundary, and it is typed.

**Typed** is the load-bearing word. Three exception types are translated,
because each names a thing the operator did that the tool declined to do:

* ``AppError`` -- a rule refused, and it carries a stable code saying which.
* ``JSONTypeError`` -- a document is not the shape the contract requires.
* ``ValueError`` -- the command line itself is wrong.

Anything else propagates and prints a traceback, because anything else is a
defect in this package rather than a refusal by it, and a defect that prints
one tidy line is a defect nobody debugs. That distinction is the whole reason
this is a translator and not an ``except Exception``.

Exit statuses are distinct for the same reason:

===  ===========================================================
  0  the command did what it was asked
  1  the command ran and the answer is negative -- triage found
     something, ``hpc3-trace`` matched nothing
  2  the command refused; nothing was submitted, staged or run
===  ===========================================================

A caller scripting these can branch on that, which it could not do when every
failure was status 1.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from platform_core.cli_args import HelpRequestedError, usage_text
from platform_core.errors import AppError
from platform_core.json_utils import JSONTypeError

from hpc3.cli import _test_hooks

EXIT_OK = 0
EXIT_NEGATIVE = 1
EXIT_REFUSED = 2


def run(main: Callable[[Sequence[str] | None], int]) -> int:
    """Run a command's ``main`` and turn its typed refusals into messages.

    Args:
        main: The command's entry function, which reads the process arguments
            when given None.

    Returns:
        ``main``'s own status when it completed, or :data:`EXIT_REFUSED` when
        it refused.

    Raises:
        Exception: Anything that is not a recognised refusal, unchanged and
            with its traceback. An unexpected exception is a bug in this
            package; presenting it as a tidy one-line message would disguise
            a defect as a decision.
    """
    try:
        return main(None)
    except HelpRequestedError as request:
        # Before the ValueError arm, which HelpRequestedError subclasses. Asking
        # what the flags are is not a refusal: it goes to stdout beside the
        # command's other output and exits zero, so `cmd --help` in a script
        # does not read as the command having declined to run.
        _test_hooks.emit(usage_text(request.allowed))
        return EXIT_OK
    except AppError as refusal:
        # ErrorCodeBase subclasses str, so a member IS its own string value.
        # Annotating rather than reading `.value` keeps the expression typed:
        # Enum.value is Any, and this package admits no Any anywhere.
        code: str = refusal.code
        # CONCATENATED, not interpolated. An f-string calls Enum.__format__,
        # which renders "Hpc3ErrorCode.ENV_PACKAGE_MISMATCH" -- a qualified
        # name nobody can grep the codebase or the board for. `str.__add__`
        # uses the value, which is the half that is stable and searchable.
        _test_hooks.emit_error(code + ": " + refusal.message)
        return EXIT_REFUSED
    except JSONTypeError as invalid:
        _test_hooks.emit_error(f"invalid document: {invalid}")
        return EXIT_REFUSED
    except ValueError as usage:
        _test_hooks.emit_error(f"usage: {usage}")
        return EXIT_REFUSED


__all__ = ["EXIT_NEGATIVE", "EXIT_OK", "EXIT_REFUSED", "run"]
