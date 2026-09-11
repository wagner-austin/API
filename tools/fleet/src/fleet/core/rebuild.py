"""Executing a ``build-bases`` queue job on the hub itself.

THE ONE VERB THAT RUNS HERE INSTEAD OF ON A NODE (MCPs board task 3c9033ff,
rule R6). Every other dispatch command stages a project onto a fleet node
over ssh; the docker base images exist only on the hub, so this one is a
local ``make`` in the MCPs checkout. It is also the only verb that is SAFE
to run unattended: ``make build-bases`` takes the fleet lock, self-skips
when the bases are already fresh (MCPs e3b1baa3), and applies no database
migration -- which is why ``deploy`` is deliberately absent from the
vocabulary end to end.

SYNCHRONOUS, UNLIKE A SUITE. A node dispatch launches detached and is
collected next tick because the ssh connection must not own the build's
lifetime. A local make has no such constraint, a bake is minutes, and
closing the job in the same tick means a died runner leaves nothing
half-collected: the job sits ``running`` until its lease lapses, becomes
reclaimable, and the next tick simply runs the idempotent make again.

THE SUBMITTER'S LABEL TRAVELS AS A MAKE COMMAND-LINE VARIABLE, measured
before it was trusted: ``make BOARD_AGENT_LABEL=<label> <target>`` exports
the variable into recipe environments (probed 2026-09-11 with the parent
environment explicitly unset, under the ``SHELL := powershell.exe`` the
real Makefile uses). That is what stamps the fleet journal's ``agent``
field with the SUBMITTING session -- the ledger convention: rows name who
asked for the work, and since MCPs e3b1baa3 the lock REFUSES an anonymous
acquire, so an unlabelled invocation would not merely be untraceable, it
would not run.

THE LABEL GRAMMAR GUARD IS LOAD-BEARING. The queue's ``agent`` field is
length-validated only, and the label lands inside an argv element and then
in the journal; a label outside the board's kebab alphabet is refused here
with a named detail rather than carried into the fleet record. One argv
element is one make assignment, so no value can smuggle a second variable
-- the guard keeps the RECORD clean, not just the call.
"""

from __future__ import annotations

import pathlib
import re
from typing import Final

from fleet.core import _test_hooks
from fleet.core._test_hooks import CommandResult

#: The board's agent-label grammar, enforced runner-side because the queue
#: only length-checks it (see module docstring).
AGENT_LABEL_PATTERN: Final = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")

#: How much captured output a closing detail keeps. The tail, because make
#: failures end with the failing rule and its error.
DETAIL_TAIL_CHARS: Final = 1500

#: Detail prefix for a runner started without the MCPs checkout path.
ROOT_MISSING_CODE: Final = "REBUILD_ROOT_MISSING"

#: Detail prefix for a submitter label outside the board grammar.
LABEL_INVALID_CODE: Final = "REBUILD_LABEL_INVALID"


def refusal_for(mcps_root: pathlib.Path, submitted_by: str) -> str | None:
    """Decide whether this runner can honestly execute a rebuild job.

    The absent-flag case is the CLI's to refuse (it owns the flag and its
    name); this function judges what it is given: the checkout and the
    label.

    Args:
        mcps_root: The MCPs checkout path the runner was started with.
        submitted_by: The queue row's submitting agent label.

    Returns:
        A ``CODE: message`` refusal detail for the queue, or None when the
        job can run.
    """
    if not _test_hooks.directory_exists(mcps_root):
        return f"{ROOT_MISSING_CODE}: --mcps-root {mcps_root} is not a directory on this machine"
    if AGENT_LABEL_PATTERN.fullmatch(submitted_by) is None:
        return (
            f"{LABEL_INVALID_CODE}: submitter label {submitted_by!r} is "
            "outside the board's kebab-case grammar; the label is exported "
            "into the fleet journal and an unparseable one would poison "
            "every consumer that mentions by it"
        )
    return None


def rebuild_argv(mcps_root: pathlib.Path, submitted_by: str) -> tuple[str, ...]:
    """Compose the make invocation for one rebuild job.

    Args:
        mcps_root: The MCPs checkout (already existence-checked).
        submitted_by: The submitting label (already grammar-checked).

    Returns:
        The argv: ``make -C <root> build-bases BOARD_AGENT_LABEL=<label>``.
        The label rides as a make command-line variable because make exports
        those into recipe environments (measured; module docstring), which
        is the only way the seam's env-less ``run`` can stamp the journal.
    """
    return (
        "make",
        "-C",
        str(mcps_root),
        "build-bases",
        f"BOARD_AGENT_LABEL={submitted_by}",
    )


def run_build_bases(mcps_root: pathlib.Path, *, submitted_by: str) -> CommandResult:
    """Run the bake, blocking until it finishes.

    Args:
        mcps_root: The MCPs checkout.
        submitted_by: The submitting label, stamped into the fleet journal.

    Returns:
        The make invocation's exit status and captured streams.
    """
    return _test_hooks.run(rebuild_argv(mcps_root, submitted_by))


def describe_result(result: CommandResult) -> str:
    """Compose a closing detail from a finished bake.

    Args:
        result: The make invocation's outcome.

    Returns:
        The exit code and the tail of the combined output -- the tail,
        because a make failure ends with the failing rule.
    """
    combined = (result["stdout"] + result["stderr"]).strip()
    tail = combined[-DETAIL_TAIL_CHARS:]
    return f"make build-bases exited {result['returncode']}: {tail}"


__all__ = [
    "AGENT_LABEL_PATTERN",
    "DETAIL_TAIL_CHARS",
    "LABEL_INVALID_CODE",
    "ROOT_MISSING_CODE",
    "describe_result",
    "rebuild_argv",
    "refusal_for",
    "run_build_bases",
]
