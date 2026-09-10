"""CLI: record one push, so its CI verdict can be addressed to whoever made it.

Usage:
    ci-wake-enrol --enrolment <path> --repo owner/name --sha <40 hex> --ref <ref>

CALLED FROM ``pre-push``, ONCE PER PUSHED TIP, AND NOWHERE ELSE. Git hands
the hook ``<local-ref> <local-sha> <remote-ref> <remote-sha>`` on stdin; the
hook loops those lines and calls this once each. It is the only moment at
which the answer to "which session is waiting on this sha" exists anywhere,
which is the whole argument of :mod:`ci_wake.enrolment`.

Environment:
    BOARD_AGENT_LABEL   the pushing session's board label. OPTIONAL, and its
        absence is a first-class outcome rather than a failure: a human
        pushing from a terminal has no board label, and their push is
        announced board-level instead of addressed to nobody. A session that
        wants to be woken exports it -- the same variable ``hpc3`` already
        reads when recording a submitter, so one export covers both bridges.

This command writes one line and exits. It does not reach the network, does
not consult GitHub, and does not care whether the push it is recording will
succeed -- see :mod:`ci_wake.enrolment` on why the record names an attempt.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.config import _optional_env_str

from ci_wake import _test_hooks
from ci_wake.enrolment import (
    AGENT_VARIABLE,
    PushAttempt,
    append_attempt,
    require_agent,
    require_repository,
    require_sha,
)

ENROLMENT_FLAG = "--enrolment"
REPO_FLAG = "--repo"
SHA_FLAG = "--sha"
REF_FLAG = "--ref"

ALLOWED_FLAGS = (ENROLMENT_FLAG, REPO_FLAG, SHA_FLAG, REF_FLAG)


def main(argv: Sequence[str]) -> int:
    """Append one enrolment row for the push named on the command line.

    Args:
        argv: Arguments excluding the program name.

    Returns:
        0 always. Every failure raises instead, so the hook that called this
        fails the push rather than printing a status line nobody reads.

    Raises:
        AppError: ``ENROLMENT_FIELD_MALFORMED`` when the repository, sha or
            exported agent label cannot address anything.
        ValueError: A missing or repeated flag.
        OSError: The enrolment record cannot be written.
    """
    parsed = cli_args.parse_single_flags(argv, ALLOWED_FLAGS)
    exported = _optional_env_str(AGENT_VARIABLE)
    record = PushAttempt(
        repo=require_repository(cli_args.require_flag(parsed, REPO_FLAG)),
        sha=require_sha(cli_args.require_flag(parsed, SHA_FLAG)),
        ref=cli_args.require_flag(parsed, REF_FLAG),
        agent=require_agent("" if exported is None else exported),
        attempted_unix=_test_hooks.now(),
    )
    append_attempt(pathlib.Path(cli_args.require_flag(parsed, ENROLMENT_FLAG)), record)
    _test_hooks.emit(
        f"ci-wake: enrolled {record['sha'][:7]} in {record['repo']} for "
        + (f"@{record['agent']}" if record["agent"] != "" else f"nobody (no ${AGENT_VARIABLE})")
    )
    return 0


def entrypoint() -> None:
    """Console-script wrapper.

    Raises:
        SystemExit: Always, carrying :func:`main`'s status.
    """
    raise SystemExit(main(sys.argv[1:]))


# Without this, ``python -m ci_wake.cli.enrol`` imports the module, runs
# nothing and exits 0 -- a form that looks like an enrolment that happened.
if __name__ == "__main__":
    entrypoint()
