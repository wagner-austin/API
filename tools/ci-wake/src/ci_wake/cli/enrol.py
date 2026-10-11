"""CLI: record one push, so its CI verdict can be addressed to whoever made it.

Usage:
    ci-wake-enrol --enrolment <path> --repo owner/name --sha <40 hex> --ref <ref>

CALLED FROM ``pre-push``, ONCE PER PUSHED TIP, AND NOWHERE ELSE. Git hands
the hook ``<local-ref> <local-sha> <remote-ref> <remote-sha>`` on stdin; the
hook loops those lines and calls this once each. It is the only moment at
which the answer to "which session is waiting on this sha" exists anywhere,
which is the whole argument of :mod:`ci_wake.enrolment`.

Environment:
    CLAUDE_CODE_SESSION_ID   the acting session's id, exported by the harness
        into every shell it spawns. When set, the board is asked which label
        it bound to that session (MCPs board task 3843d29f): an unset
        BOARD_AGENT_LABEL is filled from the answer, and one naming a
        different label REFUSES the push, the same shape as the board's
        TASK_IDENTITY_MISMATCH, because a wrong name enrolled here is a
        wrong name the bridge @mentions for as long as the push is open.
    BOARD_AGENT_LABEL   the pushing session's board label. Outside a session
        (no session id: a human pushing from a terminal) it is taken as
        given, and its absence is a first-class outcome rather than a
        failure: that push is announced board-level instead of addressed
        to nobody. The same variable ``hpc3`` resolves when recording a
        submitter, so both bridges name the same session the same way.

This command writes at most one line and exits. Inside a session it reaches
the taskboard where the MCPs stack declares it to ask who the session is.
It asks the repository being pushed, through git in the hook's directory,
for the commit the pushed object names and the workflows in that commit,
and records the push only when some workflow's push trigger admits the
remote ref (MCPs board task 24547f50: a tag no workflow runs on was enrolled
and announced as NO RUN EVER APPEARED). It does not consult GitHub, and does
not care whether the push it is recording will succeed -- see
:mod:`ci_wake.enrolment` on why the record names an attempt.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.config import _optional_env_str
from platform_core.session_label import LABEL_VARIABLE, SESSION_ID_VARIABLE, resolve_label

from ci_wake import _test_hooks
from ci_wake.enrolment import (
    PushAttempt,
    append_attempt,
    require_agent,
    require_ref,
    require_repository,
    require_sha,
)
from ci_wake.pushed_commit import WORKFLOWS_DIR, peel_commit, read_workflows
from ci_wake.workflow_triggers import admits, describe

ENROLMENT_FLAG = "--enrolment"
REPO_FLAG = "--repo"
SHA_FLAG = "--sha"
REF_FLAG = "--ref"

ALLOWED_FLAGS = (ENROLMENT_FLAG, REPO_FLAG, SHA_FLAG, REF_FLAG)


def main(argv: Sequence[str]) -> int:
    """Append one enrolment row for the push named on the command line.

    Or none, saying so with every workflow it read, when no workflow in the
    pushed commit starts for the remote ref.

    Args:
        argv: Arguments excluding the program name.

    Returns:
        0 always. Every failure raises instead, so the hook that called this
        fails the push rather than printing a status line nobody reads.

    Raises:
        AppError: ``ENROLMENT_FIELD_MALFORMED`` when the repository, sha,
            ref or resolved agent label cannot address anything;
            ``PUSHED_COMMIT_UNREADABLE`` when git cannot peel the pushed
            object or read its workflows; ``WORKFLOW_TRIGGER_UNREAD`` when
            a workflow's ``on:`` is in a shape the reader does not read;
            ``SESSION_LABEL_MISMATCH`` when the shell's label is not the one
            the board bound to the acting session; ``SESSION_ID_MALFORMED``,
            ``STACK_ENDPOINT_UNDECLARED`` or
            ``SESSION_LABEL_CREDENTIALS_MISSING`` when the board cannot
            be asked, and the :class:`McpClientErrorCode` failures when it
            does not answer.
        ValueError: A missing or repeated flag.
        OSError: The enrolment record cannot be written.
    """
    parsed = cli_args.parse_single_flags(argv, ALLOWED_FLAGS)
    agent = require_agent(
        resolve_label(
            session_id=_optional_env_str(SESSION_ID_VARIABLE),
            exported=_optional_env_str(LABEL_VARIABLE),
            read_text=_test_hooks.read_text,
            post=_test_hooks.http_post,
        )
    )
    repo = require_repository(cli_args.require_flag(parsed, REPO_FLAG))
    ref = require_ref(cli_args.require_flag(parsed, REF_FLAG))
    commit = peel_commit(require_sha(cli_args.require_flag(parsed, SHA_FLAG)))
    workflows = read_workflows(commit)
    starts = [w.name for w in workflows if w.push is not None and admits(w.push, ref)]
    read = f"{len(starts)} of {len(workflows)} workflow(s) in {WORKFLOWS_DIR}"
    if not starts:
        triggers = "; ".join(describe(workflow) for workflow in workflows)
        _test_hooks.emit(
            f"ci-wake: {ref} at {commit[:7]} starts no workflow, so no run will ever exist "
            f"for it: {read} admit it"
            + (f" ({triggers})" if triggers else "")
            + "; nothing to enrol"
        )
        return 0
    record = PushAttempt(
        repo=repo,
        sha=commit,
        ref=ref,
        agent=agent,
        attempted_unix=_test_hooks.now(),
    )
    append_attempt(pathlib.Path(cli_args.require_flag(parsed, ENROLMENT_FLAG)), record)
    _test_hooks.emit(
        f"ci-wake: enrolled {record['sha'][:7]} in {record['repo']} for "
        + (
            f"@{record['agent']}"
            if record["agent"] != ""
            else f"nobody (no ${LABEL_VARIABLE} and no board binding)"
        )
        + f"; {ref} starts {', '.join(starts)} ({read})"
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
