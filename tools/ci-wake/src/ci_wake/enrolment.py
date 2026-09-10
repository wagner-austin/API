"""Who pushed what, recorded at the one instant anybody knows.

THIS FILE IS THE SUBSCRIPTION. A GitHub workflow run carries a head sha, a
workflow name, a conclusion and a commit author -- and on this machine the
commit author is one person for every one of six concurrent AI sessions. No
field anywhere in the Actions API says which session typed ``git push``, and
no later query can recover it. The ``pre-push`` hook is the only code that
runs while that fact is still true, so the hook writes it down here and the
bridge reads it back.

WHICH IS WHY ENROLMENT IS TAKEN RATHER THAN OFFERED. A subscription anyone
can forget is a subscription most people forget, and the workspace has the
receipts: a workflow left red for weeks "with nothing watching", and two
pushes of a barrel importing an untracked module that survived for hours
because nothing was attempting the clean-checkout build. Making the record a
side effect of pushing removes the step that was being skipped.

AN ATTEMPT, NOT A PUSH, AND THE NAME IS DELIBERATE. ``pre-push`` runs BEFORE
the transfer, and git offers no hook that runs after one, so a row can name
a sha that a non-fast-forward rejection stopped from ever reaching the
remote. Such a row is inert -- no run will ever exist for it -- and
:mod:`ci_wake.verdicts` closes it out by age rather than leaving it queried
forever. Calling the record a push would be a claim this code cannot make.

WHAT IT DELIBERATELY DOES NOT COVER, stated here rather than discovered
later: a push made with ``--no-verify``, from another machine, or through
the GitHub web UI writes no row and is never announced. The bridge is
enrolment-driven by design -- it asks GitHub only about shas somebody
enrolled -- because the alternative is listing every recent run in every
repository on every cycle to find the handful anyone is waiting on. The
exclusion is real and it is bounded: on this machine the hook runs on every
push that is not deliberately bypassed.

FIELDS ARE VALIDATED HERE, AT THE PUSH, AND NOT AT THE ANNOUNCEMENT. A
malformed repository or sha addresses nothing, and a malformed agent label
is refused by the board itself (``assertSessionLabel``, mig 415) -- so a bad
row enrolled now becomes a post the board rejects on every subsequent cycle,
wedging every OTHER session's announcement behind it. Refusing at enrolment
turns a permanent bridge outage into one failed push, told to the one person
who can fix it, at the moment they can still fix it.
"""

from __future__ import annotations

import pathlib
import re
from collections.abc import Sequence
from typing import Final

from platform_core.error_codes_tooling import CiWakeErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    dump_json_str,
    load_json_str,
    require_int,
    require_str,
)
from typing_extensions import TypedDict

from ci_wake import _test_hooks

#: ``owner/name``, GitHub's own character set for both halves.
_REPOSITORY = re.compile(r"^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$")

#: A full commit sha. Abbreviations are refused: ``head_sha`` in the Actions
#: API is always the full 40, so a short one would match no run and the
#: bridge would report "no run ever appeared" for a push that had one.
_SHA = re.compile(r"^[0-9a-f]{40}$")

#: The board's own agent-label rule, restated so a bad label costs one push
#: instead of every later cycle. See this module's docstring.
_AGENT = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")

#: Shortest and longest agent label the board accepts.
_AGENT_LENGTH: Final = (3, 64)

#: Environment variable a session exports to be addressable in announcements.
#:
#: The SAME variable ``hpc3`` reads when recording a submitter, so a session
#: exports one name once and is reachable from both bridges. A second
#: variable for the second bridge would be a second thing to remember, and
#: the one that was forgotten would look exactly like a session that had
#: opted out.
AGENT_VARIABLE: Final = "BOARD_AGENT_LABEL"


class PushAttempt(TypedDict):
    """One ``git push`` observed by a ``pre-push`` hook.

    Attributes:
        repo: ``owner/name`` as GitHub spells it -- the address the Actions
            API is queried under, not a local remote alias, because two
            machines can name one remote differently and the API cannot.
        sha: The full 40-character commit sha the push would land.
        ref: The ref being pushed, e.g. ``refs/heads/main``. Recorded but
            never queried on: a run is found by head sha, and the ref is
            here so a person reading the file can see what a stray row was.
        agent: The pushing session's board label, or the empty string when
            it exported none. Empty is a first-class case, not a defect: a
            human pushing from a terminal has no board label, and the
            announcement for their push is posted board-level rather than
            addressed to nobody.
        attempted_unix: When the hook ran, whole seconds since the epoch.
            The clock the abandonment horizon in :mod:`ci_wake.verdicts` is
            measured from.
    """

    repo: str
    sha: str
    ref: str
    agent: str
    attempted_unix: int


def attempt_key(repo: str, sha: str) -> str:
    """The identity of one enrolled push, shared by both records.

    Args:
        repo: ``owner/name``.
        sha: The full commit sha.

    Returns:
        The key. A pair rather than the sha alone because the same commit
        can exist in two repositories -- a fork, a mirror -- and announcing
        one repository's verdict under the other's name would be worse than
        announcing nothing.
    """
    return f"{repo}@{sha}"


def _refuse(field: str, value: str, expectation: str) -> AppError[CiWakeErrorCode]:
    """Build the refusal for a field that cannot address anything.

    Args:
        field: The field's name, as the flag spells it.
        value: What was supplied.
        expectation: What the field must look like, in words.

    Returns:
        The error to raise, naming all three. The value is quoted because
        the usual cause is a shell that expanded to nothing, and an empty
        string is invisible in a message that does not quote it.
    """
    return AppError(
        code=CiWakeErrorCode.ENROLMENT_FIELD_MALFORMED,
        message=(
            f"{field} {value!r} is not {expectation}; enrolling it would record a push "
            "this bridge could never announce, and the refusal is here rather than "
            "three minutes later because only the pushing session can fix it"
        ),
    )


def require_repository(value: str) -> str:
    """Validate a repository address.

    Args:
        value: The candidate, as the ``--repo`` flag supplied it.

    Returns:
        The value unchanged.

    Raises:
        AppError: ``ENROLMENT_FIELD_MALFORMED`` when it is not ``owner/name``.
    """
    if _REPOSITORY.match(value) is None:
        raise _refuse("--repo", value, "an owner/name repository address")
    return value


def require_sha(value: str) -> str:
    """Validate a commit sha.

    Args:
        value: The candidate, as the ``--sha`` flag supplied it.

    Returns:
        The value unchanged.

    Raises:
        AppError: ``ENROLMENT_FIELD_MALFORMED`` when it is not 40 lowercase
            hex characters.
    """
    if _SHA.match(value) is None:
        raise _refuse("--sha", value, "a full 40-character lowercase commit sha")
    return value


def require_agent(value: str) -> str:
    """Validate a board label, accepting the empty string as "unaddressed".

    Args:
        value: The label the pushing session exported, or the empty string
            when it exported none.

    Returns:
        The value unchanged.

    Raises:
        AppError: ``ENROLMENT_FIELD_MALFORMED`` when a NON-empty label is not
            kebab-case within the board's length bounds. Empty is allowed
            and means the push is announced board-level; a malformed label
            is not, because the board would refuse the post carrying it and
            every other session's announcement in that cycle with it.
    """
    if value == "":
        return value
    low, high = _AGENT_LENGTH
    if _AGENT.match(value) is None or not low <= len(value) <= high:
        raise _refuse(
            f"${AGENT_VARIABLE}",
            value,
            f"a kebab-case board label of {low}-{high} characters",
        )
    return value


def encode_push_attempt(record: PushAttempt) -> JSONObject:
    """Encode one enrolment row.

    Args:
        record: The row to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "repo": record["repo"],
        "sha": record["sha"],
        "ref": record["ref"],
        "agent": record["agent"],
        "attempted_unix": record["attempted_unix"],
    }


def decode_push_attempt(value: JSONValue) -> PushAttempt:
    """Decode and validate one enrolment row.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated row.

    Raises:
        JSONTypeError: If the value is not an object, or a field is missing
            or mistyped.
        AppError: ``ENROLMENT_FIELD_MALFORMED`` if a field is present and
            well-typed but cannot address anything. Re-validated on the way
            IN as well as on the way out, because this file is written by a
            git hook and hand-edited by whoever is debugging one, and a row
            that only the writer validated is a row nobody validated.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"enrolment row must be a JSON object, got {type(value).__name__}")
    return PushAttempt(
        repo=require_repository(require_str(value, "repo")),
        sha=require_sha(require_str(value, "sha")),
        ref=require_str(value, "ref"),
        agent=require_agent(require_str(value, "agent")),
        attempted_unix=require_int(value, "attempted_unix"),
    )


def read_attempts(path: pathlib.Path) -> tuple[PushAttempt, ...]:
    """Read every enrolled push, in the order the hooks wrote them.

    Args:
        path: The enrolment record's path.

    Returns:
        Every row. An absent file reads as empty rather than raising: a
        machine that has not pushed since the bridge was installed has
        enrolled nothing, and refusing the first cycle for having no history
        would make the bridge impossible to start.

    Raises:
        InvalidJsonError: If a line is not valid JSON at all.
        JSONTypeError: If a line is valid JSON but not a row.
        AppError: ``ENROLMENT_FIELD_MALFORMED`` from the decoder.
    """
    if not _test_hooks.file_exists(path):
        return ()
    rows: list[PushAttempt] = []
    for index, line in enumerate(_test_hooks.read_text(path).splitlines(), start=1):
        if line.strip() == "":
            continue
        value = load_json_str(line)
        if not isinstance(value, dict):
            raise JSONTypeError(
                f"{path} line {index} is a {type(value).__name__}, not an object; a "
                "line that cannot be read is a push whose author is never told what "
                "their CI did, which is the silence this bridge exists to remove"
            )
        rows.append(decode_push_attempt(value))
    return tuple(rows)


def latest_attempts(rows: Sequence[PushAttempt]) -> tuple[PushAttempt, ...]:
    """Collapse repeated enrolments of one sha to the most recent.

    THE LAST ROW WINS, and the case it exists for is ordinary: a push
    refused as non-fast-forward is enrolled, rebased, and pushed again --
    sometimes by a different session, since the tree is shared. The session
    waiting on the verdict is the one that pushed most recently, so that is
    the label the announcement carries.

    Args:
        rows: Every enrolled push, in file order.

    Returns:
        One row per ``(repo, sha)``, carrying the newest row's fields, in
        FIRST-appearance order. First-appearance rather than last so a
        cycle's output does not reorder itself when an old sha is re-pushed,
        which would make two cycles over the same work post different things.
    """
    collapsed: dict[str, PushAttempt] = {}
    for row in rows:
        collapsed[attempt_key(row["repo"], row["sha"])] = row
    return tuple(collapsed.values())


def append_attempt(path: pathlib.Path, record: PushAttempt) -> None:
    """Append one enrolment row.

    Args:
        path: The enrolment record's path.
        record: The push just observed by a hook.
    """
    _test_hooks.append_text(path, dump_json_str(encode_push_attempt(record)))


__all__ = [
    "AGENT_VARIABLE",
    "PushAttempt",
    "append_attempt",
    "attempt_key",
    "decode_push_attempt",
    "encode_push_attempt",
    "latest_attempts",
    "read_attempts",
    "require_agent",
    "require_repository",
    "require_sha",
]
