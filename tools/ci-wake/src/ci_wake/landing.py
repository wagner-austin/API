"""Whether a push that produced no run ever reached its branch.

WHY THIS EXISTS (MCPs board task be748588). The pre-push hooks enrol a push
before its checks and before git sends anything, by design, so a push that
is then refused leaves a row, and a row with no run past
:data:`ci_wake.verdicts.NO_RUN_SECONDS` was announced NO RUN EVER APPEARED.
:mod:`ci_wake.verdicts` itself names "a push git refused" as the ordinary
cause. On 2026-10-16 three such posts on MCPs main were each a
worktree-publish attempt whose retry landed: d357385da52f, which pre-push
refused and GitHub never received, and 1091c36a200b and 868ae9417, which
the remote refused because main had moved after GitHub had stored their
objects. A push that never landed has nothing to report, and announcing it
the same way as a landed sha with no run teaches the reader to skip the one
post that means Actions ran nothing.

TWO QUESTIONS, BECAUSE ONE CANNOT ANSWER BOTH. GitHub's compare answers 404
for a sha it never received, which ``gh`` reports as a failure like any
other, and telling that 404 apart from a logged-out ``gh`` would mean
reading its words. GraphQL's ``repository.object(expression:)`` answers
``null`` for that sha without an error, so it is asked first; a sha GitHub
holds is then compared with the branch, where ``ahead`` or ``identical``
(the branch descends from the sha) means the branch holds it, and
``diverged`` or ``behind`` that it does not. Every ``gh`` failure still
propagates, as everywhere in this package.
"""

from __future__ import annotations

from typing import Final, NamedTuple

from platform_core.json_utils import JSONTypeError, JSONValue, narrow_json_to_dict

from ci_wake.coverage import DESCENDS, compare_argv, decode_compare_status
from ci_wake.enrolment import PushAttempt
from ci_wake.runs import gh_json

#: The GraphQL query naming one object of a repository, or ``null``.
OBJECT_QUERY: Final = (
    "query($owner:String!,$name:String!,$sha:String!)"
    "{repository(owner:$owner,name:$name){object(expression:$sha){__typename}}}"
)

#: What :func:`landing_of` answers for a sha GitHub never received.
ABSENT: Final = "absent"

#: The state an :class:`UnlandedPush` is recorded under in the position file.
UNLANDED: Final = "unlanded"


def object_argv(repo: str, sha: str) -> tuple[str, ...]:
    """Build the command asking whether GitHub holds an object.

    Args:
        repo: ``owner/name``.
        sha: The pushed sha.

    Returns:
        The argument vector, its ``--jq`` keeping only the object under the
        key ``object``, which is ``null`` for a sha GitHub never received.
        Wrapped because ``gh``'s ``--jq`` prints nothing at all for a bare
        ``null`` (measured 2026-10-16), and nothing is not JSON.
    """
    owner, name = repo.split("/", 1)
    return (
        "gh",
        "api",
        "graphql",
        "-f",
        f"query={OBJECT_QUERY}",
        "-f",
        f"owner={owner}",
        "-f",
        f"name={name}",
        "-f",
        f"sha={sha}",
        "--jq",
        "{object: .data.repository.object}",
    )


def decode_held(payload: JSONValue) -> bool:
    """Read :func:`object_argv`'s reply.

    Args:
        payload: ``{"object": null}``, or ``{"object": {"__typename": ...}}``.

    Returns:
        False for a ``null`` object, True for an object.

    Raises:
        JSONTypeError: For a reply that is not an object, that carries no
            ``object`` key (absent must be said, never inferred), or whose
            ``object`` is neither ``null`` nor an object.
    """
    reply = narrow_json_to_dict(payload)
    if "object" not in reply:
        raise JSONTypeError(f"GraphQL object reply carries no 'object' key: {reply!r}")
    held = reply["object"]
    if held is None:
        return False
    narrow_json_to_dict(held)
    return True


def landing_of(repo: str, branch: str, sha: str) -> str:
    """Say where a pushed sha stands against its branch on GitHub.

    Args:
        repo: ``owner/name``.
        branch: The branch the push updated.
        sha: The pushed sha.

    Returns:
        :data:`ABSENT` when GitHub never received the sha, otherwise the
        compare status of the branch against it: ``ahead`` or ``identical``
        when the branch holds it, ``behind`` or ``diverged`` when not.

    Raises:
        AppError: ``GH_COMMAND_FAILED`` when ``gh`` refused either call.
        JSONTypeError: When a reply is not the shape this reads.
    """
    if not decode_held(gh_json(object_argv(repo, sha))):
        return ABSENT
    return decode_compare_status(gh_json(compare_argv(repo, sha, branch)))


class UnlandedPush(NamedTuple):
    """An enrolled push whose branch never held its sha, closed unannounced.

    Attributes:
        attempt: The enrolment row.
        branch: The branch it was pushed to.
        landing: What :func:`landing_of` answered: :data:`ABSENT`,
            ``behind`` or ``diverged``.
    """

    attempt: PushAttempt
    branch: str
    landing: str

    def line(self) -> str:
        """Name it on the cycle's output, with the reason GitHub gave.

        Returns:
            One line.
        """
        reason = (
            "GitHub never received it"
            if self.landing == ABSENT
            else f"compare of {self.branch} against it reads {self.landing}"
        )
        return (
            f"unlanded {self.attempt['repo']}@{self.attempt['sha'][:12]}: "
            f"{self.attempt['ref']} never held it ({reason}), so no run was due; "
            "closed unannounced"
        )


def holds(landing: str) -> bool:
    """Does a :func:`landing_of` answer mean the branch holds the sha?

    Args:
        landing: The answer.

    Returns:
        True for ``ahead`` and ``identical``.
    """
    return landing in DESCENDS


__all__ = [
    "ABSENT",
    "OBJECT_QUERY",
    "UNLANDED",
    "UnlandedPush",
    "decode_held",
    "holds",
    "landing_of",
    "object_argv",
]
