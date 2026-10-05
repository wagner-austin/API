"""Asking the cluster checkout whether a run's declared commit is the code it runs.

:mod:`hpc3.contracts.code_claim` says what a run claims; this module checks
the claim at the one moment it can be checked, submission, against the one
tree it describes. A committed run document cannot be checked in CI: the
checkout lives on the cluster, CI cannot see it, and it moves with every
``git pull`` there, so a document's claim is true or false only relative to
the instant it is submitted. Preflight is that instant.

THE GATE IS EQUALITY, NOT ANCESTRY. A run executes the checkout's HEAD, so
the only true declaration is HEAD itself. ``git merge-base --is-ancestor``
would also admit a declared commit the checkout has since moved past --
code that was there once and is not what runs now. Ancestry is asked anyway,
for the refusal's MESSAGE: it says which way the declaration is wrong, and
the two ways have different repairs.

THREE OUTCOMES, NEVER TWO. ``--is-ancestor`` exits 0 for yes, 1 for no, and
128 when it could not answer; a check that folded every non-zero exit into
"not reachable" would convict on a question it never answered (measured
2026-09-10 on board task 2cca4a98, which commissioned this module). Here an
unanswered relation does not decide anything: the refusal rests on the two
resolved commits being different, which git DID answer, and the message says
the direction is unknown rather than inventing one.

A DECLARED COMMIT THE CHECKOUT CANNOT RESOLVE is a refusal too, and it is
exactly rung 5's case: ``20d9159`` resolves to nothing in ``/pub/wagnera3/api``
because that clone never fetched it. A clone that has not fetched a commit is
not running it.
"""

from __future__ import annotations

import shlex

from platform_core.errors import AppError, Hpc3ErrorCode
from typing_extensions import TypedDict

from hpc3.contracts.code_claim import CodeClaim, code_claim
from hpc3.core import remote

_HEAD = "head="
_DECLARED = " declared="
_ANCESTOR = " ancestor="


class CheckoutReading(TypedDict):
    """What the checkout said about itself and the declared commit.

    Attributes:
        head: The checkout's HEAD, as a full sha.
        declared: The declared commit resolved to a full sha, or empty when
            the checkout holds no single commit by that name.
        ancestor: ``git merge-base --is-ancestor``'s exit status for the
            declared commit against HEAD, as text: ``"0"`` (an ancestor),
            ``"1"`` (not one), any other status (it could not answer), or
            empty when there was no resolved commit to ask about.
    """

    head: str
    declared: str
    ancestor: str


def probe_command(claim: CodeClaim) -> str:
    """Build the one remote command that reads the checkout against the claim.

    Args:
        claim: The run's validated claim.

    Returns:
        A shell command that exits non-zero only when the checkout itself is
        unusable -- absent, or not a git repository with a HEAD -- and
        otherwise prints one ``head=<sha> declared=<sha> ancestor=<status>``
        line. ``--verify --quiet`` makes an unresolvable or ambiguous name an
        empty ``declared`` rather than a failure, because a name the checkout
        cannot resolve is an answer about the claim, not a broken probe.
    """
    tree = shlex.quote(claim["tree"])
    commit = shlex.quote(f"{claim['commit']}^{{commit}}")
    return (
        f"cd {tree} && head=$(git rev-parse --verify HEAD) && "
        f"if declared=$(git rev-parse --verify --quiet {commit}); then "
        'git merge-base --is-ancestor "$declared" "$head"; '
        'echo "head=$head declared=$declared ancestor=$?"; '
        'else echo "head=$head declared= ancestor="; fi'
    )


def parse_probe(output: str, claim: CodeClaim) -> CheckoutReading:
    """Read the probe's one line.

    Args:
        output: The probe's standard output.
        claim: The claim being checked, named in the error.

    Returns:
        The checkout's reading.

    Raises:
        AppError: With ``REPO_COMMIT_PROBE_UNREADABLE`` when no line has the
            probe's shape, or HEAD came back empty. A login banner or a shell
            that mangled the line lands here and is never read as a verdict.
    """
    for line in output.splitlines():
        if not line.startswith(_HEAD):
            continue
        head, found_declared, rest = line[len(_HEAD) :].partition(_DECLARED)
        declared, found_ancestor, ancestor = rest.partition(_ANCESTOR)
        if found_declared == "" or found_ancestor == "" or head == "":
            break
        return CheckoutReading(head=head, declared=declared, ancestor=ancestor.strip())
    raise AppError(
        Hpc3ErrorCode.REPO_COMMIT_PROBE_UNREADABLE,
        f"Asking {claim['tree']} about {claim['commit']} printed no "
        f"'head=<sha> declared=<sha> ancestor=<status>' line; got {output.strip()!r}.",
    )


def explain_mismatch(claim: CodeClaim, reading: CheckoutReading) -> str:
    """Say which way a declaration is wrong, as far as git could tell.

    Args:
        claim: The run's claim.
        reading: The checkout's reading, whose HEAD is not the declared
            commit.

    Returns:
        The refusal's message, naming the tree, both commits and the
        direction -- or saying plainly that the direction is unknown.
    """
    opening = (
        f"This run declares repo_commit {claim['commit']} for {claim['tree']}, "
        f"but that checkout is at {reading['head']}, and that is the code the "
        "payload would run."
    )
    if reading["declared"] == "":
        detail = (
            f" The checkout resolves no single commit named {claim['commit']}: it has "
            "never fetched it, or the name is mistyped or ambiguous. A clone that has "
            "not fetched a commit is not running it -- this is how cleargbm P6 rung 5 "
            "declared the submitter's 20d9159 while running 80221ea."
        )
    elif reading["ancestor"] == "0":
        detail = (
            " The declared commit is in the checkout's history, so the checkout has "
            "moved past it since the document was written."
        )
    elif reading["ancestor"] == "1":
        detail = (
            f" The declared commit {reading['declared']} is present but NOT in the "
            "checkout's history: it names code the checkout does not contain."
        )
    else:
        detail = (
            f" Which way they differ is unknown: git merge-base --is-ancestor exited "
            f"{reading['ancestor']}, which is git declining to answer, not a 'no'."
        )
    return (
        opening
        + detail
        + " Declare the commit the checkout is at, or move the checkout to the one "
        "declared, then submit again."
    )


def check_code_claim(host: str, experiment: dict[str, str]) -> str | None:
    """Refuse a run whose declared commit is not the checkout's HEAD.

    Args:
        host: SSH destination.
        experiment: The run's identity record.

    Returns:
        The checkout's HEAD as a full sha when the claim holds, or None when
        the run makes no claim -- in which case nothing is asked of the
        cluster at all.

    Raises:
        AppError: With ``REPO_COMMIT_NOT_HEAD`` when the checkout is at
            another commit, ``REPO_COMMIT_PROBE_UNREADABLE`` when its answer
            cannot be read, or ``REMOTE_COMMAND_FAILED`` when the checkout is
            absent or not a git repository.
        JSONTypeError: If the record carries half a claim or a malformed one;
            unreachable for a decoded spec, which was refused at decode.
    """
    claim = code_claim(experiment)
    if claim is None:
        return None
    reading = parse_probe(remote.run_remote(host, probe_command(claim)), claim)
    if reading["declared"] == reading["head"]:
        return reading["head"]
    raise AppError(Hpc3ErrorCode.REPO_COMMIT_NOT_HEAD, explain_mismatch(claim, reading))


__all__ = [
    "CheckoutReading",
    "check_code_claim",
    "explain_mismatch",
    "parse_probe",
    "probe_command",
]
