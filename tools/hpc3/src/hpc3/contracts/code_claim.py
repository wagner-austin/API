"""A run's claim about which code it executes, in a form that can be checked.

An experiment record is free-form (see :mod:`hpc3.contracts.experiment`), and
until 2026-10-05 ``repo_commit`` was one more free-form pair: six cleargbm
sweep documents declared it and nothing read it. Five were right. The sixth,
``sweep-cleargbm-p6-rung5.json``, declared ``20d9159`` -- the WORKSTATION's
HEAD at the moment of submission -- while every member ran from the cluster
checkout ``/pub/wagnera3/api``, whose last reflog entry put it at ``80221ea``
two hours earlier and which had never fetched ``20d9159`` at all. A declared
commit right four times out of five is worse than none: a reader who spot
checks the first finds it right and trusts the sixth.

THE FIX IS TO MAKE THE CLAIM CHECKABLE, AND CHECKED. A commit alone is not
checkable, because it does not say WHICH checkout it describes, and the
submitter's tree and the cluster's tree are different trees -- that confusion
is the whole defect. So a run that declares ``repo_commit`` must also declare
``repo_tree``, the absolute cluster path of the checkout its payload runs
from, and :func:`hpc3.core.code_claim.check_code_claim` asks that checkout,
at submission, whether its HEAD is the declared commit.

Both keys stay in the experiment record rather than becoming spec fields.
They are part of what identifies the run, so they belong in the ledger row
beside the job id, and the record already carries them there.

WHAT THIS DOES NOT COVER. For an imaged run the image's own wheels are code
the checkout does not describe; their commit is the image spec's
``git_commit``, and verifying that is a separate claim with a separate owner.
``repo_tree`` names what the payload reads from the cluster filesystem -- for
the cleargbm sweeps, ``scripts.optimize`` under the checkout -- and nothing
more.
"""

from __future__ import annotations

import re

from platform_core.json_utils import JSONTypeError
from typing_extensions import TypedDict

REPO_COMMIT_KEY = "repo_commit"
"""The experiment key naming the commit the payload's checkout is at."""

REPO_TREE_KEY = "repo_tree"
"""The experiment key naming the cluster checkout the payload runs from."""

_COMMIT = re.compile(r"[0-9a-f]{7,40}")
"""A commit as git abbreviates or spells it: seven to forty lowercase hex.

Seven is git's own minimum abbreviation. A shorter prefix is refused here
rather than left for the checkout to call ambiguous, because a declaration
that cannot name one commit in a large history is not naming one.
"""


class CodeClaim(TypedDict):
    """What a run says about the code it executes.

    Attributes:
        tree: Absolute cluster path of the git checkout the payload runs
            from.
        commit: The commit the run says that checkout's HEAD is, as written
            -- abbreviated or full.
    """

    tree: str
    commit: str


def code_claim(experiment: dict[str, str]) -> CodeClaim | None:
    """Read a run's code claim out of its experiment record.

    Args:
        experiment: The run's identity pairs, already validated by
            :func:`~hpc3.contracts.experiment.require_experiment`.

    Returns:
        The claim, or None when the record declares neither key. A run that
        names no commit is making no claim to check; the ledger records that
        it named none, which is true.

    Raises:
        JSONTypeError: If the record declares one key without the other, if
            ``repo_tree`` is not an absolute path, or if ``repo_commit`` is
            not seven to forty lowercase hex characters. A commit with no
            tree cannot be checked against anything, and a commit that was
            never checked is the defect this module exists to end.
    """
    commit = experiment.get(REPO_COMMIT_KEY)
    tree = experiment.get(REPO_TREE_KEY)
    if commit is None and tree is None:
        return None
    if commit is None or tree is None:
        missing = REPO_COMMIT_KEY if commit is None else REPO_TREE_KEY
        present = REPO_TREE_KEY if commit is None else REPO_COMMIT_KEY
        raise JSONTypeError(
            f"experiment declares {present!r} without {missing!r}. A commit is only "
            "checkable against the checkout it describes, and the submitter's tree "
            "is not the cluster's: name both, or neither."
        )
    if not tree.startswith("/"):
        raise JSONTypeError(
            f"experiment {REPO_TREE_KEY!r} must be an absolute cluster path, got {tree!r}"
        )
    if _COMMIT.fullmatch(commit) is None:
        raise JSONTypeError(
            f"experiment {REPO_COMMIT_KEY!r} must be 7 to 40 lowercase hex characters, "
            f"got {commit!r}"
        )
    return CodeClaim(tree=tree, commit=commit)


__all__ = ["REPO_COMMIT_KEY", "REPO_TREE_KEY", "CodeClaim", "code_claim"]
