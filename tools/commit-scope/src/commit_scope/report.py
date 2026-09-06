"""What the author reads at the moment their commit is stopped, or is not.

Separated from IO so the exact wording is asserted by tests rather than
eyeballed once in a terminal. The wording matters more than usual here: the
reader is mid-commit, and the obvious way to make a refusal go away -- ``git
add`` the named paths -- is the one action that converts a near miss into the
exact incident this package exists to prevent. So the refusal says not to,
explicitly, above the two things that actually resolve it.
"""

from __future__ import annotations

from commit_scope.contracts import ScopeDecision
from commit_scope.scope import refuses


def _receipt(decision: ScopeDecision) -> tuple[str, ...]:
    """Render the undeclared case.

    Args:
        decision: A decision whose ``declared`` is False.

    Returns:
        The staged set, plus how to have it enforced next time.
    """
    return (
        f"=== commit-scope: staging receipt ({len(decision['staged'])} path(s)) ===",
        *(f"    {path}" for path in decision["staged"]),
        "Nothing was declared, so nothing is enforced. The index is shared, so",
        "this list is what will ship under your message. Declare",
        "COMMIT_SCOPE to have it checked, or use `git commit -- <paths>`,",
        "which git honours natively.",
    )


def _accepted(decision: ScopeDecision) -> tuple[str, ...]:
    """Render a declared commit that may proceed.

    Args:
        decision: A declared decision with nothing out of scope.

    Returns:
        A one-line confirmation, plus any declared entry that matched nothing.
    """
    header = f"=== commit-scope: staged scope OK ({len(decision['staged'])} path(s)) ==="
    if not decision["unmatched"]:
        return (header,)
    return (
        header,
        "Declared but nothing staged matched (likely a typo, not fatal):",
        *(f"    {entry}" for entry in decision["unmatched"]),
    )


def _refusal(decision: ScopeDecision) -> tuple[str, ...]:
    """Render a commit that must be stopped.

    Args:
        decision: A declared decision carrying out-of-scope paths.

    Returns:
        The intruding paths, the mechanism, and the two resolutions -- with
        the tempting wrong one named first so it is not discovered instead.
    """
    return (
        "=== COMMIT BLOCKED: the index carries paths you did not declare ===",
        "",
        "Staged but outside COMMIT_SCOPE:",
        *(f"    {path}" for path in decision["out_of_scope"]),
        "",
        "This is the shared-index sweep. `git add` protects the add, not the",
        "commit, and `git commit` takes the WHOLE index. Another session very",
        "likely staged these between your add and your commit; committing now",
        "ships their work under your message and your authorship.",
        "",
        "Do NOT `git add` them to make this pass. Either:",
        "  - commit only yours:  git commit -- <your paths>",
        "  - or widen COMMIT_SCOPE if they really are yours.",
        "Never --amend in a shared tree: amend takes the index too, and --only",
        "does not help because it rebases onto a commit that already swept.",
    )


def render(decision: ScopeDecision) -> tuple[str, ...]:
    """Render any decision as the lines a hook should print.

    Args:
        decision: The decision to report.

    Returns:
        Lines to write, in order, without trailing newlines.
    """
    if not decision["declared"]:
        return _receipt(decision)
    if refuses(decision):
        return _refusal(decision)
    return _accepted(decision)
