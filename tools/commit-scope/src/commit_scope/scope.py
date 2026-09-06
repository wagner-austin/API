"""The decision: does this commit carry only what its author declared?

Pure. No git, no environment, no output -- given a :class:`ScopeQuestion` it
returns a :class:`ScopeDecision`, and every interesting property of this
package is testable here without a repository.

THE SEPARATOR IN :func:`covers` IS LOAD-BEARING. A declared directory covers a
path only as an exact match or followed by ``/``. Without that, declaring
``libs/platform_core`` would silently cover ``libs/platform_core_extras`` --
a sibling package, and exactly the kind of neighbour a sweep picks up. The
prefix test is the one place where being slightly too permissive reintroduces
the whole defect.
"""

from __future__ import annotations

from commit_scope.contracts import ScopeDecision, ScopeQuestion


def covers(path: str, entry: str) -> bool:
    """Does one declared entry cover one staged path?

    Args:
        path: A normalised staged path.
        entry: A normalised declared entry.

    Returns:
        True when the entry is the path itself or a directory containing it.
    """
    return path == entry or path.startswith(f"{entry}/")


def decide(question: ScopeQuestion) -> ScopeDecision:
    """Compare a staged set against a declaration.

    An absent declaration is NOT an empty declaration. With nothing declared
    there is no statement of intent to compare against, so no path can be out
    of scope and the decision carries ``declared: False`` for the caller to
    act on. Treating it as an empty allow-list would refuse every commit in
    the repository, and a check people must disable protects nothing.

    Args:
        question: The staged set and the declared scope.

    Returns:
        The decision, carrying the staged set it was made over so a caller can
        report exactly what it judged.
    """
    if not question["scope"]:
        return {
            "declared": False,
            "staged": question["staged"],
            "out_of_scope": (),
            "unmatched": (),
        }

    return {
        "declared": True,
        "staged": question["staged"],
        "out_of_scope": tuple(
            path
            for path in question["staged"]
            if not any(covers(path, entry) for entry in question["scope"])
        ),
        "unmatched": tuple(
            entry
            for entry in question["scope"]
            if not any(covers(path, entry) for path in question["staged"])
        ),
    }


def refuses(decision: ScopeDecision) -> bool:
    """Must the commit be stopped?

    Args:
        decision: The decision to read.

    Returns:
        True only when a declaration was made AND the index carries paths
        outside it. An undeclared commit never refuses; an unmatched
        declaration entry never refuses, because a superfluous entry cannot
        admit anything.
    """
    return decision["declared"] and bool(decision["out_of_scope"])
