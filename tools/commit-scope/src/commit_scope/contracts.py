"""The types this package decides over, and the validation that admits them.

Two shapes, and the split between them is the whole design. :class:`ScopeQuestion`
is what the caller supplies -- the staged set and the declaration -- and it is
DECODED, because both halves arrive as untrusted text from outside the process.
:class:`ScopeDecision` is what this package answers, and it is constructed
rather than decoded, because nothing outside builds one.

WHY A DECLARATION IS VALIDATED RATHER THAN NORMALISED INTO SUBMISSION. git
reports staged paths repo-relative with forward slashes, always. An absolute
entry, or one climbing out of the tree with ``..``, can therefore never match
a staged path -- so silently keeping it would produce a declaration that
protects nothing while reading exactly like protection. That failure mode is
the one this package exists to remove, so an unmatchable entry is refused at
the point the author can still fix it, with a code naming which kind it was.

A declaration that is merely WRONG -- naming a real path the author did not
touch -- is not an error. It cannot fail open: a superfluous entry widens the
allowed set only for paths nobody staged, and :attr:`ScopeDecision.unmatched`
reports it so a typo is visible without being fatal.
"""

from __future__ import annotations

from typing import Final, TypedDict

from platform_core.error_codes_tooling import CommitScopeErrorCode
from platform_core.errors import AppError

#: Separators accepted inside a declaration.
#:
#: Newline and comma only. A path may legitimately contain a space -- this
#: repository has none today, but a declaration that silently split
#: ``my dir/x.py`` into two never-matching entries would fail open, which is
#: the one outcome this module refuses to allow.
SCOPE_SEPARATORS: Final = ("\n", ",")


class ScopeQuestion(TypedDict):
    """What was staged, and what the author said they were staging.

    Attributes:
        staged: Repo-relative paths currently in the index, normalised.
        scope: Declared entries, normalised. Empty means the author declared
            nothing, which is a distinct state from declaring an empty set --
            see :func:`commit_scope.scope.decide`.
    """

    staged: tuple[str, ...]
    scope: tuple[str, ...]


class ScopeDecision(TypedDict):
    """Whether this commit may proceed, and why not when it may not.

    Attributes:
        declared: True when the author declared a non-empty scope.
        staged: The staged set the decision was made over.
        out_of_scope: Staged paths matching no declared entry. Non-empty only
            when ``declared`` is True.
        unmatched: Declared entries no staged path matched. Reported, never
            fatal -- a superfluous entry cannot admit anything.
    """

    declared: bool
    staged: tuple[str, ...]
    out_of_scope: tuple[str, ...]
    unmatched: tuple[str, ...]


def normalise_path(raw: str) -> str:
    """Reduce a path or declaration entry to its comparison form.

    Backslashes fold to forward slashes because git reports forward slashes on
    every platform while an author on Windows will type either, and trailing
    slashes are stripped so ``libs/platform_core`` and ``libs/platform_core/``
    are one entry rather than two.

    Args:
        raw: A path or entry as written.

    Returns:
        The comparison form, empty when the input was blank.
    """
    return raw.strip().replace("\\", "/").rstrip("/")


def require_relative_scope_entry(entry: str) -> str:
    """Admit one declaration entry, or refuse it by kind.

    Args:
        entry: A normalised entry.

    Returns:
        The entry, unchanged.

    Raises:
        AppError: ``SCOPE_ENTRY_NOT_RELATIVE`` when the entry is absolute --
            POSIX-rooted or carrying a Windows drive letter -- and
            ``SCOPE_ENTRY_ESCAPES_REPO`` when any segment is ``..``. Both can
            never match a repo-relative staged path, so both would declare a
            protection that silently covers nothing.
    """
    if entry.startswith("/") or (len(entry) > 1 and entry[1] == ":"):
        raise AppError(
            code=CommitScopeErrorCode.SCOPE_ENTRY_NOT_RELATIVE,
            message=(
                f"Declared scope entry {entry!r} is an absolute path. git reports "
                "staged paths relative to the repository root, so this entry can "
                "never match one and would allow nothing while appearing to."
            ),
        )
    if ".." in entry.split("/"):
        raise AppError(
            code=CommitScopeErrorCode.SCOPE_ENTRY_ESCAPES_REPO,
            message=(
                f"Declared scope entry {entry!r} climbs out of the repository. A "
                "staged path is always inside it, so this entry can never match "
                "one and would allow nothing while appearing to."
            ),
        )
    return entry


def decode_scope_declaration(raw: str | None) -> tuple[str, ...]:
    """Turn a raw declaration into validated entries.

    Args:
        raw: The declaration as supplied, or None when the author made none.

    Returns:
        Normalised, validated entries in declaration order. Empty when the
        declaration was absent, blank, or only separators -- all three mean
        "declared nothing", never "declared an empty set".

    Raises:
        AppError: Propagated from :func:`require_relative_scope_entry` for the
            first unmatchable entry.
    """
    if raw is None:
        return ()
    entries = [raw]
    for separator in SCOPE_SEPARATORS:
        entries = [piece for entry in entries for piece in entry.split(separator)]
    return tuple(
        require_relative_scope_entry(normalised)
        for normalised in (normalise_path(entry) for entry in entries)
        if normalised
    )


def decode_staged_paths(raw: str) -> tuple[str, ...]:
    """Turn ``git diff --cached --name-only`` output into paths.

    Args:
        raw: The command's whole stdout.

    Returns:
        One entry per non-blank line, normalised. An empty index yields an
        empty tuple, which is a legitimate state rather than a failure -- a
        hook may run when another session has just emptied the index.
    """
    return tuple(
        normalised
        for normalised in (normalise_path(line) for line in raw.split("\n"))
        if normalised
    )


def encode_scope_decision(decision: ScopeDecision) -> dict[str, bool | list[str]]:
    """Render a decision as plain JSON-ready values.

    Exists so a decision can be logged or asserted structurally rather than by
    scraping the human report, which is prose and free to change.

    Args:
        decision: The decision to encode.

    Returns:
        A mapping with ``declared`` as a bool and the three path sets as
        lists, in the order the decision holds them.
    """
    return {
        "declared": decision["declared"],
        "staged": list(decision["staged"]),
        "out_of_scope": list(decision["out_of_scope"]),
        "unmatched": list(decision["unmatched"]),
    }
