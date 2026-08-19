"""Select the source files a corpus is built from.

Selection is gitignore-aware by construction rather than by reimplementation:
the candidate universe is ``git ls-files``, so anything the repository does
not track -- virtualenvs, caches, node_modules, build output -- never enters
consideration. What remains is filtered by language extension and by two
exclusion rules that each earn their place:

*Generated files.* ``document_categories.py`` is emitted by a generator; a
style model trained on it would learn to hand-write generator output, which
is the opposite of the house discipline it encodes.

*Whitespace-only files.* An empty document trains nothing and would still
count toward the manifest's totals, so it is excluded and counted instead.

Duplicate handling is deliberately NOT here: identical files are common in
this monorepo (the per-project guard scripts are byte-identical by design),
and deduplication must run across every repository at once, so it lives in
the emission layer where the combined list exists.
"""

from __future__ import annotations

import hashlib
import pathlib
from collections.abc import Sequence

from code_corpus.contracts.corpus import RepoPin
from code_corpus.core import _test_hooks

LANGUAGE_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "python": (".py",),
}

# Generator output tracked in git. Selected by basename because the twin
# files carry the same name wherever the generator writes them.
GENERATED_BASENAMES: frozenset[str] = frozenset({"document_categories.py"})

CHARS_PER_TOKEN_ESTIMATE = 4


class SelectedFile:
    """One source file that will become a corpus document.

    Attributes:
        repo: Name of the repository the file came from.
        path: Repository-relative forward-slashed path.
        language: Language the file was selected as.
        sha256: Digest of the normalized content (UTF-8, LF endings).
        tokens_approx: Estimated tokens of the content, chars/4.
        text: Normalized content.
    """

    __slots__ = ("language", "path", "repo", "sha256", "text", "tokens_approx")

    def __init__(
        self,
        *,
        repo: str,
        path: str,
        language: str,
        sha256: str,
        tokens_approx: int,
        text: str,
    ) -> None:
        """Initialise a selected file.

        Args:
            repo: Name of the repository the file came from.
            path: Repository-relative forward-slashed path.
            language: Language the file was selected as.
            sha256: Digest of the normalized content.
            tokens_approx: Estimated tokens of the content.
            text: Normalized content.
        """
        self.repo = repo
        self.path = path
        self.language = language
        self.sha256 = sha256
        self.tokens_approx = tokens_approx
        self.text = text


class SelectionOutcome:
    """What one repository contributed, and what it was refused.

    Attributes:
        files: Selected files in git's listing order.
        excluded_generated: Tracked files refused as generator output.
        excluded_empty: Tracked files refused as whitespace-only.
    """

    __slots__ = ("excluded_empty", "excluded_generated", "files")

    def __init__(
        self,
        *,
        files: Sequence[SelectedFile],
        excluded_generated: int,
        excluded_empty: int,
    ) -> None:
        """Initialise a selection outcome.

        Args:
            files: Selected files in git's listing order.
            excluded_generated: Tracked files refused as generator output.
            excluded_empty: Tracked files refused as whitespace-only.
        """
        self.files = tuple(files)
        self.excluded_generated = excluded_generated
        self.excluded_empty = excluded_empty


def approx_tokens(text: str) -> int:
    """Estimate a token count for budgeting only.

    This is chars/4. It sizes runs against one another; it is not a
    measurement and nothing is reported from it.

    Args:
        text: Text to estimate.

    Returns:
        Estimated tokens, at least one.
    """
    return max(1, len(text) // CHARS_PER_TOKEN_ESTIMATE)


def detect_language(path: str) -> str | None:
    """Name the language a path's extension selects it as.

    Args:
        path: Repository-relative path.

    Returns:
        The language name, or None when no known language claims the
        extension.
    """
    for language in sorted(LANGUAGE_EXTENSIONS):
        if path.endswith(LANGUAGE_EXTENSIONS[language]):
            return language
    return None


def tracked_files(repo_root: pathlib.Path) -> list[str]:
    """List the repository's tracked files, in git's stable listing order.

    Args:
        repo_root: Repository to list.

    Returns:
        Repository-relative forward-slashed paths.
    """
    out = _test_hooks.run_git(repo_root, ("ls-files", "-z"))
    return [path for path in out.split("\0") if path]


def git_head(repo_root: pathlib.Path) -> str:
    """Resolve the commit the repository is checked out at.

    Args:
        repo_root: Repository to resolve.

    Returns:
        The full commit hash of HEAD.
    """
    return _test_hooks.run_git(repo_root, ("rev-parse", "HEAD")).strip()


def git_dirty(repo_root: pathlib.Path) -> bool:
    """Report whether the working tree differs from HEAD.

    A dirty emission cannot be reproduced from the recorded commit alone,
    so the condition is pinned into the manifest rather than ignored.

    Args:
        repo_root: Repository to inspect.

    Returns:
        True when tracked or untracked changes exist.
    """
    return _test_hooks.run_git(repo_root, ("status", "--porcelain")).strip() != ""


def repo_pin(name: str, repo_root: pathlib.Path) -> RepoPin:
    """Pin the state a repository is in.

    Args:
        name: Short repository name for the manifest.
        repo_root: Repository to pin.

    Returns:
        The repository's name, HEAD commit and dirtiness.
    """
    return RepoPin(name=name, commit=git_head(repo_root), dirty=git_dirty(repo_root))


def decode_source_text(repo: str, path: str, raw: bytes) -> str:
    """Decode a source file's bytes into normalized text.

    Args:
        repo: Repository name, for the error message.
        path: Repository-relative path, for the error message.
        raw: The file's bytes.

    Returns:
        The content as UTF-8 text with line endings normalized to LF, so a
        CRLF checkout and an LF checkout of the same commit digest and train
        identically.

    Raises:
        ValueError: If the bytes are not valid UTF-8. A tracked source file
            that does not decode is a repository defect to surface, not a
            record to skip.
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{repo}/{path} is not valid UTF-8: {exc}") from exc
    return text.replace("\r\n", "\n")


def _wanted_languages(languages: Sequence[str]) -> frozenset[str]:
    """Validate the requested language names.

    Args:
        languages: Language names to select for.

    Returns:
        The names as a set.

    Raises:
        ValueError: If a name is not a known language, naming the known ones.
    """
    for language in languages:
        if language not in LANGUAGE_EXTENSIONS:
            known = ", ".join(sorted(LANGUAGE_EXTENSIONS))
            raise ValueError(f"unknown language '{language}'; known languages: {known}")
    return frozenset(languages)


def select_files(
    repo_name: str, repo_root: pathlib.Path, languages: Sequence[str]
) -> SelectionOutcome:
    """Select one repository's corpus files.

    Args:
        repo_name: Short repository name carried onto every selected file.
        repo_root: Repository to select from.
        languages: Languages to select for.

    Returns:
        The selected files and the counts of what was refused.

    Raises:
        ValueError: If a language name is unknown, or a tracked source file
            is not valid UTF-8.
    """
    wanted = _wanted_languages(languages)
    files: list[SelectedFile] = []
    excluded_generated = 0
    excluded_empty = 0
    for path in tracked_files(repo_root):
        language = detect_language(path)
        if language is None or language not in wanted:
            continue
        if path.rsplit("/", 1)[-1] in GENERATED_BASENAMES:
            excluded_generated += 1
            continue
        raw = _test_hooks.read_bytes(repo_root / path)
        text = decode_source_text(repo_name, path, raw)
        if text.strip() == "":
            excluded_empty += 1
            continue
        files.append(
            SelectedFile(
                repo=repo_name,
                path=path,
                language=language,
                sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                tokens_approx=approx_tokens(text),
                text=text,
            )
        )
    return SelectionOutcome(
        files=files,
        excluded_generated=excluded_generated,
        excluded_empty=excluded_empty,
    )


__all__ = [
    "CHARS_PER_TOKEN_ESTIMATE",
    "GENERATED_BASENAMES",
    "LANGUAGE_EXTENSIONS",
    "SelectedFile",
    "SelectionOutcome",
    "approx_tokens",
    "decode_source_text",
    "detect_language",
    "git_dirty",
    "git_head",
    "repo_pin",
    "select_files",
    "tracked_files",
]
