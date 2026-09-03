"""Tests for file selection.

Git is faked at the ``run_git`` seam with canned CLI output, so the parsing
of what git actually prints is exercised by the real code; the integration
tests in ``test_cli`` run the production hook against real repositories.
"""

from __future__ import annotations

import hashlib
import pathlib
import re
from collections.abc import Sequence

import pytest

from code_corpus.contracts.corpus import RepoPin
from code_corpus.core import _test_hooks as core_hooks
from code_corpus.core.select import (
    approx_tokens,
    decode_source_text,
    detect_language,
    git_dirty,
    git_head,
    repo_pin,
    require_reproducible_pins,
    select_files,
    tracked_files,
)

HEAD = "e" * 40


class _FakeGit:
    """Canned git output keyed by subcommand."""

    def __init__(self, *, ls: str = "", head: str = HEAD + "\n", status: str = "") -> None:
        self._by_subcommand = {"ls-files": ls, "rev-parse": head, "status": status}
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, repo_root: pathlib.Path, args: Sequence[str]) -> str:
        self.calls.append(tuple(args))
        return self._by_subcommand[args[0]]


def _install_contents(contents: dict[str, bytes]) -> None:
    def _fake_read(path: pathlib.Path) -> bytes:
        return contents[path.name]

    core_hooks.read_bytes = _fake_read


class TestPureHelpers:
    def test_detects_python_by_extension(self) -> None:
        assert detect_language("src/code_corpus/core/select.py") == "python"

    def test_detects_nothing_for_an_unknown_extension(self) -> None:
        assert detect_language("README.txt") is None

    def test_token_estimate_is_chars_over_four(self) -> None:
        assert approx_tokens("x" * 8) == 2

    def test_token_estimate_never_reports_zero(self) -> None:
        assert approx_tokens("xy") == 1

    def test_decodes_and_normalizes_line_endings(self) -> None:
        assert decode_source_text("api", "m.py", b"a = 1\r\nb = 2\r\n") == "a = 1\nb = 2\n"

    def test_rejects_invalid_utf8_naming_the_file(self) -> None:
        with pytest.raises(ValueError, match=re.escape("api/bad.py is not valid UTF-8")):
            decode_source_text("api", "bad.py", b"\xff\xfe")


class TestGitReads:
    def test_tracked_files_parses_nul_separated_output(self) -> None:
        core_hooks.run_git = _FakeGit(ls="a.py\0sub/b.py\0notes.txt\0")
        assert tracked_files(pathlib.Path("repo")) == ["a.py", "sub/b.py", "notes.txt"]

    def test_tracked_files_of_an_empty_repository(self) -> None:
        core_hooks.run_git = _FakeGit(ls="")
        assert tracked_files(pathlib.Path("repo")) == []

    def test_head_strips_the_trailing_newline(self) -> None:
        core_hooks.run_git = _FakeGit()
        assert git_head(pathlib.Path("repo")) == HEAD

    def test_a_clean_tree_is_not_dirty(self) -> None:
        core_hooks.run_git = _FakeGit(status="")
        assert git_dirty(pathlib.Path("repo")) is False

    def test_a_modified_tree_is_dirty(self) -> None:
        core_hooks.run_git = _FakeGit(status=" M src/m.py\n")
        assert git_dirty(pathlib.Path("repo")) is True

    def test_repo_pin_combines_head_and_dirtiness(self) -> None:
        core_hooks.run_git = _FakeGit(status="?? new.py\n")
        assert repo_pin("api", pathlib.Path("repo")) == RepoPin(name="api", commit=HEAD, dirty=True)


class TestSelectFiles:
    def test_selects_filters_and_counts(self) -> None:
        core_hooks.run_git = _FakeGit(ls="keep.py\0document_categories.py\0empty.py\0notes.txt\0")
        _install_contents(
            {
                "keep.py": b"x = 1\n",
                "document_categories.py": b"GENERATED = True\n",
                "empty.py": b"  \n\n",
            }
        )
        outcome = select_files("api", pathlib.Path("repo"), ["python"])
        assert [file.path for file in outcome.files] == ["keep.py"]
        assert outcome.excluded_generated == 1
        assert outcome.excluded_empty == 1

    def test_a_selected_file_carries_digest_tokens_and_language(self) -> None:
        core_hooks.run_git = _FakeGit(ls="keep.py\0")
        _install_contents({"keep.py": b"value = 1234\n"})
        outcome = select_files("api", pathlib.Path("repo"), ["python"])
        file = outcome.files[0]
        assert file.repo == "api"
        assert file.language == "python"
        assert file.text == "value = 1234\n"
        assert file.sha256 == hashlib.sha256(b"value = 1234\n").hexdigest()
        assert file.tokens_approx == 3

    def test_digests_the_normalized_content_not_the_raw_bytes(self) -> None:
        core_hooks.run_git = _FakeGit(ls="keep.py\0")
        _install_contents({"keep.py": b"x = 1\r\n"})
        outcome = select_files("api", pathlib.Path("repo"), ["python"])
        assert outcome.files[0].sha256 == hashlib.sha256(b"x = 1\n").hexdigest()

    def test_excludes_generated_files_in_subdirectories(self) -> None:
        core_hooks.run_git = _FakeGit(ls="pkg/document_categories.py\0")
        _install_contents({})
        outcome = select_files("api", pathlib.Path("repo"), ["python"])
        assert outcome.files == ()
        assert outcome.excluded_generated == 1

    def test_rejects_an_unknown_language(self) -> None:
        core_hooks.run_git = _FakeGit(ls="")
        with pytest.raises(ValueError, match="unknown language 'go'; known languages: python"):
            select_files("api", pathlib.Path("repo"), ["go"])


class TestRefusingAnUnreproducibleEmission:
    """A corpus whose commit does not describe it is not evidence.

    The condition was ALREADY recorded and already ignored: code-corpus-v1
    was emitted with both repositories dirty, then trained on, evaluated and
    reported, with ``dirty: true`` sitting in its manifest the whole time.
    Recording a fact nobody reads is not a check, which is why this refuses.
    """

    def test_a_clean_set_of_pins_is_admitted(self) -> None:
        """The ordinary case must stay silent."""
        pins = [
            RepoPin(name="api", commit="a" * 40, dirty=False),
            RepoPin(name="mcp", commit="b" * 40, dirty=False),
        ]

        require_reproducible_pins(pins)

    def test_one_dirty_repository_is_refused(self) -> None:
        """Refused at emission, because it cannot be repaired afterwards."""
        pins = [RepoPin(name="api", commit="a" * 40, dirty=True)]

        with pytest.raises(ValueError, match="refusing to emit: api has uncommitted changes"):
            require_reproducible_pins(pins)

    def test_every_dirty_repository_is_named(self) -> None:
        """A caller with two repositories needs to know which to clean.

        Naming only the first would send someone to clean one tree and hit
        the same refusal again on the next run.

        The clean repository is called ``tidyrepo`` rather than ``clean``
        because the refusal's own text contains the word "clean" -- the first
        version of this test asserted the absence of a substring the message
        legitimately carries, and failed for a reason that had nothing to do
        with the behaviour.
        """
        pins = [
            RepoPin(name="api", commit="a" * 40, dirty=True),
            RepoPin(name="tidyrepo", commit="b" * 40, dirty=False),
            RepoPin(name="mcp", commit="c" * 40, dirty=True),
        ]

        with pytest.raises(ValueError) as excinfo:
            require_reproducible_pins(pins)

        assert "api, mcp" in str(excinfo.value)
        assert "tidyrepo" not in str(excinfo.value)

    def test_the_refusal_says_how_to_proceed(self) -> None:
        """A refusal that names no way forward is one people work around."""
        with pytest.raises(ValueError, match="git worktree add"):
            require_reproducible_pins([RepoPin(name="api", commit="a" * 40, dirty=True)])
