"""Decoding and validating what arrives from outside the process."""

from __future__ import annotations

import pytest
from platform_core.error_codes_tooling import CommitScopeErrorCode
from platform_core.errors import AppError

from commit_scope.contracts import (
    ScopeDecision,
    decode_scope_declaration,
    decode_staged_paths,
    encode_scope_decision,
    normalise_path,
    require_relative_scope_entry,
)


class TestNormalisePath:
    """Reducing a written path to its comparison form."""

    def test_folds_backslashes_and_strips_trailing_slash(self) -> None:
        """A Windows-style directory becomes the git form."""
        assert normalise_path("libs\\platform_core\\") == "libs/platform_core"

    def test_trims_surrounding_whitespace(self) -> None:
        """A declaration split across lines leaves padding behind."""
        assert normalise_path("  libs/platform_core  ") == "libs/platform_core"

    def test_blank_becomes_empty(self) -> None:
        """Whitespace is not a path, and callers drop the empty result."""
        assert normalise_path("   ") == ""


class TestRequireRelativeScopeEntry:
    """Refusing declaration entries that could never match."""

    def test_admits_a_relative_entry(self) -> None:
        """The ordinary case passes through unchanged."""
        assert require_relative_scope_entry("libs/platform_core") == "libs/platform_core"

    def test_refuses_a_posix_absolute_entry(self) -> None:
        """A rooted path cannot match a repo-relative staged path."""
        with pytest.raises(AppError) as caught:
            require_relative_scope_entry("/etc/passwd")
        assert caught.value.code is CommitScopeErrorCode.SCOPE_ENTRY_NOT_RELATIVE

    def test_refuses_a_windows_drive_entry(self) -> None:
        """The form an operator on this machine would actually paste."""
        with pytest.raises(AppError) as caught:
            require_relative_scope_entry("C:/Users/Test/PROJECTS/API/libs")
        assert caught.value.code is CommitScopeErrorCode.SCOPE_ENTRY_NOT_RELATIVE

    def test_refuses_an_entry_that_climbs_out(self) -> None:
        """``..`` can never match, so it would allow nothing while appearing to."""
        with pytest.raises(AppError) as caught:
            require_relative_scope_entry("../other-repo/src")
        assert caught.value.code is CommitScopeErrorCode.SCOPE_ENTRY_ESCAPES_REPO

    def test_admits_a_name_merely_containing_two_dots(self) -> None:
        """Only a whole ``..`` SEGMENT escapes; a filename may contain dots."""
        assert require_relative_scope_entry("libs/a..b/x.py") == "libs/a..b/x.py"

    def test_admits_a_single_character_entry(self) -> None:
        """The drive-letter test reads index 1 and must not do so blindly."""
        assert require_relative_scope_entry("x") == "x"


class TestDecodeScopeDeclaration:
    """Turning the raw declaration into validated entries."""

    def test_absent_declaration_is_no_entries(self) -> None:
        """Unset is the undeclared case, not an empty allow-list."""
        assert decode_scope_declaration(None) == ()

    def test_blank_declaration_is_no_entries(self) -> None:
        """Separators alone declare nothing rather than declaring emptiness."""
        assert decode_scope_declaration("  \n,,\n ") == ()

    def test_splits_on_newline_and_comma_but_not_space(self) -> None:
        """A path may contain a space; splitting on it would fail open."""
        assert decode_scope_declaration("a b.py,c.py\nd.py") == ("a b.py", "c.py", "d.py")

    def test_preserves_declaration_order(self) -> None:
        """The unmatched report reads back in the order the author wrote."""
        assert decode_scope_declaration("z.py,a.py") == ("z.py", "a.py")

    def test_propagates_the_first_unmatchable_entry(self) -> None:
        """One bad entry refuses the whole declaration rather than being dropped."""
        with pytest.raises(AppError) as caught:
            decode_scope_declaration("libs/ok,/etc/passwd")
        assert caught.value.code is CommitScopeErrorCode.SCOPE_ENTRY_NOT_RELATIVE


class TestDecodeStagedPaths:
    """Reading git's index listing."""

    def test_drops_blank_lines_and_normalises(self) -> None:
        """Git's output is newline-terminated, so the last split is empty."""
        assert decode_staged_paths("a.py\n\n  \nb/c.py\n") == ("a.py", "b/c.py")

    def test_empty_index_is_no_paths(self) -> None:
        """A legitimate state: another session may have just emptied it."""
        assert decode_staged_paths("") == ()


class TestEncodeScopeDecision:
    """Rendering a decision structurally rather than as prose."""

    def test_encodes_every_field_as_json_ready_values(self) -> None:
        """Tuples become lists so the result survives a JSON round trip."""
        decision: ScopeDecision = {
            "declared": True,
            "staged": ("a.py", "b.py"),
            "out_of_scope": ("b.py",),
            "unmatched": ("c.py",),
        }
        assert encode_scope_decision(decision) == {
            "declared": True,
            "staged": ["a.py", "b.py"],
            "out_of_scope": ["b.py"],
            "unmatched": ["c.py"],
        }
