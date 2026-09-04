"""Code kept for an old caller is refused for every package at once.

The rule this replaces lived in ``clients/TankpitBot/scripts/shim_rules.py``
and was called by nothing but its own test, so it protected one package out of
forty-four. The other forty-three had accumulated exactly what it bans: five
re-export modules in ``handwriting-ai``, and one each in ``grandma-api``,
``platform_calendar`` and ``github-stats-api``, every one carrying a comment
that said so.

The vocabulary here is narrower than the rule it lifts, and the module
docstring records the measurement that narrowed it. These tests pin the
narrowing, because a pattern dropped without a test is a pattern that gets
added back.
"""

from __future__ import annotations

import pathlib

import pytest

from monorepo_guards import shim_rules
from monorepo_guards.shim_rules import ShimRule, compatibility_markers


def _module(tmp_path: pathlib.Path, body: str, *, root: str = "src") -> pathlib.Path:
    """Write one module under a package-shaped directory.

    Args:
        tmp_path: The test's temporary directory.
        body: The module source.
        root: The scanned root to write it under.

    Returns:
        The written file.
    """
    directory = tmp_path / root / "pkg"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "module.py"
    path.write_text(body, encoding="utf-8")
    return path


def _kinds(paths: list[pathlib.Path]) -> list[str]:
    """Run the rule and report the violation kinds it found.

    Args:
        paths: Files to check.

    Returns:
        One kind per violation, in file order.
    """
    return [violation.kind for violation in ShimRule().run(paths)]


class TestTheDefectItExistsFor:
    """Prose announcing a shim is a shim."""

    @pytest.mark.parametrize(
        "marker",
        [
            "# Re-export for backwards compatibility.",
            "# Kept for backward compatibility with existing code.",
            "# back-compat alias",
            "# back compat",
            '"""Re-exported from platform_core for backward compatibility."""',
            "# kept for api compatibility",
            "# kept for signature compatibility",
        ],
    )
    def test_each_announcement_is_refused(self, tmp_path: pathlib.Path, marker: str) -> None:
        path = _module(tmp_path, f"{marker}\nVALUE = 1\n")

        assert _kinds([path]) == ["shim-compatibility-marker"]

    def test_the_refusal_says_what_to_do_instead(self, tmp_path: pathlib.Path) -> None:
        """ "Marker found" leaves the reader to guess whether deleting it is
        allowed. There is no released version to stay compatible with, and
        saying so is the whole argument for the rule."""
        path = _module(tmp_path, "# Re-export for backwards compatibility.\nVALUE = 1\n")

        assert "no released version to stay compatible with" in ShimRule().run([path])[0].line

    def test_the_matched_text_is_quoted_back(self, tmp_path: pathlib.Path) -> None:
        path = _module(tmp_path, "VALUE = 1\n# kept for api compatibility\n")

        violation = ShimRule().run([path])[0]

        assert "kept for api compatibility" in violation.line
        assert violation.line_no == 2

    def test_every_offending_line_is_reported(self, tmp_path: pathlib.Path) -> None:
        path = _module(tmp_path, "# back-compat\nVALUE = 1\n# kept for api compatibility\n")

        assert [v.line_no for v in ShimRule().run([path])] == [1, 3]


class TestTheVocabularyThatWasDroppedOnPurpose:
    """Three patterns named a THING or an act, not an intent, and were removed.

    Measured across this monorepo, all three were false positives every time.
    The cases here are real lines from the tree, kept as tests so that
    re-adding any of them fails rather than quietly renaming a probe.
    """

    def test_nvidias_legacy_entry_point_is_not_a_shim(self, tmp_path: pathlib.Path) -> None:
        """`legacy_gemm_probe` exists to measure the `cublasSgemm` legacy
        path. Banning the word would force renaming the probe after its
        subject."""
        path = _module(tmp_path, '"""``mm(x, w)`` takes the legacy ``cublasSgemm`` path."""\n')

        assert _kinds([path]) == []

    def test_a_legacy_file_format_is_not_a_shim(self, tmp_path: pathlib.Path) -> None:
        path = _module(tmp_path, "# openpyxl cannot read legacy .xls files\n")

        assert _kinds([path]) == []

    def test_a_boundary_conversion_is_not_a_shim(self, tmp_path: pathlib.Path) -> None:
        """The real line, from `chunked_csv_reader`. Converting a Polars frame
        into the list-of-lists its callers take is a boundary conversion, which
        this workspace asks for. Only the full phrase "backward compatibility"
        states the banned intent, and every real shim in the tree used it."""
        path = _module(
            tmp_path,
            "# Converts to list-of-lists format for compatibility with\n"
            "# existing loader infrastructure.\n",
        )

        assert _kinds([path]) == []

    def test_a_third_party_deprecation_warning_is_not_a_shim(self, tmp_path: pathlib.Path) -> None:
        """Every `deprecat` match in the tree was somebody else's warning --
        SWIG's, fasttext's, ddtrace's."""
        path = _module(tmp_path, "# suppress the SWIG deprecation warnings\n")

        assert _kinds([path]) == []


class TestWhatItDeliberatelyIgnores:
    """Scope is shipped code, and the alias shapes belong to another rule."""

    def test_a_test_may_assert_that_there_is_no_back_compat(self, tmp_path: pathlib.Path) -> None:
        """Real docstrings in the tree read "Missing ``manual_mode`` raises --
        no back-compat default". Firing on those would make the rule punish
        the evidence that it holds."""
        path = _module(
            tmp_path,
            '"""Missing field raises -- no back-compat default."""\n',
            root="tests",
        )

        assert _kinds([path]) == []

    def test_a_file_outside_a_scanned_root_is_not_shipped_code(
        self, tmp_path: pathlib.Path
    ) -> None:
        path = tmp_path / "notes.py"
        path.write_text("# kept for backward compatibility\n", encoding="utf-8")

        assert _kinds([path]) == []

    def test_scripts_are_shipped_code(self, tmp_path: pathlib.Path) -> None:
        path = _module(tmp_path, "# kept for backward compatibility\n", root="scripts")

        assert _kinds([path]) == ["shim-compatibility-marker"]

    def test_an_alias_is_left_to_the_rule_that_owns_it(self, tmp_path: pathlib.Path) -> None:
        """`PassthroughRule` was measured at 27 findings and no false
        positives over 4896 files. A second, blunter copy here reported four,
        all of them correct code -- constants and a module, which its
        type-spelling predicate is what makes it skip."""
        path = _module(
            tmp_path,
            "from other import RING_SLOT_RADIUS\n\nJOB_RADIUS = RING_SLOT_RADIUS\n"
            '__all__ = ["JOB_RADIUS"]\n',
        )

        assert _kinds([path]) == []

    def test_the_module_defining_the_vocabulary_does_not_report_itself(self) -> None:
        """It necessarily contains every pattern it bans. Exempted by resolved
        path, so this is an identity rather than an allowlist that could grow
        a second entry."""
        owner = pathlib.Path(shim_rules.__file__)

        assert compatibility_markers(owner.read_text(encoding="utf-8")) != []
        assert _kinds([owner]) == []


class TestTheHelper:
    """`compatibility_markers` is the half a caller can reuse."""

    def test_it_reports_line_numbers_and_matched_text(self) -> None:
        source = "import os\n# back-compat shim\nVALUE = 1\n"

        assert compatibility_markers(source) == [(2, "back-compat")]

    def test_a_clean_module_reports_nothing(self) -> None:
        assert compatibility_markers("import os\nVALUE = 1\n") == []

    def test_matching_is_case_insensitive(self) -> None:
        assert compatibility_markers("# Kept For Backward Compatibility\n") == [
            (1, "Backward Compatibility")
        ]

    def test_a_marker_split_across_a_line_break_is_found(self) -> None:
        """The real docstring from `covenant-radar-api`'s history entry. A
        line-at-a-time scan reported it clean, which means `ruff format`
        rewrapping a comment could have hidden any of these."""
        source = (
            "stored as flat best_* fields for backward\n"
            "compatibility with the JSONL history format.\n"
        )

        assert compatibility_markers(source) == [(1, "backward compatibility")]

    def test_the_reported_text_reads_as_one_phrase(self) -> None:
        """A wrapped marker must not carry its second line's indentation into
        the refusal, or the message is unreadable at exactly the moment it
        matters."""
        source = "for backward\n        compatibility with the old format\n"

        assert compatibility_markers(source) == [(1, "backward compatibility")]


__all__ = [
    "TestTheDefectItExistsFor",
    "TestTheHelper",
    "TestTheVocabularyThatWasDroppedOnPurpose",
    "TestWhatItDeliberatelyIgnores",
]
