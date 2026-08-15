"""Deriving what each language supports from the rule files on disk.

The previous version of this file reached into the module and reassigned
``_RULE_DIR`` and ``get_supported_languages`` to reach two defensive branches:
"the language claims Latin but no Latin file exists" and its IPA twin. Those
branches were unreachable in production — support is *derived* from the files,
so a format cannot be reported without its file — and the only way to reach
them was to make the module lie to itself.

The branches are gone, and with them the patching. :func:`scan_supported`
takes the directory as an argument, so a test hands it real files in a
temporary directory instead.
"""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest

from turkic_api.core.translit import (
    IPA_FORMAT,
    LATIN_FORMAT,
    UnsupportedTransliterationError,
    clear_translit_caches,
    get_supported_languages,
    scan_supported,
    to_ipa,
    to_latin,
)


@pytest.fixture(autouse=True)
def _clear_caches() -> Generator[None, None, None]:
    """Leave the module's caches as they were found."""
    yield
    clear_translit_caches()


def _touch(directory: Path, *names: str) -> None:
    """Create empty rule files with the given names."""
    for name in names:
        (directory / name).write_text("", encoding="utf-8")


class TestScanSupported:
    """Reading a directory of rule files."""

    def test_a_latin_file_makes_a_language_support_latin(self, tmp_path: Path) -> None:
        _touch(tmp_path, "kk_lat.rules")

        assert scan_supported(tmp_path) == {"kk": [LATIN_FORMAT]}

    def test_an_ipa_file_makes_a_language_support_ipa(self, tmp_path: Path) -> None:
        _touch(tmp_path, "kk_ipa.rules")

        assert scan_supported(tmp_path) == {"kk": [IPA_FORMAT]}

    def test_a_language_can_support_both(self, tmp_path: Path) -> None:
        _touch(tmp_path, "kk_ipa.rules", "kk_lat.rules")

        assert scan_supported(tmp_path) == {"kk": [IPA_FORMAT, LATIN_FORMAT]}

    def test_several_languages_are_reported_separately(self, tmp_path: Path) -> None:
        _touch(tmp_path, "kk_lat.rules", "ky_lat.rules")

        assert scan_supported(tmp_path) == {"kk": [LATIN_FORMAT], "ky": [LATIN_FORMAT]}

    def test_a_file_with_no_suffix_is_not_a_rule_file(self, tmp_path: Path) -> None:
        _touch(tmp_path, "notes.rules")

        assert scan_supported(tmp_path) == {}

    def test_a_suffix_no_format_claims_is_ignored(self, tmp_path: Path) -> None:
        """``uzc`` is reached through the Uzbek pass, not as a language of its own."""
        _touch(tmp_path, "kk_cyrillic.rules")

        assert scan_supported(tmp_path) == {}

    def test_an_empty_directory_supports_nothing(self, tmp_path: Path) -> None:
        assert scan_supported(tmp_path) == {}

    def test_non_rule_files_are_ignored(self, tmp_path: Path) -> None:
        (tmp_path / "PROVENANCE.md").write_text("", encoding="utf-8")
        _touch(tmp_path, "kk_lat.rules")

        assert scan_supported(tmp_path) == {"kk": [LATIN_FORMAT]}


class TestPackagedLanguages:
    """What the shipped rule files actually provide."""

    def test_every_language_with_ipa_rules_is_reported(self) -> None:
        supported = get_supported_languages()

        with_ipa = {code for code, formats in supported.items() if IPA_FORMAT in formats}
        assert with_ipa == {"az", "fi", "kk", "ky", "ru", "tr", "ug", "uz", "uzc"}

    def test_every_language_with_latin_rules_is_reported(self) -> None:
        supported = get_supported_languages()

        with_latin = {code for code, formats in supported.items() if LATIN_FORMAT in formats}
        assert with_latin == {"ar", "kk", "ky", "tr"}

    def test_the_answer_is_remembered_between_calls(self) -> None:
        assert get_supported_languages() == get_supported_languages()

    def test_clearing_the_caches_lets_it_be_scanned_again(self) -> None:
        first = get_supported_languages()

        clear_translit_caches()

        assert get_supported_languages() == first


class TestUnsupportedRequests:
    """Asking for a format a language has no rules for."""

    def test_latin_for_a_language_with_only_ipa_rules(self) -> None:
        with pytest.raises(UnsupportedTransliterationError) as caught:
            to_latin("text", "fi")

        assert caught.value.language == "fi"
        assert caught.value.output_format == LATIN_FORMAT
        assert caught.value.code == "TURKIC_TRANSLIT_001_UNSUPPORTED_FORMAT"

    def test_the_error_lists_what_could_have_been_asked_for(self) -> None:
        with pytest.raises(UnsupportedTransliterationError) as caught:
            to_latin("text", "fi")

        assert caught.value.available == ("ar", "kk", "ky", "tr")

    def test_ipa_for_a_language_with_no_rules_at_all(self) -> None:
        with pytest.raises(UnsupportedTransliterationError) as caught:
            to_ipa("text", "xx")

        assert caught.value.output_format == IPA_FORMAT

    def test_latin_for_a_language_with_no_rules_at_all(self) -> None:
        with pytest.raises(UnsupportedTransliterationError):
            to_latin("text", "xx")
