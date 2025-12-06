from __future__ import annotations

import pytest

from turkic_api.core.translit import to_ipa, to_latin


def test_supported_languages_latin_and_ipa() -> None:
    # Smoke tests on rule-driven transliteration
    assert to_latin("Қазақстан", "kk").startswith("Q") or to_latin("Қазақстан", "kk").startswith(
        "Q".lower()
    )
    out = to_ipa("Қазақстан", "kk")
    assert type(out) is str
    assert out  # non-empty output


def test_to_latin_unknown_language_raises() -> None:
    with pytest.raises(ValueError, match="not supported"):
        to_latin("text", "xx")


def test_to_latin_with_arabic_transliteration() -> None:
    # Arabic content should pass through ar_lat rules when include_arabic=True
    out = to_latin("سلام", "kk", include_arabic=True)
    assert type(out) is str
    assert out


def test_to_ipa_unknown_language_raises() -> None:
    with pytest.raises(ValueError, match="not supported"):
        to_ipa("text", "xx")
