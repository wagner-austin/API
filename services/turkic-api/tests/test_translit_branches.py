from __future__ import annotations

from turkic_api.core.translit import to_ipa


def test_to_ipa_uz_latin_script() -> None:
    # uz now has uz_ipa.rules (Latin script), so this should work
    result = to_ipa("O'zbekiston", "uz")
    assert result  # Should not raise, should return IPA transcription
    assert "ɔ" in result  # Uzbek Latin 'o' should map to /ɔ/


def test_to_ipa_uz_cyrillic_script() -> None:
    # ADDITION 2025-12-02: Test Cyrillic Uzbek via uzc_ipa.rules
    # See docs/ipa-rules-refactor.md
    result = to_ipa("Ўзбекистон", "uz")
    assert result  # Should not raise
    assert "o" in result  # Cyrillic Ў maps to /o/
    assert "ɔ" in result  # Cyrillic О maps to /ɔ/
