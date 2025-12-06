"""Test Finnish → IPA transliteration using core to_ipa()."""

from __future__ import annotations

import pytest

from turkic_api.core.translit import to_ipa

# ---------------------------------------------------------------------------
# 1.  Single-letter gold standard
# Source: Suomi, K., Toivanen, J., & Ylitalo, R. (2008).
#         Finnish sound structure: Phonetics, phonology, phonotactics and prosody.
#         Studia Humaniora Ouluensia 9. Oulu University Press.
#         ISBN: 978-951-42-8984-2
GOLD_SHORT = {
    "a": "ɑ",
    "ä": "æ",
    "e": "e",
    "i": "i",
    "o": "o",
    "ö": "ø",
    "u": "u",
    "y": "y",
    "b": "b",
    "d": "d",
    "f": "f",
    "g": "ɡ",
    "h": "h",
    "j": "j",
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "p": "p",
    "r": "r",
    "s": "s",
    "t": "t",
    "v": "ʋ",
    "w": "ʋ",
}

# Long vowels (geminate vowels)
GOLD_LONG_VOWELS = {
    "aa": "ɑː",
    "ää": "æː",
    "ee": "eː",
    "ii": "iː",
    "oo": "oː",
    "öö": "øː",
    "uu": "uː",
    "yy": "yː",
}

# Long consonants
GOLD_LONG_CONSONANTS = {
    "pp": "pː",
    "tt": "tː",
    "kk": "kː",
    "mm": "mː",
    "nn": "nː",
    "ll": "lː",
    "rr": "rː",
    "ss": "sː",
}

# ---------------------------------------------------------------------------
# 2.  Context-sensitive expectations

CONTEXT_TESTS = {
    # nk → ŋk
    "kenkä": "keŋkæ",
    # ng → ŋː
    "kengän": "keŋːæn",
    # sh → ʃ, oo → oː
    "shampoo": "ʃɑmpoː",
}

# ---------------------------------------------------------------------------
# 3.  Word-level sanity checks

WORD_TESTS = {
    "hei": "hei",
    "moi": "moi",
    "kiitos": "kiːtos",
    "ole hyvä": "ole hyʋæ",
    "suomi": "suomi",
    "kala": "kɑlɑ",
}

# ---------------------------------------------------------------------------
# 4.  Parametrised tests


@pytest.mark.parametrize(("input", "expected"), GOLD_SHORT.items())
def test_letter_to_ipa(input: str, expected: str) -> None:
    """Single Finnish letters."""
    assert to_ipa(input, "fi") == expected


@pytest.mark.parametrize(("input", "expected"), GOLD_LONG_VOWELS.items())
def test_long_vowels(input: str, expected: str) -> None:
    """Finnish long vowels."""
    assert to_ipa(input, "fi") == expected


@pytest.mark.parametrize(("input", "expected"), GOLD_LONG_CONSONANTS.items())
def test_long_consonants(input: str, expected: str) -> None:
    """Finnish long consonants."""
    assert to_ipa(input, "fi") == expected


@pytest.mark.parametrize(("input", "expected"), CONTEXT_TESTS.items())
def test_context_sensitive_ipa(input: str, expected: str) -> None:
    """Context-sensitive Finnish rules."""
    assert to_ipa(input, "fi") == expected


def test_finnish_word_examples() -> None:
    """Common Finnish words."""
    for fi, ipa in WORD_TESTS.items():
        assert to_ipa(fi, "fi") == ipa
