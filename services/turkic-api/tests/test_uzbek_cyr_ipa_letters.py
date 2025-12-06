"""Test Uzbek Cyrillic → IPA transliteration using core to_ipa().

Source: Ido, S. (2025). "Uzbek".
        Journal of the International Phonetic Association, 55(1-2): 152-168.
        DOI: https://doi.org/10.1017/S0025100324000148

Uses language code 'uzc' (Cyrillic variant).
"""

from __future__ import annotations

import pytest

from turkic_api.core.translit import to_ipa

# ---------------------------------------------------------------------------
# 1.  Single-letter gold standard (Uzbek Cyrillic)
# Source: Ido, S. (2025). Standard Tashkent Uzbek Cyrillic orthography

GOLD_CONSONANTS = {
    "б": "b",
    "в": "v",
    "г": "ɡ",
    "ғ": "ʁ",  # gʻ
    "д": "d",
    "ж": "d͡ʃ",  # affricate
    "з": "z",
    "й": "j",
    "к": "k",
    "қ": "q",
    "л": "l",
    "м": "m",
    "н": "n",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "ў": "o",  # oʻ - close-mid back rounded
    "у": "u",
    "ф": "f",
    "х": "x",
    "ҳ": "h",
    "ч": "t͡ʃ",
    "ш": "ʃ",
    "ц": "t͡s",
}

GOLD_VOWELS = {
    "а": "a",
    "и": "i",
    "о": "ɔ",  # open-mid
    "у": "u",
    "э": "e",
    # Note: "е" is context-sensitive, tested separately in CONTEXT_E
}

# Iotated vowels (word-initial or after vowels)
GOLD_IOTATED = {
    "ё": "jo",
    "ю": "ju",
    "я": "ja",
}

# Context-sensitive: е → je word-initially
CONTEXT_E = {
    "ем": "jem",  # word-initial
    "еш": "jeʃ",
}

# Digraphs
DIGRAPH_TESTS = {
    "нг": "ŋ",
    "тс": "t͡s",
}

# Hard/soft signs
SPECIAL_TESTS = {
    "ъ": "ʔ",  # glottal stop
    "ь": "ʲ",  # palatalization
}

# ---------------------------------------------------------------------------
# 2.  Combined gold standard

GOLD = {**GOLD_CONSONANTS, **GOLD_VOWELS}

# ---------------------------------------------------------------------------
# 3.  Parametrised tests


@pytest.mark.parametrize(("input", "expected"), GOLD.items())
def test_letter_to_ipa(input: str, expected: str) -> None:
    """Single Uzbek Cyrillic letters."""
    assert to_ipa(input, "uzc") == expected


@pytest.mark.parametrize(("input", "expected"), GOLD_IOTATED.items())
def test_iotated_vowels(input: str, expected: str) -> None:
    """Iotated vowels in Uzbek Cyrillic."""
    assert to_ipa(input, "uzc") == expected


@pytest.mark.parametrize(("input", "expected"), CONTEXT_E.items())
def test_context_e(input: str, expected: str) -> None:
    """Context-sensitive е → je."""
    assert to_ipa(input, "uzc") == expected


@pytest.mark.parametrize(("input", "expected"), DIGRAPH_TESTS.items())
def test_digraphs(input: str, expected: str) -> None:
    """Uzbek Cyrillic digraphs."""
    assert to_ipa(input, "uzc") == expected


@pytest.mark.parametrize(("input", "expected"), SPECIAL_TESTS.items())
def test_special_chars(input: str, expected: str) -> None:
    """Hard/soft signs."""
    assert to_ipa(input, "uzc") == expected


# ---------------------------------------------------------------------------
# 4.  Word-level tests


def test_uzbek_cyr_words() -> None:
    """Common Uzbek Cyrillic words."""
    examples = {
        "китоб": "kitɔb",  # book
        "қалам": "qalam",  # pen
        "мактаб": "maktab",  # school
    }
    for uz, ipa in examples.items():
        assert to_ipa(uz, "uzc") == ipa
