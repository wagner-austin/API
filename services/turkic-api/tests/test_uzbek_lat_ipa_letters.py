"""Test Uzbek Latin → IPA transliteration using core to_ipa().

Source: Ido, S. (2025). "Uzbek".
        Journal of the International Phonetic Association, 55(1-2): 152-168.
        DOI: https://doi.org/10.1017/S0025100324000148

Uses language code 'uz' (Latin is the default).
"""

from __future__ import annotations

import pytest

from turkic_api.core.translit import to_ipa

# ---------------------------------------------------------------------------
# 1.  Single-letter gold standard (Uzbek Latin)
# Source: Ido, S. (2025). Standard Tashkent Uzbek Latin orthography

GOLD_CONSONANTS = {
    "b": "b",
    "v": "v",
    "g": "ɡ",
    "d": "d",
    "j": "d͡ʒ",  # affricate
    "z": "z",
    "y": "j",  # palatal glide
    "k": "k",
    "q": "q",
    "l": "l",
    "m": "m",
    "n": "n",
    "p": "p",
    "r": "r",
    "s": "s",
    "t": "t",
    "f": "f",
    "x": "x",
    "h": "h",
}

GOLD_VOWELS = {
    "a": "a",
    "e": "e",
    "i": "i",
    "o": "ɔ",  # open-mid (not oʻ)
    "u": "u",
}

# Digraphs
DIGRAPH_TESTS = {
    "ng": "ŋ",
    "sh": "ʃ",
    "ch": "t͡ʃ",
    "ts": "t͡s",
}

# Letters with apostrophe
APOSTROPHE_TESTS = {
    "oʻ": "o",  # close-mid back rounded
    "gʻ": "ʁ",  # voiced uvular fricative
}

# Iotated sequences (loans)
IOTATED_TESTS = {
    "yo": "jo",
    "yu": "ju",
    "ya": "ja",
    "ye": "je",
}

# ---------------------------------------------------------------------------
# 2.  Combined gold standard

GOLD = {**GOLD_CONSONANTS, **GOLD_VOWELS}

# ---------------------------------------------------------------------------
# 3.  Parametrised tests


@pytest.mark.parametrize(("input", "expected"), GOLD.items())
def test_letter_to_ipa(input: str, expected: str) -> None:
    """Single Uzbek Latin letters."""
    assert to_ipa(input, "uz") == expected


@pytest.mark.parametrize(("input", "expected"), DIGRAPH_TESTS.items())
def test_digraphs(input: str, expected: str) -> None:
    """Uzbek Latin digraphs."""
    assert to_ipa(input, "uz") == expected


@pytest.mark.parametrize(("input", "expected"), APOSTROPHE_TESTS.items())
def test_apostrophe_letters(input: str, expected: str) -> None:
    """Letters with apostrophe (oʻ, gʻ)."""
    assert to_ipa(input, "uz") == expected


@pytest.mark.parametrize(("input", "expected"), IOTATED_TESTS.items())
def test_iotated_sequences(input: str, expected: str) -> None:
    """Iotated sequences in loans."""
    assert to_ipa(input, "uz") == expected


# ---------------------------------------------------------------------------
# 4.  Word-level tests


def test_uzbek_lat_words() -> None:
    """Common Uzbek Latin words."""
    examples = {
        "kitob": "kitɔb",  # book
        "qalam": "qalam",  # pen
        "maktab": "maktab",  # school
        "shahar": "ʃahar",  # city
        "choy": "t͡ʃɔj",  # tea
        "gʻisht": "ʁiʃt",  # brick
        "oʻqish": "oqiʃ",  # reading
    }
    for uz, ipa in examples.items():
        assert to_ipa(uz, "uz") == ipa
