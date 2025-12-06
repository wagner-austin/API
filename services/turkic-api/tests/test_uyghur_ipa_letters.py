"""Test Uyghur → IPA transliteration using core to_ipa().

Source: McCollum, A. G. (2021). "Transparency, locality, and contrast in Uyghur backness harmony".
        Laboratory Phonology: Journal of the Association for Laboratory Phonology, 12(1): article 8.
        DOI: https://doi.org/10.5334/labphon.239
"""

from __future__ import annotations

import pytest

from turkic_api.core.translit import to_ipa

# ---------------------------------------------------------------------------
# 1.  Single-letter gold standard (Arabic script → IPA)
# Source: McCollum, A. G. (2021)

GOLD_CONSONANTS = {
    "ب": "b",
    "پ": "p",
    "ت": "t",
    "ج": "d͡ʒ",
    "چ": "t͡ʃ",
    "خ": "χ",
    "د": "d",
    "ر": "r",
    "ز": "z",
    "ژ": "ʒ",
    "س": "s",
    "ش": "ʃ",
    "غ": "ʁ",
    "ف": "f",
    "ق": "q",
    "ك": "k",
    "گ": "ɡ",
    "ڭ": "ŋ",
    "ل": "l",
    "م": "m",
    "ن": "n",
    "ھ": "h",
    "ۋ": "w",
    "ي": "j",
}

GOLD_VOWELS = {
    "ا": "ɑ",
    "ە": "æ",
    "ې": "e",
    "ى": "i",
    "و": "o",
    "ۇ": "u",
    "ۆ": "ø",
    "ۈ": "y",
}

# Hamza should be deleted
HAMZA_TEST = {
    "ئا": "ɑ",  # hamza + alif = just the vowel
    "ئە": "æ",
}

# ---------------------------------------------------------------------------
# 2.  Combined tests (all graphemes)

GOLD = {**GOLD_CONSONANTS, **GOLD_VOWELS}

# ---------------------------------------------------------------------------
# 3.  Parametrised tests


@pytest.mark.parametrize(("input", "expected"), GOLD.items())
def test_letter_to_ipa(input: str, expected: str) -> None:
    """Single Uyghur letters (Arabic script)."""
    assert to_ipa(input, "ug") == expected


@pytest.mark.parametrize(("input", "expected"), HAMZA_TEST.items())
def test_hamza_deletion(input: str, expected: str) -> None:
    """Uyghur hamza deletion."""
    assert to_ipa(input, "ug") == expected
