"""Test Azerbaijani → IPA transliteration using core to_ipa().

Source: Ghaffarvand Mokari, P. & Werner, S. (2017). "Azerbaijani".
        Journal of the International Phonetic Association, 47(2): 207–212.
        DOI: https://doi.org/10.1017/S0025100317000184
"""

from __future__ import annotations

import pytest

from turkic_api.core.translit import to_ipa

# ---------------------------------------------------------------------------
# 1.  Single-letter gold standard
# Source: Ghaffarvand Mokari, P. & Werner, S. (2017)
GOLD = {
    "a": "ɑ",
    "b": "b",
    "c": "d͡ʒ",
    "ç": "t͡ʃ",
    "d": "d",
    "e": "e",
    "ə": "æ",
    "f": "f",
    "g": "ɟ",
    "ğ": "ɣ",
    "h": "h",
    "x": "x",
    "ı": "ɯ",
    "i": "i",
    "j": "ʒ",
    "k": "k",
    "q": "ɡ",
    "l": "l",
    "m": "m",
    "n": "n",
    "o": "o",
    "ö": "œ",
    "p": "p",
    "r": "ɾ",
    "s": "s",
    "ş": "ʃ",
    "t": "t",
    "u": "u",
    "ü": "y",
    "v": "v",
    "y": "j",
    "z": "z",
}

# ---------------------------------------------------------------------------
# 2.  Word-level sanity checks

WORD_TESTS = {
    "salam": "sɑlɑm",
    "xoş": "xoʃ",
    "gəlmisiniz": "ɟælmisiniz",
    "sağol": "sɑɣol",
}

# ---------------------------------------------------------------------------
# 3.  Parametrised tests


@pytest.mark.parametrize(("input", "expected"), GOLD.items())
def test_letter_to_ipa(input: str, expected: str) -> None:
    """Single Azerbaijani letters."""
    assert to_ipa(input, "az") == expected


def test_azerbaijani_word_examples() -> None:
    """Common Azerbaijani words."""
    for az, ipa in WORD_TESTS.items():
        assert to_ipa(az, "az") == ipa
