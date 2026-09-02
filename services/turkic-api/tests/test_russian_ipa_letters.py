"""Test Russian → IPA transliteration using core to_ipa().

Source: Yanushevskaya, I. & Bunčić, D. (2015). "Russian".
        Journal of the International Phonetic Association, 45(2): 221-228.
        DOI: https://doi.org/10.1017/S0025100314000395
"""

from __future__ import annotations

import pytest

from turkic_api.core.translit import to_ipa

# ---------------------------------------------------------------------------
# 1. Single-letter gold standard (basic graphemes)
# Source: Yanushevskaya & Bunčić (2015), Table on p. 222

CONSONANTS = {
    "б": "b",
    "в": "v",
    "г": "ɡ",
    "д": "d",
    "ж": "ʒ",  # always hard
    "з": "z",
    "й": "j",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "ф": "f",
    "х": "x",
    "ц": "t͡s",  # always hard
    "ч": "t͡ʃʲ",  # always palatalized
    "ш": "ʃ",  # always hard
    "щ": "ʃʲː",  # always palatalized, long
}

VOWELS = {
    "а": "a",
    "э": "e",
    "и": "i",
    "о": "o",
    "у": "u",
    "ы": "ɨ",  # close central unrounded
}

IOTATED_VOWELS = {
    "е": "je",
    "ё": "jo",
    "ю": "ju",
    "я": "ja",
}

SPECIAL = {
    "ь": "ʲ",  # soft sign (palatalizes preceding consonant)
    "ъ": "",  # hard sign (silent, blocks palatalization)
}

# ---------------------------------------------------------------------------
# 2. Palatalized consonants (consonant + soft sign)

PALATALIZED = {
    "бь": "bʲ",
    "вь": "vʲ",
    "гь": "ɡʲ",
    "дь": "dʲ",
    "зь": "zʲ",
    "кь": "kʲ",
    "ль": "lʲ",
    "мь": "mʲ",
    "нь": "nʲ",
    "пь": "pʲ",
    "рь": "rʲ",
    "сь": "sʲ",
    "ть": "tʲ",
    "фь": "fʲ",
    "хь": "xʲ",
}

# ---------------------------------------------------------------------------
# 3. Word-level examples from the JIPA article

WORD_TESTS = {
    # From consonant examples (p. 222)
    "бас": "bas",  # 'bass'
    "дом": "dom",  # 'house'
    "нос": "nos",  # 'nose'
    "сад": "sad",  # 'garden'
    "кот": "kot",  # 'tomcat'
    "год": "ɡod",  # 'year'
    # Palatalized examples. ‹я› after a pairable consonant palatalises it
    # and inserts no glide, which is what the Illustration prints: /ˈrʲat/,
    # p. 222. The earlier expectation here was 'rjad', carried over from a
    # rule file that mapped every iotated vowel to a j-sequence.
    "рад": "rad",  # '(am) glad'
    "ряд": "rʲad",  # 'row'
    # Affricates
    "царь": "t͡sarʲ",  # 'tzar'
    "шар": "ʃar",  # 'ball'
    "жар": "ʒar",  # 'heat'
    "яма": "jama",  # 'pit'
}

# ---------------------------------------------------------------------------
# 4. Parametrised tests


@pytest.mark.parametrize(("cyr", "ipa"), CONSONANTS.items())
def test_consonant_to_ipa(cyr: str, ipa: str) -> None:
    """Russian consonant letters."""
    assert to_ipa(cyr, "ru") == ipa


@pytest.mark.parametrize(("cyr", "ipa"), VOWELS.items())
def test_vowel_to_ipa(cyr: str, ipa: str) -> None:
    """Russian vowel letters."""
    assert to_ipa(cyr, "ru") == ipa


@pytest.mark.parametrize(("cyr", "ipa"), IOTATED_VOWELS.items())
def test_iotated_vowel_to_ipa(cyr: str, ipa: str) -> None:
    """Russian iotated vowel letters."""
    assert to_ipa(cyr, "ru") == ipa


@pytest.mark.parametrize(("cyr", "ipa"), SPECIAL.items())
def test_special_to_ipa(cyr: str, ipa: str) -> None:
    """Russian special characters (ь, ъ)."""
    assert to_ipa(cyr, "ru") == ipa


@pytest.mark.parametrize(("cyr", "ipa"), PALATALIZED.items())
def test_palatalized_to_ipa(cyr: str, ipa: str) -> None:
    """Russian palatalized consonants (consonant + soft sign)."""
    assert to_ipa(cyr, "ru") == ipa


def test_russian_word_examples() -> None:
    """Common Russian words from the JIPA article."""
    for ru, ipa in WORD_TESTS.items():
        assert to_ipa(ru, "ru") == ipa, f"Failed for {ru!r}: expected {ipa!r}"
