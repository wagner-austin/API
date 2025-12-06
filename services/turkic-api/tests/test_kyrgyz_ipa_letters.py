"""
Single Cyrillic characters → IPA

Source: McCollum, A. G. (2020). "Vowel harmony and positional variation in Kyrgyz".
        Laboratory Phonology, 11(1): article 25.
        DOI: https://doi.org/10.5334/labphon.247

Note: Updated for 2025-06 long-vowel & ɕː revision.
"""

import pytest

GOLD = {
    "а": "ɑ",
    "б": "b",
    "в": "v",
    "г": "ɡ",
    "д": "d",
    "е": "e",
    "ё": "jo",
    "ж": "ʒ",
    "з": "z",
    "и": "i",
    "й": "j",
    "к": "k",
    "қ": "q",
    "л": "l",
    "м": "m",
    "н": "n",
    "ң": "ŋ",
    "о": "o",
    "ө": "ø",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ү": "y",
    "ф": "f",
    "х": "x",
    "ц": "ʦ",
    "ч": "ʧ",
    "ш": "ʃ",
    "щ": "ɕː",  # ← updated (was ʃt͡ʃ)
    "ы": "ɯ",
    "э": "e",
    "ю": "ju",
    "я": "ja",
}


@pytest.mark.parametrize(("cyr", "ipa"), GOLD.items())
def test_kyrgyz_letter_to_ipa(cyr: str, ipa: str) -> None:
    from turkic_api.core.translit import to_ipa

    assert to_ipa(cyr, "ky") == ipa, f"{cyr} → {to_ipa(cyr, 'ky')}, expected {ipa}"
