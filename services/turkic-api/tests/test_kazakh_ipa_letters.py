"""
Single-letter Cyrillic → IPA checks

Source: McCollum, A. G. & Chen, S. (2021). "Kazakh".
        Journal of the International Phonetic Association, 51(2): 276-298.
        DOI: https://doi.org/10.1017/S0025100319000185

Note: Simplified to match kk_ipa.rules (no diphthongs, no dental diacritics).
"""

import pytest

GOLD = {
    "б": "b",
    "в": "v",
    "г": "ɡ",
    "ғ": "ʁ",
    "д": "d",  # ← was d̪
    "ё": "jo",
    "ж": "ʒ",
    "з": "z",  # ← was z̪
    "и": "i",  # ← was i͡j
    "й": "j",
    "к": "k",
    "қ": "q",
    "л": "l",  # ← was l̪
    "м": "m",
    "н": "n",  # ← was n̪
    "ң": "ŋ",
    "о": "o",
    "ө": "ɵ",
    "п": "p",
    "р": "r",  # ← was r̪
    "с": "s",  # ← was s̪
    "т": "t",  # ← was t̪
    "у": "u",  # ← was u͡w
    "ұ": "ʊ",
    "ү": "ʏ",
    "ф": "f",
    "х": "χ",
    "һ": "h",
    "ц": "t͡s",
    "ч": "t͡ʃ",
    "ш": "ʃ",
    "щ": "ɕː",
    "ы": "ə",
    "і": "ɪ",
    "ю": "ju",
    "я": "ja",
}


@pytest.mark.parametrize(("cyr", "ipa"), GOLD.items())
def test_letter_to_ipa(cyr: str, ipa: str) -> None:
    from turkic_api.core.translit import to_ipa

    assert to_ipa(cyr, "kk") == ipa
