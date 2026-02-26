"""
Gold-standard word list: Kyrgyz orthography → broad IPA
Source: McCollum 2020 “Vowel harmony and positional variation in Kyrgyz”,
Laboratory Phonology 11(1): 25 (CC-BY 4.0).

The list is adapted to the 2025-06 rule updates (long vowels and ɕː).
"""

import unicodedata as ud

import pytest

from turkic_api.core.translit import to_ipa

# -------------------------------------------------------------------------
# Orthographic word  →  IPA  (canonicalised)
# -------------------------------------------------------------------------
GOLD = {
    # monosyllabic roots (Table 3)
    "бал": "bɑl",
    "бел": "bel",
    "көл": "køl",
    "жыл": "ʒɯl",
    # disyllabic roots (Appendix)
    "молдо": "moldo",
    "илим": "ilim",
    "керме": "kerme",
    "кыргыз": "kɯrɡɯz",
    "сулуу": "suluː",  # ← long /uː/ from ‹уу›
    "үгүт": "yɡyt",
    # harmony alternations (Table 3)
    "балда": "bɑldɑ",
    "балды": "bɑldɯ",
    "көлдө": "køldø",
    "көлдү": "køldy",
    "жылда": "ʒɯldɑ",
    "жылды": "ʒɯldɯ",
}


def _canonical(ipa: str) -> str:
    """Normalise alternative glyphs to those emitted by ky_ipa.rules."""
    return (
        ipa.replace("ʤ", "dʒ")
        .replace("ʦ", "t͡s")
        .replace("ʧ", "t͡ʃ")
        .replace("q", "k")
        .replace("ʁ", "ɡ")
    )


@pytest.mark.parametrize(("cyr", "ipa"), GOLD.items())
def test_kyrgyz_word_to_ipa(cyr: str, ipa: str) -> None:
    predicted = _canonical(ud.normalize("NFC", to_ipa(cyr, "ky")))
    expected = _canonical(ipa)
    assert predicted == expected, f"{cyr} → {predicted!r}, expected {expected!r}"


def test_у_after_vowel_becomes_w() -> None:
    """Test context-sensitive у → w rule.

    In Kyrgyz, у is /u/ (vowel) normally, but /w/ (glide) after another vowel.
    Per McCollum (2020) consonant inventory: Glide = w, j
    """
    # у after vowel → w
    assert to_ipa("тау", "ky") == "tɑw"  # post-vocalic у = w
    assert to_ipa("бауыр", "ky") == "bɑwɯr"  # а + у = aw

    # у NOT after vowel → u
    assert to_ipa("ун", "ky") == "un"  # word-initial у = u
    assert to_ipa("бул", "ky") == "bul"  # after consonant у = u
    assert to_ipa("кул", "ky") == "kul"  # 'slave' from article
