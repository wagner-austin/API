"""Test Turkish → IPA transliteration using core to_ipa().

Source: Zimmer, K. & Orgun, O. (1992). "Turkish".
        Journal of the International Phonetic Association, 22(1-2): 43-45.
        DOI: https://doi.org/10.1017/S0025100300004588
"""

import pytest

# ---------------------------------------------------------------------------
# 1.  Single-letter gold standard
# Source: Zimmer, K. & Orgun, O. (1992)
GOLD = {
    "a": "a",
    "e": "e",
    "ı": "ɯ",
    "i": "i",
    "o": "o",
    "ö": "ø",
    "u": "u",
    "ü": "y",
    "b": "b",
    "c": "d͡ʒ",
    "ç": "t͡ʃ",
    "d": "d",
    "f": "f",
    "g": "ɡ",
    "ğ": "ː",  # default: length
    "h": "h",
    "j": "ʒ",
    "k": "k",
    "l": "l",
    "m": "m",
    "n": "n",
    "p": "p",
    "r": "ɾ",
    "s": "s",
    "ş": "ʃ",
    "t": "t",
    "v": "v",
    "y": "j",
    "z": "z",
}

# ---------------------------------------------------------------------------
# 2.  Context-sensitive expectations
CONTEXT_TESTS = {
    # Soft-g behaviour
    "ağ": "aː",
    "eğ": "eː",
    "ığ": "ɯː",
    "uğ": "uː",
    # Intervocalic soft-g (vowel length)
    "ağa": "aːa",
    "oğu": "oːu",
    "eği": "eːi",
    "öğü": "øːy",
}

# ---------------------------------------------------------------------------
# 3.  Parametrised tests


@pytest.mark.parametrize(("input", "expected"), GOLD.items())
def test_letter_to_ipa(input: str, expected: str) -> None:
    """Single Turkish letters."""
    from turkic_api.core.translit import to_ipa

    assert to_ipa(input, "tr") == expected


@pytest.mark.parametrize(("input", "expected"), CONTEXT_TESTS.items())
def test_context_sensitive_ipa(input: str, expected: str) -> None:
    """Context-sensitive Turkish rules."""
    from turkic_api.core.translit import to_ipa

    assert to_ipa(input, "tr") == expected


# ---------------------------------------------------------------------------
# 4.  Word-level sanity checks


def test_turkish_word_examples() -> None:
    """Common Turkish words."""
    from turkic_api.core.translit import to_ipa

    examples = {
        "merhaba": "meɾhaba",
        "teşekkür": "teʃekkyɾ",
        "güzel": "ɡyzel",
        "kitap": "kitap",
        "çok": "t͡ʃok",
        "değil": "deːil",
        "soğuk": "soːuk",
    }
    for tr, ipa in examples.items():
        assert to_ipa(tr, "tr") == ipa


def test_turkish_sentences() -> None:
    """Short sentences."""
    from turkic_api.core.translit import to_ipa

    assert to_ipa("Nasılsınız?", "tr") == "nasɯlsɯnɯz?"
    assert to_ipa("Günaydın", "tr") == "ɡynajdɯn"
    assert to_ipa("İyi günler", "tr") == "iji ɡynleɾ"
