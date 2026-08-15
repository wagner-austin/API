"""Turkish transliteration checked against the description the rules cite.

Every expectation here is a value Zimmer & Orgun print, with the page it
appears on. Where our output differs, it differs by a deviation the rule
file declares, and that deviation is applied to the published form
explicitly rather than being absorbed into a hand-edited expectation. A
reader can therefore see both what the source says and what this project
does about it.

The keyword lists are the right instrument for this. A passage
transcription records one speaker's performance, including variation and
error, which a deterministic rule set can neither reproduce nor should
try to. The keyword lists are statements about the mapping itself.
"""

from __future__ import annotations

import pytest

from turkic_api.core.rule_provenance import read_rule_source
from turkic_api.core.translit import _RULE_DIR, to_ipa

INHERITS_SOURCE = "https://doi.org/10.1017/S0025100300004588"

# Applied to the published transcription to obtain what these rules
# produce. Each entry is a simplification the rule file declares, not an
# adjustment made to force a match.
# The source's œ was read off a 1992 scan and confirmed against the
# page by a second reader, since the two glyphs are easy to confuse and
# enlarging a scan of that vintage does not settle it.
NOTATION: tuple[tuple[str, str, str], ...] = (
    ("œ", "ø", "the rules write the front rounded mid vowel with the close-mid symbol"),
    ("g", "ɡ", "the journal typesets the voiced velar plosive as U+0067; IPA is U+0261"),
)

DECLARED_SIMPLIFICATIONS: tuple[tuple[str, str, str], ...] = (
    ("ɫ", "l", "the rules carry one lateral; the velarized dental rule is not enabled"),
)

# orthography, Zimmer & Orgun's transcription, gloss, page
VOWEL_KEYWORDS: tuple[tuple[str, str, str, int], ...] = (
    ("kil", "kil", "clay", 44),
    ("kül", "kyl", "ashes", 44),
    ("kel", "kel", "bald", 44),
    ("göl", "gœl", "lake", 44),
    ("kal", "kaɫ", "stay", 44),
    ("kıl", "kɯɫ", "hair", 44),
    ("kul", "kuɫ", "slave", 44),
    ("kol", "koɫ", "arm", 44),
)

CONSONANT_KEYWORDS: tuple[tuple[str, str, str, int], ...] = (
    ("pul", "puɫ", "stamp", 43),
    ("bul", "buɫ", "find", 43),
    ("tel", "tel", "wire", 43),
    ("del", "del", "pierce", 43),
    ("kar", "kaɾ", "snow", 43),
    ("gam", "gam", "grief", 43),
    ("mal", "maɫ", "property", 43),
    ("nal", "naɫ", "horseshoe", 43),
    ("far", "faɾ", "headlight", 43),
    ("var", "vaɾ", "exists", 43),
    ("sar", "saɾ", "rap", 43),
    ("zar", "zaɾ", "membrane", 43),
    ("çam", "t͡ʃam", "pine", 43),
    ("cam", "d͡ʒam", "glass", 43),
    ("her", "heɾ", "every", 43),
    ("yer", "jeɾ", "place", 43),
    ("lâle", "laːle", "tulip", 43),
)

KEYWORDS = VOWEL_KEYWORDS + CONSONANT_KEYWORDS

# Soft g, per the same source (p. 44) and the Berkeley TELL sheet the
# rule file names beside it: lengthening in a coda, silence in an onset.
SOFT_G: tuple[tuple[str, str, str], ...] = (
    ("dağ", "daː", "word-final, so a coda"),
    ("yağmur", "jaːmuɾ", "before a consonant, so a coda"),
    ("iğne", "iːne", "before a consonant, so a coda"),
    ("ağaç", "aat͡ʃ", "between vowels, so an onset"),
    ("oğul", "oul", "between vowels, so an onset"),
    ("değer", "deeɾ", "between vowels, so an onset"),
)


def as_this_project_writes_it(published: str) -> str:
    """Apply the declared deviations to a published transcription.

    Args:
        published: The transcription exactly as the source prints it.

    Returns:
        The same transcription in the notation these rules produce.
    """
    for source_symbol, ours, _reason in (*NOTATION, *DECLARED_SIMPLIFICATIONS):
        published = published.replace(source_symbol, ours)
    return published


def test_the_declared_source_is_the_one_the_rules_cite() -> None:
    """These expectations come from the paper the rule file names.

    Without this the source above would be a comment: a test could claim
    any provenance it liked while the rules implemented another
    description entirely.
    """
    declared = read_rule_source(_RULE_DIR / "tr_ipa.rules")

    assert declared["identifier"] == INHERITS_SOURCE
    assert declared["authors"] == "Zimmer, K. & Orgun, O."
    assert declared["year"] == 1992


@pytest.mark.parametrize(("orthography", "published", "gloss", "page"), KEYWORDS)
def test_keyword_transliterates_as_the_source_prints_it(
    orthography: str, published: str, gloss: str, page: int
) -> None:
    """Each keyword matches the source, allowing the declared deviations."""
    assert page in {43, 44}
    assert gloss != ""
    assert to_ipa(orthography, "tr") == as_this_project_writes_it(published)


@pytest.mark.parametrize(("orthography", "expected", "environment"), SOFT_G)
def test_soft_g_follows_its_syllable_position(
    orthography: str, expected: str, environment: str
) -> None:
    """Soft g lengthens in a coda and is silent in an onset."""
    assert environment != ""
    assert to_ipa(orthography, "tr") == expected


def test_a_deviation_is_only_declared_when_it_is_real() -> None:
    """Every declared deviation changes at least one published keyword.

    A deviation nobody needs is a licence to differ from the source
    without saying why, so the list is held to the ones that do work.
    """
    for source_symbol, ours, reason in (*NOTATION, *DECLARED_SIMPLIFICATIONS):
        assert reason != ""
        assert any(source_symbol in published for _o, published, _g, _p in KEYWORDS), (
            f"declared deviation {source_symbol!r} -> {ours!r} applies to no keyword"
        )


def test_the_palatal_series_is_absent_as_the_rule_file_declares() -> None:
    """A front-vowel loanword gets the velar, not the palatal, stop.

    Zimmer & Orgun give /caɾ/ for kâr 'profit' against /kaɾ/ for kar
    'snow' (p. 44), a minimal pair these rules cannot express: they
    carry no palatal series, and they read the circumflex as length.
    Pinning the divergence keeps it visible rather than forgotten.
    """
    assert to_ipa("kâr", "tr") == "kaːɾ"
    assert to_ipa("kar", "tr") == "kaɾ"
