"""Cyrillic Uzbek checked against the same description as the Latin rules.

Ido prints the Latin orthography, so the Cyrillic spellings here are the
standard correspondences of those same words. That inference is not
taken on trust: a wrong spelling produces a different transcription and
fails, which is how the affricate defect below was found.

The deviations are the same six the Latin file declares, since the two
rule sets transcribe one language in its two alphabets and must agree.
"""

from __future__ import annotations

import pytest

from turkic_api.core.rule_provenance import read_rule_source
from turkic_api.core.translit import _RULE_DIR, to_ipa

INHERITS_SOURCE = "https://doi.org/10.1017/S0025100324000148"

NOTATION: tuple[tuple[str, str, str], ...] = (
    ("d͡ʑ", "d͡ʒ", "the source writes the alveolo-palatal affricate, p. 153"),
    ("t͡ɕ", "t͡ʃ", "the source writes the alveolo-palatal affricate, p. 153"),
    ("ɕ", "ʃ", "the source writes the alveolo-palatal fricative, p. 153"),
    ("ɾ", "r", "the source writes the rhotic as a tap, p. 153"),
    ("χ", "x", "the source writes the voiceless fricative as uvular, p. 153"),
    ("g", "ɡ", "the journal typesets the voiced velar plosive as U+0067; IPA is U+0261"),
)

# Cyrillic, the Latin spelling Ido prints, his transcription, gloss, page
KEYWORDS: tuple[tuple[str, str, str, str, int], ...] = (
    ("пеш", "pesh", "peɕ", "front", 154),
    ("беш", "besh", "beɕ", "five", 154),
    ("тор", "tor", "tɔɾ", "narrow", 154),
    ("дор", "dor", "dɔɾ", "rope", 154),
    ("сол", "sol", "sɔl", "raft", 154),
    ("зол", "zol", "zɔl", "adept", 154),
    ("шол", "shol", "ɕɔl", "woolen fabric", 154),
    ("мол", "mol", "mɔl", "livestock", 154),
    ("нол", "nol", "nɔl", "zero", 154),
    ("чин", "chin", "t͡ɕin", "genuine", 154),
    ("жин", "jin", "d͡ʑin", "genie", 154),
    ("кўр", "ko'r", "koɾ", "blind", 154),
    ("гўр", "go'r", "goɾ", "tomb", 154),
    ("қўр", "qo'r", "qoɾ", "coal", 154),
    ("лол", "lol", "lɔl", "speechless", 154),
    ("ёл", "yol", "jɔl", "mane", 154),
    ("рол", "rol", "ɾɔl", "role", 154),
    ("хам", "xam", "χam", "adroop", 154),
    ("ғам", "g'am", "ʁam", "grief", 154),
    ("ҳам", "ham", "ham", "also", 154),
)


def as_this_project_writes_it(published: str) -> str:
    """Apply the declared deviations to a published transcription.

    Args:
        published: The transcription exactly as the source prints it.

    Returns:
        The same transcription in the notation these rules produce.
    """
    for source_symbol, ours, _reason in NOTATION:
        published = published.replace(source_symbol, ours)
    return published


def test_the_declared_source_is_the_one_the_rules_cite() -> None:
    """These expectations come from the paper the rule file names."""
    declared = read_rule_source(_RULE_DIR / "uzc_ipa.rules")

    assert declared["identifier"] == INHERITS_SOURCE
    assert declared["authors"] == "Ido, S."


@pytest.mark.parametrize(("cyrillic", "latin", "published", "gloss", "page"), KEYWORDS)
def test_keyword_transliterates_as_the_source_prints_it(
    cyrillic: str, latin: str, published: str, gloss: str, page: int
) -> None:
    """Each keyword matches the source, allowing the declared deviations."""
    assert page == 154
    assert gloss != ""
    assert latin != ""
    assert to_ipa(cyrillic, "uzc") == as_this_project_writes_it(published)


@pytest.mark.parametrize(("cyrillic", "latin", "published", "gloss", "page"), KEYWORDS)
def test_the_two_alphabets_agree(
    cyrillic: str, latin: str, published: str, gloss: str, page: int
) -> None:
    """One word transcribes the same whichever alphabet it is written in.

    The two rule sets describe one language, so a word that differs
    between them means one of the two is wrong. This is what exposed the
    Cyrillic affricate being written with a voiceless second element,
    which is not a producible segment and which the Latin rules never
    had.
    """
    assert page == 154
    assert gloss != ""
    assert published != ""
    assert to_ipa(cyrillic, "uzc") == to_ipa(latin, "uz")
