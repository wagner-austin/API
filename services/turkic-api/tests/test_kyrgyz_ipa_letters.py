"""Kyrgyz transliteration checked against the description the rules cite.

McCollum 2020 is a research article rather than an Illustration, so it
carries no keyword chart. What it does carry is Table 1 (the consonant
inventory), Table 2 (the vowel inventory) and Table 3, which gives eight
roots in IPA alongside their suffixed forms. Those eight are the
expectations here, written as the article prints them.

An earlier version of this file cited a Wikipedia page rather than the
article the rule file names, which is why a mismatch in <ж> went
unnoticed for so long: the rules gave the letter its Russian value, a
fricative, where Table 3 prints the affricate in both roots it supplies.
Nothing internal could catch that, because the fricative is a real Kyrgyz
phoneme and so passes an inventory check; only the source settles it.
"""

from __future__ import annotations

import pytest

from turkic_api.core.rule_provenance import read_rule_source
from turkic_api.core.translit import _RULE_DIR, to_ipa

INHERITS_SOURCE = "https://doi.org/10.5334/labphon.247"

# The article prints affricates with the precomposed ligatures, which the
# IPA has since withdrawn. These rules use the tie bar, as the other six
# rule files do, so that one phoneme is the same number of characters in
# every language a character-level model compares. Notation, not phonology.
NOTATION: tuple[tuple[str, str, str], ...] = (
    ("ʤ", "d͡ʒ", "U+02A4 is withdrawn from the IPA; the tie bar is current"),
)

# A simplification the rules make deliberately, applied to the published
# form to obtain what they produce.
DECLARED_SIMPLIFICATIONS: tuple[tuple[str, str, str], ...] = (
    ("q", "k", "the rules carry no dorsal backness allophony; Table 1 lists k and q separately"),
)

# Cyrillic, the root exactly as Table 3 prints it, gloss. The article is
# numbered rather than paginated, so Table 3 is the locator.
TABLE_3_ROOTS: tuple[tuple[str, str, str], ...] = (
    ("тил", "til", "language"),
    ("бел", "bel", "lower back"),
    ("гүл", "ɡyl", "flower"),
    ("көл", "køl", "lake"),
    ("бал", "bɑl", "honey"),
    ("кул", "qul", "slave"),
    ("жыл", "ʤɯl", "year"),
    ("жол", "ʤol", "road"),
)


def as_this_project_writes_it(published: str) -> str:
    """Rewrite a published transcription in the notation these rules use.

    Args:
        published: The transcription exactly as Table 3 prints it.

    Returns:
        The same transcription in the glyphs and the level of detail
        these rules emit.
    """
    for source_symbol, ours, _reason in (*NOTATION, *DECLARED_SIMPLIFICATIONS):
        published = published.replace(source_symbol, ours)
    return published


def test_the_declared_source_is_the_one_the_rules_cite() -> None:
    """These expectations come from the article the rule file names."""
    declared = read_rule_source(_RULE_DIR / "ky_ipa.rules")

    assert declared["identifier"] == INHERITS_SOURCE
    assert declared["authors"] == "McCollum, A. G."
    assert declared["year"] == 2020


@pytest.mark.parametrize(("cyrillic", "published", "gloss"), TABLE_3_ROOTS)
def test_table_3_root_transliterates_as_the_source_prints_it(
    cyrillic: str, published: str, gloss: str
) -> None:
    """Each of the eight roots matches, under the declared differences.

    Args:
        cyrillic: The root in Kyrgyz orthography.
        published: The root as Table 3 prints it.
        gloss: The gloss Table 3 gives, to identify the row on failure.
    """
    assert to_ipa(cyrillic, "ky") == as_this_project_writes_it(published), (
        f"{cyrillic} {gloss!r}: Table 3 prints {published!r}"
    )


def test_zhe_is_the_affricate_the_source_gives_not_the_russian_fricative() -> None:
    """The letter carries its Kyrgyz value rather than its Russian one.

    Table 3 supplies two roots containing <ж> and prints the affricate in
    both. Table 1 does list the fricative, but as a phoneme of the
    language rather than as the value of this letter, and mapping the
    letter onto it reproduces the Russian orthography instead of the
    Kyrgyz one. Kazakh is the language where the fricative is right, so
    the error also made these two look alike on a segment that separates
    them.
    """
    assert to_ipa("жол", "ky") == "d͡ʒol"
    assert to_ipa("жыл", "ky") == "d͡ʒɯl"

    everyday_words = to_ipa("жана же жаткан жакшы", "ky")
    assert "ʒ" not in everyday_words.replace("d͡ʒ", ""), (
        f"a bare fricative survives in {everyday_words!r}"
    )


def test_the_digraph_spelling_gives_the_same_phoneme_once() -> None:
    """<дж> is a variant spelling of <ж>, not d followed by it.

    The digraph rule has to be tried before the letter rule; if the
    letter rule wins, the digraph surfaces as d plus the affricate.
    """
    assert to_ipa("дж", "ky") == "d͡ʒ"
    assert to_ipa("дждж", "ky") == "d͡ʒd͡ʒ"


def test_affricates_are_written_with_a_tie_bar_not_a_ligature() -> None:
    """Kyrgyz writes its affricates the way the other six languages do.

    These rules used the withdrawn precomposed ligatures while every
    other rule file used a tie bar, so one phoneme was one character in
    Kyrgyz and three elsewhere. A character-level model comparing
    languages reads that as a difference between the languages.
    """
    assert to_ipa("чын", "ky") == "t͡ʃɯn"
    for ligature in ("ʧ", "ʤ", "ʥ", "ʨ"):
        assert ligature not in to_ipa("чын жол", "ky")
