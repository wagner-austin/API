"""Finnish transliteration checked against the description the rules cite.

Suomi, Toivanen and Ylitalo devote a chapter to the relationship between
sound structure and orthography, so unlike the JIPA Illustrations this
source states grapheme-to-phoneme correspondences directly rather than
only printing keywords. Three kinds of evidence are used here, all of
them printed in the book:

* the quantity series on p. 19, where each orthographic form is given
  with its phonemic transcription;
* the series of eight word forms on p. 20 that runs through the eight
  vowel phonemes in order;
* the velar nasal, which p. 141 calls the one fully systematic exception
  to the one-grapheme-one-phoneme principle, together with the two words
  the phonotactics chapter transcribes in brackets on pp. 57-58.

The book writes a long segment as a sequence of two identical phonemes
and marks the dental place on the coronal plosives; these rules use the
length mark and leave the place unmarked. Both differences are declared
below and applied to the published transcription before comparison, so
that what is compared is the same claim in two notations.
"""

from __future__ import annotations

import pytest

from turkic_api.core.rule_provenance import read_rule_source
from turkic_api.core.translit import _RULE_DIR, to_ipa

INHERITS_SOURCE = "https://urn.fi/URN:ISBN:9789514289842"

DENTAL = "̪"
LENGTH_MARK = "ː"

# Applied to the published transcription to obtain what these rules
# produce.
NOTATION: tuple[tuple[str, str, str], ...] = (
    (DENTAL, "", "the source marks /t̪ d̪/ dental; these rules leave the place off, p. 25"),
)

# p. 19: the quantity series, each orthographic form printed with its
# phonemic transcription. Syllable dots are dropped from the orthography
# as the book itself does when the word is not being syllabified.
QUANTITY_SERIES: tuple[tuple[str, str], ...] = (
    ("taka", "t̪ɑkɑ"),
    ("taaka", "t̪ɑɑkɑ"),
    ("takka", "t̪ɑkkɑ"),
    ("taakka", "t̪ɑɑkkɑ"),
    ("takaa", "t̪ɑkɑɑ"),
    ("taakaa", "t̪ɑɑkɑɑ"),
    ("taakkaa", "t̪ɑɑkkɑɑ"),
    ("taika", "t̪ɑikɑ"),
)

# p. 20: "the eight vowel phonemes could be given as /i/, /e/, /y/, /ø/,
# /æ/, /ɑ/, /o/ and /u/ ... They occur e.g. in the series of word forms
# mikin - mekin - mykin - mökin - mäkin - makin - mokin - mukin". Each
# gloss is the one the same paragraph supplies.
VOWEL_SERIES: tuple[tuple[str, str, str], ...] = (
    ("mikin", "i", "genitive singular of Mikki"),
    ("mekin", "e", "me 'we' + kin 'also'"),
    ("mykin", "y", "plural instructive of mykkä 'dumb'"),
    ("mökin", "ø", "genitive singular of mökki 'cottage'"),
    ("mäkin", "æ", "genitive singular of Mäkki"),
    ("makin", "ɑ", "genitive singular of maki 'lemur'"),
    ("mokin", "o", "plural instructive of mokka 'suede'"),
    ("mukin", "u", "genitive singular of muki 'mug'"),
)

# pp. 57-58: words the phonotactics chapter prints in phonetic brackets.
# These are the two environments in which the orthographic sequences <ng>
# and <gn> do not carry the geminate.
TRANSCRIBED_IN_BRACKETS: tuple[tuple[str, str, int], ...] = (
    ("Englanti", "eŋlɑnt̪i", 57),
    ("kognitio", "koŋnit̪io", 57),
    ("magneetti", "mɑŋneet̪ːi", 58),
)

# p. 141: the four statements of the orthography chapter about /ŋ/,
# with the example word each is given with. The expected form of each
# word follows from those statements plus the one-to-one map that the
# same book states holds elsewhere (p. 20).
VELAR_NASAL: tuple[tuple[str, str, str], ...] = (
    ("kenkä", "keŋkæ", "<n> before <k> is /ŋ/"),
    ("kengän", "keŋːæn", "<ng> between vowels is /ŋŋ/"),
    ("tango", "tɑŋːo", "<ng> between vowels is /ŋŋ/"),
)

# p. 57, restriction 5: an obstruent may be followed by /h/ across a
# morpheme boundary. Finnish compounds put an -s against an h-initial
# second member constantly, so a <sh> digraph rule would delete the
# sibilant from all of these.
S_BEFORE_H: tuple[tuple[str, str, str], ...] = (
    ("keskushallinto", "keskushɑlːinto", "keskus + hallinto, 'central administration'"),
    ("kuningashuone", "kuniŋːɑshuone", "kuningas + huone, 'royal house'"),
    ("rakennushanke", "rɑkenːushɑŋke", "rakennus + hanke, 'building project'"),
)


def as_this_project_writes_it(published: str) -> str:
    """Apply the declared deviations to a published transcription.

    Args:
        published: The transcription exactly as the source prints it.

    Returns:
        The same transcription in the notation these rules produce, with
        the dental mark dropped and each doubled segment rewritten with
        the length mark.
    """
    for source_symbol, ours, _reason in NOTATION:
        published = published.replace(source_symbol, ours)
    collapsed: list[str] = []
    for symbol in published:
        if collapsed and symbol == collapsed[-1]:
            collapsed.append(LENGTH_MARK)
        else:
            collapsed.append(symbol)
    return "".join(collapsed)


def test_the_declared_source_is_the_one_the_rules_cite() -> None:
    """These expectations come from the book the rule file names."""
    declared = read_rule_source(_RULE_DIR / "fi_ipa.rules")

    assert declared["identifier"] == INHERITS_SOURCE
    assert declared["authors"] == "Suomi, K., Toivanen, J. & Ylitalo, R."
    assert declared["year"] == 2008


def test_the_notation_rewrite_does_what_it_claims() -> None:
    """A doubled segment becomes the length mark and nothing else moves.

    The comparison below is only as trustworthy as this rewrite, so it
    is stated on the source's own example rather than left implicit.
    """
    assert as_this_project_writes_it("t̪ɑɑkkɑ") == "tɑːkːɑ"
    assert as_this_project_writes_it("t̪ɑkɑ") == "tɑkɑ"
    assert as_this_project_writes_it("t̪ɑikɑ") == "tɑikɑ"


@pytest.mark.parametrize(("orthography", "published"), QUANTITY_SERIES)
def test_quantity_series_transliterates_as_the_source_prints_it(
    orthography: str, published: str
) -> None:
    """Each form of the p. 19 series matches its published transcription."""
    assert to_ipa(orthography, "fi") == as_this_project_writes_it(published)


@pytest.mark.parametrize(("orthography", "vowel", "gloss"), VOWEL_SERIES)
def test_each_word_of_the_series_carries_its_vowel(
    orthography: str, vowel: str, gloss: str
) -> None:
    """The series runs through the eight vowel phonemes in order.

    The book gives the words and the phonemes but not full
    transcriptions, so what is checked is the first-syllable vowel, which
    is the segment the series exists to contrast.
    """
    assert gloss != ""
    produced = to_ipa(orthography, "fi")

    assert produced[1] == vowel
    assert produced.startswith("m")


def test_the_series_covers_the_whole_vowel_inventory() -> None:
    """Eight words, eight distinct vowels, the inventory of p. 20."""
    vowels = {vowel for _orthography, vowel, _gloss in VOWEL_SERIES}

    assert vowels == set("ieyøæɑou")
    assert len(VOWEL_SERIES) == len(vowels)


@pytest.mark.parametrize(("orthography", "published", "page"), TRANSCRIBED_IN_BRACKETS)
def test_bracketed_word_transliterates_as_the_source_prints_it(
    orthography: str, published: str, page: int
) -> None:
    """Words the phonotactics chapter transcribes match segment for segment.

    Englanti and kognitio are the source's own counterexamples to a naive
    reading of the orthography chapter: <ng> is not always the geminate,
    and <gn> is not /ɡn/ at all. The rules used to produce eŋːlɑnti and
    koɡnitio.
    """
    assert page in {57, 58}
    assert to_ipa(orthography, "fi") == as_this_project_writes_it(published)


@pytest.mark.parametrize(("orthography", "expected", "statement"), VELAR_NASAL)
def test_velar_nasal_follows_the_orthography_chapter(
    orthography: str, expected: str, statement: str
) -> None:
    """The Latin alphabet has no letter for /ŋ/, so <n> and <ng> carry it."""
    assert statement != ""
    assert to_ipa(orthography, "fi") == expected


def test_the_two_lengths_of_the_velar_nasal_stay_apart() -> None:
    """Between vowels the nasal is long, before a consonant it is short.

    Stated separately because the rules formerly wrote the geminate in
    both environments, and a per-word check alone would not say that the
    distinction is the point.
    """
    assert to_ipa("tango", "fi").count(LENGTH_MARK) == 1
    assert LENGTH_MARK not in to_ipa("Englanti", "fi")


@pytest.mark.parametrize(("orthography", "expected", "gloss"), S_BEFORE_H)
def test_sibilant_survives_before_h_in_a_compound(
    orthography: str, expected: str, gloss: str
) -> None:
    """A compound's -s is not swallowed into a postalveolar fricative."""
    assert gloss != ""
    assert to_ipa(orthography, "fi") == expected


def test_the_postalveolar_fricative_needs_its_own_grapheme() -> None:
    """/ʃ/ comes from <š>, the grapheme the source gives it (pp. 25, 141)."""
    assert to_ipa("šampoo", "fi") == "ʃɑmpoː"
    assert "ʃ" not in to_ipa("shampoo", "fi")


def test_v_and_w_are_not_distinguished() -> None:
    """The graphemes "are non-distinct in Finnish" (p. 142).

    Both are the labiodental approximant of group (1), not the fricative
    the letter <v> suggests.
    """
    assert to_ipa("vaha", "fi") == to_ipa("waha", "fi") == "ʋɑhɑ"
