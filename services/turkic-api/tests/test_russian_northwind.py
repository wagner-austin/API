"""
North Wind & Sun passage in Russian.

The orthographic text is the article's own (p. 226), and the expected
transcriptions below are read from the article's broad transcription of that
same passage, not produced by running the rules over it. An expectation taken
from the transliterator certifies the transliterator against itself, which is
how ``solnt͡sje`` and ``sjevjernɨj`` sat here as gold values while both were
wrong.

Source: Yanushevskaya, I. & Bunčić, D. (2015). "Russian".
        Journal of the International Phonetic Association, 45(2): 221-228.
        DOI: https://doi.org/10.1017/S0025100314000395

Note: The article's transcription includes stress marks and phonological
processes (vowel reduction, etc.) that our simplified orthographic rules
do not capture. This test validates the basic grapheme-to-phoneme mapping.
"""

from unicodedata import category, normalize


def strip_combining(s: str) -> str:
    """Remove combining marks (stress, length, etc.) for comparison."""
    return "".join(c for c in normalize("NFD", s) if category(c) != "Mn")


# Full orthographic text from the JIPA article (p. 226)
# fmt: off
ORTHOGRAPHIC = (
    "Однажды северный ветер и солнце поспорили, кто из них сильнее. "
    "Как раз в это время они заметили закутанного в плащ путника, "
    "который шёл по дороге, и решили, что тот из них будет считаться "
    "самым сильным, кому раньше удастся заставить путника снять плащ. "
    "Тут северный ветер принялся дуть изо всех сил; но чем сильнее он дул, "
    "тем сильнее кутался путник в свой плащ, так что в конце концов "
    "северный ветер должен был отказаться от своей затеи. "
    "Тогда засияло солнышко, путник понемногу отогрелся и вскоре снял свой плащ. "
    "Таким образом, северный ветер вынужден был признать, что солнце сильнее его."
)
# fmt: on

# Individual sentences for granular testing
SENTENCES = [
    "Однажды северный ветер и солнце поспорили, кто из них сильнее.",
    (
        "Как раз в это время они заметили закутанного в плащ путника, "
        "который шёл по дороге, и решили, что тот из них будет считаться "
        "самым сильным, кому раньше удастся заставить путника снять плащ."
    ),
    (
        "Тут северный ветер принялся дуть изо всех сил; но чем сильнее он дул, "
        "тем сильнее кутался путник в свой плащ, так что в конце концов "
        "северный ветер должен был отказаться от своей затеи."
    ),
    "Тогда засияло солнышко, путник понемногу отогрелся и вскоре снял свой плащ.",
    "Таким образом, северный ветер вынужден был признать, что солнце сильнее его.",
]


def test_northwind_deterministic() -> None:
    """Verify that the transcription is deterministic (same input → same output)."""
    from turkic_api.core.translit import to_ipa

    # Run twice and compare
    result1 = strip_combining(to_ipa(ORTHOGRAPHIC, "ru"))
    result2 = strip_combining(to_ipa(ORTHOGRAPHIC, "ru"))
    assert result1 == result2


def test_northwind_sentences() -> None:
    """Transcribe each sentence and verify basic properties."""
    from turkic_api.core.translit import to_ipa

    for sent in SENTENCES:
        ipa = to_ipa(sent, "ru")
        # Should not contain any Cyrillic letters
        for char in ipa:
            assert not ("\u0400" <= char <= "\u04ff"), f"Cyrillic {char!r} in output"
        # Should produce non-empty output (input has content)
        assert ipa, f"Empty IPA output for: {sent[:30]!r}..."


def test_northwind_key_words() -> None:
    """Check specific words from the passage against expected IPA."""
    from turkic_api.core.translit import to_ipa

    # Key words from the passage. Every consonant before an iotated vowel
    # carries ʲ and no glide, which is how the Illustration's own broad
    # transcription of this passage prints them (p. 226): ˈsʲevʲirnɨj,
    # ˈvʲetʲir, sʲiˈlʲnʲeji. The remaining difference from the published
    # forms is vowel reduction, which the rules declare out of scope.
    #
    # These previously read sjevjernɨj, vjetjer, solnt͡sje and silʲnjeje,
    # taken from what the rules then produced rather than from the article,
    # so they pinned the defect instead of catching it.
    key_words = {
        "северный": "sʲevʲernɨj",  # 'northern'
        "ветер": "vʲetʲer",  # 'wind'
        "солнце": "solnt͡se",  # 'sun', ц is unpaired hard
        "сильнее": "sʲilʲnʲeje",  # 'stronger'
        "плащ": "plaʃʲː",  # 'cloak'
        "путник": "putnʲik",  # 'traveler'
        "дуть": "dutʲ",  # 'to blow'
        "снять": "snʲatʲ",  # 'to take off'
    }

    for word, expected in key_words.items():
        actual = to_ipa(word, "ru")
        assert actual == expected, f"{word!r}: expected {expected!r}, got {actual!r}"
