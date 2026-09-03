"""The pure-Python rule engine must reproduce ICU, output for output.

This service's methodology claim — that every corpus in the LSTM MI
experiments was produced by the same rule files — holds only if this
engine interprets those files the way ICU does. It replaces
``test_pyicu_parity.py``, which enforced the same invariant by importing
PyICU from a sibling checkout of ``turkic-transliteration``. That made the
test suite depend on another working copy being present and on ICU being
installed; both dependencies are gone, and the invariant is now enforced
against frozen digests instead. See ``tests/golden_sweep.py`` for how
those digests were measured.

Two layers, for two different failure modes:

* The **digests** cover 61,796 probes per commit and will catch any
  divergence at all, but say nothing about what diverged.
* The **spot checks** are hand-picked and name the semantics they protect,
  so a digest failure has somewhere to look. Each one is a case where a
  plausible misreading of the rule language gives a different answer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest

from tests.golden_sweep import SWEEP_DIGESTS, sweep_digest, sweep_probes
from turkic_api.core.rule_engine import apply_rules
from turkic_api.core.rule_parser import load_rules

_RULES_DIR: Final[Path] = (
    Path(__file__).resolve().parent.parent / "src" / "turkic_api" / "core" / "rules"
)

RULE_FILES: Final[tuple[str, ...]] = tuple(sorted(SWEEP_DIGESTS))

# Cases where a plausible misreading of ICU's rule language changes the
# answer. Every value was measured against PyICU 78.3: the ar_lat ones
# on 2026-08-15, the rest on 2026-08-14.
SPOT_CHECKS: Final[dict[str, dict[str, str]]] = {
    # у is [w] next to a vowel and [u] otherwise. The left context is
    # matched against the IPA already emitted, not against the Cyrillic
    # source, so 'ау' only works if the engine looks at its own output.
    "kk_ipa.rules": {
        "у": "u",
        "ау": "ɑw",
        "уа": "wɑ",
        "су": "su",
        "ус": "us",
        "екеуі": "ekewɪ",
        "тау": "tɑw",
        "ұу": "ʊw",
        "ы": "ə",
        "ж": "ʒ",
    },
    # Affricates are tie-barred, not precomposed ligatures, and ж is the
    # affricate rather than the fricative. Both were corrected upstream.
    "ky_ipa.rules": {
        "ж": "d͡ʒ",
        "дж": "d͡ʒ",
        "ч": "t͡ʃ",
        "ц": "t͡s",
        "е": "e",
        "ае": "ɑje",
        "се": "se",
        "у": "u",
        "ау": "ɑw",
        "су": "su",
        "ң": "ŋ",
    },
    # Four separate ng rules, each with a different context. 'ng' bare is
    # not the velar: it needs a vowel on both sides to geminate, a
    # consonant after to be plain [ŋ], and 'sh' is deliberately not [ʃ].
    "fi_ipa.rules": {
        "ng": "nɡ",
        "nk": "ŋk",
        "nga": "nɡɑ",
        "ngs": "ŋs",
        "gn": "ŋn",
        "kangas": "kɑŋːɑs",
        "renkaat": "reŋkɑːt",
        "sh": "sh",
        "'": "ʔ",
        "a'a": "ɑʔɑ",
    },
    # ğ lengthens a preceding vowel but is deleted before one, so 'ağa'
    # loses it entirely. The rule that used to delete the vowel too was
    # fixed upstream.
    "tr_ipa.rules": {
        "ğ": "ː",
        "ağa": "aa",
        "dağ": "daː",
        "ağ": "aː",
        "sağ": "saː",
        "yağmur": "jaːmuɾ",
        "al": "al",
        "el": "el",
    },
    # yo carries the open vowel, not the close one — the correction that
    # otherwise mistranscribes every 'yo' in the corpus. All six
    # apostrophe shapes are members of one set.
    "uz_ipa.rules": {
        "yo": "jɔ",
        "Yo": "jɔ",
        "YO": "jɔ",
        "yol": "jɔl",
        "oʻzbek": "ozbek",
        "o'": "o",
        "oʼ": "o",
        "o‘": "o",
        "sh": "ʃ",
        "ch": "t͡ʃ",
    },
    # е is iotated at the start of the text (a ^ anchor), after a vowel,
    # and after the two sign characters. ж is the affricate; the vendored
    # copy this replaced had it as voiceless d͡ʃ, which is not a sound.
    "uzc_ipa.rules": {
        "е": "je",
        "ае": "aje",
        "ъе": "ʔje",
        "ье": "ʲje",
        "ж": "d͡ʒ",
        "ё": "jɔ",
        "ы": "ɨ",
        "щ": "ʃː",
        "ў": "o",
        "ң": "ŋ",
    },
    "ru_ipa.rules": {
        "е": "je",
        "ъ": "",
        "ь": "ʲ",
        "ч": "t͡ʃʲ",
        "щ": "ʃʲː",
        "ы": "ɨ",
        "объект": "objekt",
        "ж": "ʒ",
        # Palatalisation, which the file was rewritten to mark rather than
        # spell as a glide, and the three contexts that decide it: a
        # pairable consonant takes ʲ, ц stays hard, and the soft sign keeps
        # both the ʲ and the /j/.
        "северный": "sʲevʲernɨj",
        "солнце": "solnt͡se",
        "статья": "statʲja",
        "жил": "ʒɨl",
    },
    "az_ipa.rules": {"ə": "æ", "c": "d͡ʒ", "ç": "t͡ʃ", "q": "ɡ", "x": "x", "ğ": "ɣ"},
    # The hamza carrier is the only rule in any vendored file whose match
    # depends on a negated set and therefore on a text boundary: it is
    # dropped before a vowel and written as an apostrophe elsewhere,
    # including at the end of a word. ء and ع were the quoting defect and
    # now emit the apostrophe the standard asks for.
    "ar_lat.rules": {
        "ب": "b",
        "ئ": "'",
        "ئا": "a",
        "بئا": "b'a",
        "بئ": "b'",
        "ائ": "a'",
        "ائا": "aa",
        "ء": "'",
        "ع": "'",
        "بعب": "b'b",
        "ې": "é",
        "ي": "y",
        "نگ": "n'g",
        "ڭ": "ng",
    },
    "ug_ipa.rules": {"ئا": "ɑ", "ا": "ɑ", "ې": "e", "ۇ": "u"},
    "kk_lat.rules": {
        "ә": "ä",
        "ғ": "ğ",
        "қ": "q",
        "ң": "ñ",
        "ө": "ö",
        "ұ": "ū",
        "ү": "ü",
        "ы": "y",
        "і": "ı",
    },
    "ky_lat.rules": {"ң": "ñ", "ө": "ö", "ү": "ü", "ы": "y", "ж": "j"},
    "tr_lat.rules": {"ı": "i", "İ": "I", "ş": "s", "ğ": "g", "ç": "c"},
}


@pytest.mark.parametrize("name", RULE_FILES)
def test_sweep_digest_matches_frozen_measurement(name: str) -> None:
    """The engine must reproduce what ICU produced for every probe.

    A failure here means one of three things, in order of likelihood: the
    rules were re-vendored without re-measuring the digests, the engine
    was changed and now reads some construct differently, or a rule file
    was edited in place. The spot-check test below will usually say which.
    """
    digest = sweep_digest(load_rules(name))
    assert digest == SWEEP_DIGESTS[name], (
        f"{name}: engine output no longer matches the frozen ICU measurement.\n"
        f"  expected sha256={SWEEP_DIGESTS[name]}\n"
        f"  actual   sha256={digest}\n"
        f"If the rules were re-vendored on purpose, re-measure per "
        f"docs/rule-engine-goldens.md; do not simply paste the new digest."
    )


@pytest.mark.parametrize("name", RULE_FILES)
def test_sweep_covers_every_rule(name: str) -> None:
    """Every rule must be reachable by at least one probe.

    A digest over probes that never exercise a rule certifies nothing
    about it. This asserts the probe set is not silently thin: each rule
    contributes probes built from its own window, so the count scales with
    the rules rather than staying fixed.
    """
    ruleset = load_rules(name)
    probes = sweep_probes(ruleset)
    assert len(probes) > len(ruleset.rules), (
        f"{name}: {len(probes)} probes for {len(ruleset.rules)} rules is too thin"
    )


def test_every_vendored_rule_file_has_a_frozen_digest() -> None:
    """A new rule file must arrive with a measurement, not without one.

    Without this, adding a rule file would silently add an unverified
    engine path — the file would be loaded in production and never
    compared against ICU.
    """
    on_disk = {path.name for path in _RULES_DIR.glob("*.rules")}
    assert on_disk == set(SWEEP_DIGESTS), (
        f"rule files and frozen digests disagree.\n"
        f"  files with no digest: {sorted(on_disk - set(SWEEP_DIGESTS))}\n"
        f"  digests with no file: {sorted(set(SWEEP_DIGESTS) - on_disk)}"
    )


@pytest.mark.parametrize("name", sorted(SPOT_CHECKS))
def test_spot_checks(name: str) -> None:
    """Named cases, each protecting one reading of the rule language."""
    ruleset = load_rules(name)
    for source, expected in SPOT_CHECKS[name].items():
        assert apply_rules(source, ruleset) == expected, (
            f"{name}: {source!r} -> {apply_rules(source, ruleset)!r}, expected {expected!r}"
        )


def test_spot_checks_cover_every_rule_file() -> None:
    """Every rule file needs named cases, not only a digest."""
    assert set(SPOT_CHECKS) == set(SWEEP_DIGESTS)


def test_the_arabic_quoting_defect_is_gone_from_the_vendored_copy() -> None:
    """``ar_lat.rules`` carried a defect that lost a rule, and no longer does.

    The file used to contain ``ء > ' ; ع > ' ;``. ICU reads ``'`` as a
    quote delimiter, so the two apostrophes bracketed a literal: ء mapped
    to the text `` ;   ع > `` and the rule for ع was swallowed into that
    literal and never existed, leaving ع untransliterated. This engine
    reproduced the defect faithfully, and a golden test pinned it, because
    repairing it here would have made this service disagree with the
    project that owns the rules.

    It was repaired upstream instead, in ``503d807``, by doubling the
    apostrophes — which is how the ICU syntax spells one. Both letters now
    emit an apostrophe and the old defective output appears nowhere.
    """
    ruleset = load_rules("ar_lat.rules")

    assert apply_rules("ء", ruleset) == "'"
    assert apply_rules("ع", ruleset) == "'"
    assert apply_rules("بعب", ruleset) == "b'b"
    assert " ;   ع > " not in apply_rules("ء", ruleset)


def test_a_word_final_hamza_carrier_is_written_and_an_intervocalic_one_is_not() -> None:
    """The one rule in any vendored file that depends on a text boundary.

    ``ئ } [^$AsuVowel]`` is the engine's only negated set, and the only
    place where a match turns on whether there is a character at all.
    Latin-Script Uyghur writes the hiatus as an apostrophe word-finally,
    where it stands for the etymological ﻉ or ﺀ of an Arabic loan, but
    leaves it unwritten between two vowels. Both readings are checked
    together, and once more with a following space, because an engine that
    treated the end of the string as a special case rather than the end of
    a word would pass the first and fail the last.
    """
    ruleset = load_rules("ar_lat.rules")

    assert apply_rules("ۋۇقۇئ", ruleset) == "wuqu'"
    assert apply_rules("ۋۇقۇئ ب", ruleset) == "wuqu' b"
    assert apply_rules("سائەت", ruleset) == "saet"
    assert apply_rules("قۇرئان", ruleset) == "qur'an"
