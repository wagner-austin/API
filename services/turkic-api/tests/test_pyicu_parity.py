"""Parity between ``turkic-api``'s pure-Python rule engine and PyICU.

This service reimplements the transliteration engine in pure Python so
that deployment does not require the C++ ICU library. The paper's
methodology claim ("every corpus in the LSTM MI experiments was
produced by the same rule files") holds only if both engines produce
identical output on identical input for the shared rule files.

This test enforces the invariant in two layers:

1. **File-level byte identity** — every IPA ``.rules`` file present in
   both projects must match byte-for-byte. A drift here means the two
   projects have started diverging at the source-of-truth level and
   the paper's claim is at risk.

2. **Engine-level output identity** — for each IPA language, run the
   same realistic input through both engines and assert bit-identical
   output. This is what actually validates that the pure-Python
   engine correctly interprets ICU rule semantics.

The Latin ``.rules`` files (``ar_lat``, ``kk_lat``, ``ky_lat``,
``tr_lat``) intentionally differ between projects: ``turkic-api``
ships extended variants that ``turkic-transliteration`` has not yet
adopted. Latin parity is therefore out of scope; the Latin
divergence is documented as a stated limitation and only IPA is
compared.

For Uzbek the two engines produce identical output on pure-Latin
input because ``turkic-api``'s post-pass through ``uzc_ipa.rules``
is a no-op on strings that contain no Cyrillic characters. Mixed-
script Uzbek is a documented turkic-api-only feature.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# Two package roots, resolved from this test file's location. The
# turkic-api monorepo layout keeps sibling PROJECTS/ directories two
# levels above ``services/turkic-api``, so ``PROJECTS/turkic-
# transliteration`` is three ``.parent`` steps up from this file's
# directory.
_TESTS_DIR: Path = Path(__file__).resolve().parent
_API_ROOT: Path = _TESTS_DIR.parent
_MONOREPO_ROOT: Path = _API_ROOT.parent.parent
_PROJECTS_ROOT: Path = _MONOREPO_ROOT.parent
_TURKIC_TRANSLIT_ROOT: Path = _PROJECTS_ROOT / "turkic-transliteration"

_API_RULES_DIR: Path = _API_ROOT / "src" / "turkic_api" / "core" / "rules"
_LIB_RULES_DIR: Path = _TURKIC_TRANSLIT_ROOT / "src" / "turkic_translit" / "rules"

# IPA languages present in both projects. Kept as a tuple so the
# ordering — and therefore the test-parameter names emitted by pytest
# — is stable across runs.
IPA_LANGUAGES: tuple[str, ...] = ("az", "fi", "kk", "ky", "tr", "ug", "uz")

# One short but non-trivial input per language. Each sample exercises
# multi-word context, whitespace, and punctuation so a rule-ordering
# or context-handling bug in either engine surfaces here.
PARITY_SAMPLES: dict[str, str] = {
    "az": "Şimal yeli ilə Günəş mübahisə edirdilər ki, hansı daha güclüdür.",
    "fi": "Oi maamme, Suomi, synnyinmaa, soi, sana kultainen!",
    "kk": (
        "Бір күні солтүстік жел мен күн екеуі араларында кім мықты екенін шеше алмай бәсікелеседі."
    ),
    "ky": ("Түндүк шамалы менен күн кайсынысы күчтүүрөөк экени тууралуу талашышып жатышты."),
    "tr": ("Kuzey rüzgarı ile güneş, hangisinin daha güçlü olduğu konusunda tartışıyorlardı."),
    "ug": "شىمالىي شامال بىلەن قۇياش ئۆزئارا كۈچلۈك دەپ تالىشىۋاتاتتى.",
    # Uzbek sample: pure Latin. Cyrillic-in-Latin would trigger the
    # uzc_ipa post-pass that only turkic-api applies (documented
    # divergence).
    "uz": (
        "Bir kun shimoliy shamol va quyosh qaysi biri kuchliroq "
        "ekanligi oʻrtasida tortishib qolishibdi."
    ),
}


def _file_hash(path: Path) -> str:
    """Return the SHA-256 hex digest of ``path``'s bytes.

    Args:
        path: The file to hash.

    Returns:
        The hex-encoded SHA-256 digest, computed over the raw file
        bytes (no encoding conversion).

    Raises:
        FileNotFoundError: When ``path`` does not exist.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_shared_ipa_rule_files_are_byte_identical() -> None:
    """Every IPA ``.rules`` file must match byte-for-byte across projects.

    A drift here means the two projects have started using different
    versions of the shared rule files, which invalidates the paper's
    "same rules everywhere" claim before we even get to engine
    behaviour. Fails loud with a per-language diff hint.
    """
    for lang in IPA_LANGUAGES:
        rule_filename = f"{lang}_ipa.rules"
        api_path = _API_RULES_DIR / rule_filename
        lib_path = _LIB_RULES_DIR / rule_filename
        assert api_path.is_file(), f"turkic-api missing {rule_filename}"
        assert lib_path.is_file(), f"turkic-transliteration missing {rule_filename}"
        api_hash = _file_hash(api_path)
        lib_hash = _file_hash(lib_path)
        assert api_hash == lib_hash, (
            f"{rule_filename} diverged between projects: "
            f"turkic-api sha256={api_hash[:16]}..., "
            f"turkic-transliteration sha256={lib_hash[:16]}..."
        )


def test_uzc_ipa_shared_with_library() -> None:
    """The Uzbek-Cyrillic rules file must also match across projects.

    Even though only turkic-api runs ``uzc_ipa.rules`` as a post-pass,
    both projects ship the file so that either engine can be pointed
    at Cyrillic-Uzbek input. Byte-identity keeps the post-pass a
    faithful subset of the library's rule catalogue.
    """
    api_path = _API_RULES_DIR / "uzc_ipa.rules"
    lib_path = _LIB_RULES_DIR / "uzc_ipa.rules"
    assert api_path.is_file(), "turkic-api missing uzc_ipa.rules"
    assert lib_path.is_file(), "turkic-transliteration missing uzc_ipa.rules"
    assert _file_hash(api_path) == _file_hash(lib_path)


def test_ipa_engine_output_matches_pyicu_for_every_language() -> None:
    """Both engines must produce identical IPA on identical input.

    Imports ``turkic_translit.core`` (PyICU-backed) at call time so a
    missing PyICU wheel produces a clear ImportError from the test
    body rather than a collection-time failure that hides the real
    diagnostic.
    """
    from turkic_translit.core import to_ipa as lib_to_ipa

    from turkic_api.core.translit import to_ipa as api_to_ipa

    for lang in IPA_LANGUAGES:
        sample = PARITY_SAMPLES[lang]
        api_output = api_to_ipa(sample, lang)
        lib_output = lib_to_ipa(sample, lang)
        assert api_output == lib_output, (
            f"engine divergence on {lang!r}\n"
            f"  input:  {sample!r}\n"
            f"  api:    {api_output!r}\n"
            f"  pyicu:  {lib_output!r}"
        )
