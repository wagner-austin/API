"""Transliteration by language, over the vendored rule files.

The interpreter beneath this — :mod:`turkic_api.core.rule_lexer`,
:mod:`turkic_api.core.rule_parser` and :mod:`turkic_api.core.rule_engine` —
reads one rule file. This module is the layer above: it decides which rule file
a language and output format need, caches the parsed result, and composes the
two-stage cases.

What a language supports is derived from the files on disk rather than
declared in a list, so a rule file that is added or removed cannot disagree
with a table describing it. That is also why nothing here checks whether a
rule file exists before loading it: the format is only reported as supported
because its file was found.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from turkic_api.core.rule_engine import apply_rules
from turkic_api.core.rule_parser import RuleSet, load_rules

_RULE_DIR: Final[Path] = Path(__file__).with_suffix("").parent / "rules"

LATIN_FORMAT: Final = "latin"
IPA_FORMAT: Final = "ipa"

# The file-name suffix that provides each format. One spelling each: a rule
# file is named ``{language}_{suffix}.rules``.
_SUFFIXES: Final[dict[str, str]] = {LATIN_FORMAT: "lat", IPA_FORMAT: "ipa"}

# Uyghur Arabic script, transliterated to Latin before a Latin rule file runs.
ARABIC_PREPASS: Final = "ar_lat.rules"

# Uzbek is written in both scripts, and OSCAR's Uzbek is overwhelmingly Latin
# with a small Cyrillic remainder. Running the Cyrillic rules over the Latin
# result picks that remainder up; it is a no-op on text with no Cyrillic in it,
# because every rule in that file matches a Cyrillic character.
UZBEK_CYRILLIC_PASS: Final = "uzc_ipa.rules"

ERR_UNSUPPORTED_FORMAT: Final = "TURKIC_TRANSLIT_001_UNSUPPORTED_FORMAT"

_SUPPORTED_LANGUAGES: dict[str, list[str]] | None = None
_RULE_CACHE: dict[str, RuleSet] = {}


class UnsupportedTransliterationError(ValueError):
    """Raised when no rule file provides a format for a language.

    Args:
        language (str): The language code asked for.
        output_format (str): The format asked for, ``latin`` or ``ipa``.
        available (tuple[str, ...]): Every language that does provide that
            format, so the caller can see what it could have asked for.
    """

    def __init__(self, language: str, output_format: str, available: tuple[str, ...]) -> None:
        """Render the code, the rejected request, and the alternatives."""
        super().__init__(
            f"{ERR_UNSUPPORTED_FORMAT}: {output_format} transliteration is not supported for "
            f"{language!r}; available languages are {', '.join(available)}"
        )
        self.code = ERR_UNSUPPORTED_FORMAT
        self.language = language
        self.output_format = output_format
        self.available = available


def scan_supported(directory: Path) -> dict[str, list[str]]:
    """Read which formats each language has rules for in one directory.

    Args:
        directory (Path): Directory of ``{language}_{suffix}.rules`` files.

    Returns:
        dict[str, list[str]]: Language code to the formats it supports, in
        file-name order. A file whose name carries no suffix, or a suffix no
        format claims, is ignored — it is not a rule file this module can
        route to.

        No language can list a format twice: each format has exactly one
        suffix, and one directory cannot hold two files of the same name.
    """
    formats_by_suffix = {suffix: name for name, suffix in _SUFFIXES.items()}
    supported: dict[str, list[str]] = {}
    for rule_file in sorted(directory.glob("*.rules")):
        language, separator, suffix = rule_file.stem.partition("_")
        if not separator:
            continue
        output_format = formats_by_suffix.get(suffix)
        if output_format is None:
            continue
        supported.setdefault(language, []).append(output_format)
    return supported


def get_supported_languages() -> dict[str, list[str]]:
    """Return which formats each packaged language supports.

    Scanned once and remembered, because the answer is consulted for every
    line of every corpus and the packaged directory does not change while the
    process runs. Call :func:`clear_translit_caches` to forget it.

    Returns:
        dict[str, list[str]]: Language code to the formats it supports.
    """
    global _SUPPORTED_LANGUAGES
    if _SUPPORTED_LANGUAGES is None:
        _SUPPORTED_LANGUAGES = scan_supported(_RULE_DIR)
    return _SUPPORTED_LANGUAGES


def _rules(name: str) -> RuleSet:
    """Return a parsed rule file, parsing it the first time it is asked for.

    Args:
        name (str): Rule file name, including the ``.rules`` suffix.

    Returns:
        RuleSet: The parsed rules.

    Raises:
        FileNotFoundError: When no such rule file is packaged.
        RuleParseError: When the file cannot be parsed.
    """
    cached = _RULE_CACHE.get(name)
    if cached is None:
        cached = load_rules(name)
        _RULE_CACHE[name] = cached
    return cached


def clear_translit_caches() -> None:
    """Forget the scanned languages and every parsed rule file."""
    global _SUPPORTED_LANGUAGES
    _SUPPORTED_LANGUAGES = None
    _RULE_CACHE.clear()


def _rule_file_for(language: str, output_format: str) -> str:
    """Name the rule file providing one format for one language.

    Args:
        language (str): ISO 639 code.
        output_format (str): ``latin`` or ``ipa``.

    Returns:
        str: The rule file's name.

    Raises:
        UnsupportedTransliterationError: When no rule file provides that
            format for that language.
    """
    supported = get_supported_languages()
    if output_format not in supported.get(language, []):
        available = tuple(
            sorted(code for code, formats in supported.items() if output_format in formats)
        )
        raise UnsupportedTransliterationError(language, output_format, available)
    return f"{language}_{_SUFFIXES[output_format]}.rules"


def to_latin(text: str, lang: str, include_arabic: bool = False) -> str:
    """Transliterate text into the Latin orthography of one language.

    Args:
        text (str): Text to transliterate.
        lang (str): ISO 639 code of the language whose rules to use.
        include_arabic (bool): Run the Uyghur Arabic-to-Latin rules first, for
            text that may be in Arabic script.

    Returns:
        str: The transliterated text.

    Raises:
        UnsupportedTransliterationError: When the language has no Latin rules.
    """
    rule_file = _rule_file_for(lang, LATIN_FORMAT)
    prepared = apply_rules(text, _rules(ARABIC_PREPASS)) if include_arabic else text
    return apply_rules(prepared, _rules(rule_file))


def to_ipa(text: str, lang: str) -> str:
    """Transliterate text into IPA.

    Args:
        text (str): Text to transliterate.
        lang (str): ISO 639 code of the language whose rules to use.

    Returns:
        str: The IPA transcription. For Uzbek, the Cyrillic rules run over the
        result as a second pass, so that mixed-script Uzbek is transcribed
        rather than left half in Cyrillic.

    Raises:
        UnsupportedTransliterationError: When the language has no IPA rules.
    """
    result = apply_rules(text, _rules(_rule_file_for(lang, IPA_FORMAT)))
    if lang == "uz":
        return apply_rules(result, _rules(UZBEK_CYRILLIC_PASS))
    return result


__all__ = [
    "ARABIC_PREPASS",
    "ERR_UNSUPPORTED_FORMAT",
    "IPA_FORMAT",
    "LATIN_FORMAT",
    "UZBEK_CYRILLIC_PASS",
    "UnsupportedTransliterationError",
    "clear_translit_caches",
    "get_supported_languages",
    "scan_supported",
    "to_ipa",
    "to_latin",
]
