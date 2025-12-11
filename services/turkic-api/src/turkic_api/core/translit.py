from __future__ import annotations

from pathlib import Path

from turkic_api.core.transliteval import Rule, apply_rules, load_rules

# Module-level directory path - not Final to allow test overrides
_RULE_DIR: Path = Path(__file__).with_suffix("").parent / "rules"

_SUPPORTED_LANGS: dict[str, list[str]] | None = None
_RULE_CACHE: dict[str, list[Rule]] = {}


def get_supported_languages() -> dict[str, list[str]]:
    global _SUPPORTED_LANGS
    if _SUPPORTED_LANGS is None:
        supported: dict[str, list[str]] = {}
        for rule_file in _RULE_DIR.glob("*.rules"):
            filename = rule_file.stem
            if "_" not in filename:
                continue
            lang, fmt = filename.split("_", 1)
            if fmt in {"lat2023", "lat"}:
                fmt = "latin"
            lst = supported.get(lang)
            if lst is None:
                supported[lang] = [fmt]
            elif fmt not in lst:
                lst.append(fmt)
        _SUPPORTED_LANGS = supported
    return _SUPPORTED_LANGS


def _rules(name: str) -> list[Rule]:
    cached = _RULE_CACHE.get(name)
    if cached is None:
        cached = load_rules(name)
        _RULE_CACHE[name] = cached
    return cached


def clear_translit_caches() -> None:
    global _SUPPORTED_LANGS
    _SUPPORTED_LANGS = None
    _RULE_CACHE.clear()


def to_latin(text: str, lang: str, include_arabic: bool = False) -> str:
    supported = get_supported_languages()
    if lang not in supported or "latin" not in supported[lang]:
        available = [code for code, fmts in supported.items() if "latin" in fmts]
        raise ValueError(
            f"Latin transliteration not supported for '{lang}'. "
            f"Available languages: {', '.join(sorted(available))}"
        )
    candidates = (
        f"{lang}_lat2023.rules",
        f"{lang}_lat.rules",
        f"{lang}_latin.rules",
    )
    rule_file: str | None = None
    for name in candidates:
        if (_RULE_DIR / name).exists():
            rule_file = name
            break
    if rule_file is None:
        raise ValueError(f"No Latin rules file found for language '{lang}'")
    # Optional Arabic pre-pass
    txt = text
    if include_arabic and (_RULE_DIR / "ar_lat.rules").exists():
        txt = apply_rules(txt, _rules("ar_lat.rules"))
    return apply_rules(txt, _rules(rule_file))


def to_ipa(text: str, lang: str) -> str:
    supported = get_supported_languages()
    if lang not in supported or "ipa" not in supported[lang]:
        available = [code for code, fmts in supported.items() if "ipa" in fmts]
        raise ValueError(
            f"IPA transliteration not supported for '{lang}'. "
            f"Available languages: {', '.join(sorted(available))}"
        )
    rule_file = f"{lang}_ipa.rules"
    if not (_RULE_DIR / rule_file).exists():
        raise ValueError(f"IPA rules file not found for language '{lang}'")
    result = apply_rules(text, _rules(rule_file))

    # ADDITION 2025-12-02: Uzbek mixed-script support
    # OSCAR Uzbek is 96.8% Latin, 0.8% Cyrillic - apply both rule sets
    # See docs/ipa-rules-refactor.md
    if lang == "uz":
        result = apply_rules(result, _rules("uzc_ipa.rules"))

    return result
