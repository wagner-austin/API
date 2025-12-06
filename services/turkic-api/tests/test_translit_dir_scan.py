from __future__ import annotations

from pathlib import Path

import pytest

import turkic_api.core.translit as ct


def test_get_supported_languages_scans_and_normalizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Create rule files: one without underscore (should be ignored) and two valid latin patterns
    (tmp_path / "foo.rules").write_text("", encoding="utf-8")
    (tmp_path / "kk_lat2023.rules").write_text("", encoding="utf-8")
    (tmp_path / "kk_lat.rules").write_text(
        "", encoding="utf-8"
    )  # duplicate fmt -> no second append
    (tmp_path / "ky_lat.rules").write_text("", encoding="utf-8")

    monkeypatch.setattr(ct, "_RULE_DIR", tmp_path)
    # Clear cached results after patching the rule directory
    ct.clear_translit_caches()
    supported = ct.get_supported_languages()
    # kk and ky should be present with normalized 'latin'
    assert supported.get("kk") == ["latin"]
    assert supported.get("ky") == ["latin"]
    # Reset cache to avoid leaking patched directory into other tests
    ct.clear_translit_caches()


def test_to_latin_missing_rule_file_branch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Force supported languages to claim latin exists for kk
    # Do not create any matching rule files
    monkeypatch.setattr(ct, "get_supported_languages", lambda: {"kk": ["latin"]})
    monkeypatch.setattr(ct, "_RULE_DIR", tmp_path)
    with pytest.raises(ValueError, match="No Latin rules file"):
        ct.to_latin("x", "kk")
    # No cache to clear here because we replaced the function via monkeypatch


def test_to_ipa_missing_rule_file_branch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    # Pretend "xx" supports ipa but the specific xx_ipa.rules file is missing
    monkeypatch.setattr(ct, "get_supported_languages", lambda: {"xx": ["ipa"]})
    monkeypatch.setattr(ct, "_RULE_DIR", tmp_path)
    with pytest.raises(ValueError, match="IPA rules file not found"):
        ct.to_ipa("hello", "xx")
