"""Tests for platform_stt.langid module."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from platform_stt import _test_hooks
from platform_stt.langid import (
    _extract_prob,
    _parse_label,
    detect_language,
    ensure_model_path,
    is_language,
    load_langid_model,
)
from platform_stt.testing import FakeLangIdModel


class TestParseLabel:
    """Tests for _parse_label helper."""

    def test_parse_label_vietnamese(self) -> None:
        """Parse Vietnamese label with script."""
        lang, script = _parse_label("__label__vie_Latn")
        assert lang == "vi"
        assert script == "Latn"

    def test_parse_label_english(self) -> None:
        """Parse English label with script."""
        lang, script = _parse_label("__label__eng_Latn")
        assert lang == "en"
        assert script == "Latn"

    def test_parse_label_no_script(self) -> None:
        """Parse label without script."""
        lang, script = _parse_label("__label__en")
        assert lang == "en"
        assert script is None

    def test_parse_label_unknown_lang(self) -> None:
        """Parse unknown language code."""
        lang, script = _parse_label("__label__xyz_Cyrl")
        assert lang == "xyz"  # Returns as-is if not in mapping
        assert script == "Cyrl"

    def test_parse_label_iso639_3_mappings(self) -> None:
        """Test various ISO 639-3 to 639-1 mappings."""
        # Japanese
        lang, _ = _parse_label("__label__jpn_Jpan")
        assert lang == "ja"
        # Korean
        lang, _ = _parse_label("__label__kor_Hang")
        assert lang == "ko"
        # Chinese
        lang, _ = _parse_label("__label__cmn_Hans")
        assert lang == "zh"
        # Spanish
        lang, _ = _parse_label("__label__spa_Latn")
        assert lang == "es"


class TestExtractProb:
    """Tests for _extract_prob helper."""

    def test_extract_prob_single_value(self) -> None:
        """Extract probability from single-element array."""
        probs: NDArray[np.float64] = np.zeros(1, dtype=np.float64)
        probs[0] = 0.95
        result = _extract_prob(probs)
        assert result == 0.95

    def test_extract_prob_empty_array(self) -> None:
        """Return 0.0 for empty array."""
        probs: NDArray[np.float64] = np.zeros(0, dtype=np.float64)
        result = _extract_prob(probs)
        assert result == 0.0

    def test_extract_prob_multiple_values(self) -> None:
        """Extract first probability from multi-element array."""
        probs: NDArray[np.float64] = np.zeros(3, dtype=np.float64)
        probs[0] = 0.7
        probs[1] = 0.2
        probs[2] = 0.1
        result = _extract_prob(probs)
        assert result == 0.7


class TestEnsureModelPath:
    """Tests for ensure_model_path function."""

    def test_ensure_model_path_218e_exists(self, tmp_path: Path) -> None:
        """Return path when 218e model exists."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        model_file = model_dir / "lid218e.bin"
        model_file.write_bytes(b"fake model")

        result = ensure_model_path(str(tmp_path), prefer_218e=True)
        assert result == model_file

    def test_ensure_model_path_176_exists(self, tmp_path: Path) -> None:
        """Return path when 176 model exists."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        model_file = model_dir / "lid.176.bin"
        model_file.write_bytes(b"fake model")

        result = ensure_model_path(str(tmp_path), prefer_218e=False)
        assert result == model_file

    def test_ensure_model_path_downloads_218e(self, tmp_path: Path) -> None:
        """Download 218e model when missing."""
        download_calls: list[tuple[str, Path]] = []

        def fake_download(url: str, dest: Path) -> None:
            download_calls.append((url, dest))
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"downloaded")

        original = _test_hooks.langid_download
        try:
            _test_hooks.langid_download = fake_download
            result = ensure_model_path(str(tmp_path), prefer_218e=True)

            assert len(download_calls) == 1
            assert "lid218e.bin" in download_calls[0][0]
            assert result.name == "lid218e.bin"
        finally:
            _test_hooks.langid_download = original

    def test_ensure_model_path_downloads_176(self, tmp_path: Path) -> None:
        """Download 176 model when missing."""
        download_calls: list[tuple[str, Path]] = []

        def fake_download(url: str, dest: Path) -> None:
            download_calls.append((url, dest))
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(b"downloaded")

        original = _test_hooks.langid_download
        try:
            _test_hooks.langid_download = fake_download
            result = ensure_model_path(str(tmp_path), prefer_218e=False)

            assert len(download_calls) == 1
            assert "lid.176.bin" in download_calls[0][0]
            assert result.name == "lid.176.bin"
        finally:
            _test_hooks.langid_download = original


class TestLoadLangidModel:
    """Tests for load_langid_model function."""

    def test_load_langid_model(self, tmp_path: Path) -> None:
        """Load model using factory."""
        # Set up model file
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        model_file = model_dir / "lid218e.bin"
        model_file.write_bytes(b"fake model")

        # Set up fake factory
        fake_model = FakeLangIdModel()
        factory_calls: list[str] = []

        def fake_factory(*, model_path: str) -> _test_hooks.LangIdModelProtocol:
            factory_calls.append(model_path)
            return fake_model

        original = _test_hooks.langid_get_fasttext_factory
        try:
            _test_hooks.langid_get_fasttext_factory = lambda: fake_factory
            result = load_langid_model(str(tmp_path), prefer_218e=True)

            assert result is fake_model
            assert len(factory_calls) == 1
            assert "lid218e.bin" in factory_calls[0]
        finally:
            _test_hooks.langid_get_fasttext_factory = original


class TestDetectLanguage:
    """Tests for detect_language function."""

    def test_detect_language_vietnamese(self) -> None:
        """Detect Vietnamese text."""
        model = FakeLangIdModel(label="__label__vie_Latn", confidence=0.95)
        result = detect_language("Xin chào", model)

        assert result["language"] == "vi"
        assert result["confidence"] == 0.95
        assert result["script"] == "Latn"

    def test_detect_language_english(self) -> None:
        """Detect English text."""
        model = FakeLangIdModel(label="__label__eng_Latn", confidence=0.99)
        result = detect_language("Hello world", model)

        assert result["language"] == "en"
        assert result["confidence"] == 0.99

    def test_detect_language_empty_text(self) -> None:
        """Return undetermined for empty text."""
        model = FakeLangIdModel()
        result = detect_language("", model)

        assert result["language"] == "und"
        assert result["confidence"] == 0.0
        assert result["script"] is None

    def test_detect_language_whitespace_only(self) -> None:
        """Return undetermined for whitespace-only text."""
        model = FakeLangIdModel()
        result = detect_language("   \n\t  ", model)

        assert result["language"] == "und"
        assert result["confidence"] == 0.0

    def test_detect_language_below_threshold(self) -> None:
        """Return undetermined when below threshold."""
        model = FakeLangIdModel(label="__label__vie_Latn", confidence=0.3)
        result = detect_language("maybe vietnamese", model, threshold=0.5)

        assert result["language"] == "und"
        assert result["confidence"] == 0.3

    def test_detect_language_above_threshold(self) -> None:
        """Return detected language when above threshold."""
        model = FakeLangIdModel(label="__label__vie_Latn", confidence=0.8)
        result = detect_language("xin chào", model, threshold=0.5)

        assert result["language"] == "vi"
        assert result["confidence"] == 0.8

    def test_detect_language_normalizes_newlines(self) -> None:
        """Normalize newlines in text before prediction."""
        model = FakeLangIdModel(label="__label__eng_Latn", confidence=0.9)
        result = detect_language("Hello\nworld\n", model)

        assert result["language"] == "en"


class TestIsLanguage:
    """Tests for is_language function."""

    def test_is_language_match(self) -> None:
        """Return True when language matches with confidence."""
        model = FakeLangIdModel(label="__label__vie_Latn", confidence=0.9)
        result = is_language("Xin chào", "vi", model, threshold=0.5)
        assert result is True

    def test_is_language_no_match_different_lang(self) -> None:
        """Return False when different language detected."""
        model = FakeLangIdModel(label="__label__eng_Latn", confidence=0.9)
        result = is_language("Hello", "vi", model, threshold=0.5)
        assert result is False

    def test_is_language_no_match_low_confidence(self) -> None:
        """Return False when confidence below threshold."""
        model = FakeLangIdModel(label="__label__vie_Latn", confidence=0.3)
        result = is_language("maybe vietnamese", "vi", model, threshold=0.5)
        assert result is False

    def test_is_language_exact_threshold(self) -> None:
        """Return True when confidence equals threshold."""
        model = FakeLangIdModel(label="__label__vie_Latn", confidence=0.5)
        result = is_language("text", "vi", model, threshold=0.5)
        assert result is True
