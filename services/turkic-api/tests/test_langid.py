from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from tests.conftest import make_probs

import turkic_api.core.langid as lid


def test_download_writes_only_nonempty_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "models" / "x.bin"

    class _Resp:
        def __init__(self) -> None:
            self._chunks = [b"", b"abc", b"", b"def"]
            self._idx = 0

        def iter_content(self, chunk_size: int = 8192) -> Generator[bytes]:
            yield from self._chunks

        def raise_for_status(self) -> None:
            return None

        def __enter__(self) -> _Resp:
            return self

        def __exit__(self, *_: str | int | float | bool | None) -> None:
            return None

    def _get(url: str, *, stream: bool, timeout: int) -> _Resp:
        return _Resp()

    monkeypatch.setattr("turkic_api.core.langid.requests.get", _get)
    lid._download("http://example/x", dest)
    assert dest.read_bytes() == b"abcdef"


def test_ensure_model_path_218e_downloads_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def _fake_download(url: str, dest: Path) -> None:
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"bin")

    monkeypatch.setattr(lid, "_download", _fake_download)
    out = lid.ensure_model_path(str(tmp_path), prefer_218e=True)
    assert out.name == "lid218e.bin"
    assert out.exists()
    assert calls != []
    assert "lid218e" in calls[0]


def test_ensure_model_path_218e_existing_no_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    p = tmp_path / "models" / "lid218e.bin"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"bin")

    def _boom(_u: str, _d: Path) -> None:
        raise AssertionError("download should not be called")

    monkeypatch.setattr(lid, "_download", _boom)
    out = lid.ensure_model_path(str(tmp_path), prefer_218e=True)
    assert out == p


def test_ensure_model_path_176_branch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def _fake_download(url: str, dest: Path) -> None:
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"bin")

    monkeypatch.setattr(lid, "_download", _fake_download)
    out = lid.ensure_model_path(str(tmp_path), prefer_218e=False)
    assert out.name == "lid.176.bin"
    assert calls != []
    assert "lid.176" in calls[0]

    # Call again after file exists: should not download
    calls.clear()
    out2 = lid.ensure_model_path(str(tmp_path), prefer_218e=False)
    assert out2 == out
    assert calls == []


def test_build_lang_filter_with_threshold(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    class _Model:
        def predict(self, text: str, k: int = 1) -> tuple[tuple[str, ...], NDArray[np.float64]]:
            # Return variants to hit mapping logic: __label__kk and kaz_Cyrl
            if "cyrl" in text.lower():
                return (("__label__kaz_Cyrl",), make_probs(0.95))
            return (("__label__kk",), make_probs(0.80))

    model = _Model()
    keep = lid.build_lang_filter(target_lang="kk", threshold=0.90, model=model)
    assert keep("foo cyrl") is True  # kaz_Cyrl maps to kk
    assert keep("bar") is False  # prob below threshold


def test_build_lang_script_filter_match_and_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _Model:
        def predict(self, text: str, k: int = 1) -> tuple[tuple[str, ...], NDArray[np.float64]]:
            t = text.lower()
            if "latn" in t:
                return (("__label__kaz_Latn",), make_probs(0.99))
            if "cyrl" in t:
                return (("__label__kaz_Cyrl",), make_probs(0.99))
            return (("__label__eng",), make_probs(0.99))

    model = _Model()
    # Script normalized from lower-case
    keep = lid.build_lang_script_filter(target_lang="kk", script="latn", threshold=0.5, model=model)
    assert keep("text latn") is True
    assert keep("text cyrl") is False  # script mismatch -> return False
    # Lang mismatch -> return False
    assert keep("english") is False
    # No script filter
    keep2 = lid.build_lang_script_filter(target_lang="kk", script=None, threshold=0.5, model=model)
    assert keep2("text latn") is True


def test_build_lang_script_filter_blank_script_treated_as_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class _Model:
        def predict(self, text: str, k: int = 1) -> tuple[tuple[str, ...], NDArray[np.float64]]:
            return (("__label__kaz_Latn",), make_probs(0.99))

    model = _Model()
    keep = lid.build_lang_script_filter(target_lang="kk", script="   ", threshold=0.5, model=model)
    # Blank script should be treated as None (no script gating)
    assert keep("anything") is True


def test_get_fasttext_model_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test that _get_fasttext_model_factory returns the _FastText class."""

    class _FakeModel:
        def __init__(self, *, model_path: str) -> None:
            self._model_path = model_path

        def predict(self, text: str, k: int = 1) -> tuple[tuple[str, ...], NDArray[np.float64]]:
            return (("__label__en",), make_probs(0.99))

    class _FakeFastTextModule:
        _FastText = _FakeModel

    def fake_import(
        name: str, globals_: dict[str, str] | None = None, fromlist: list[str] | None = None
    ) -> _FakeFastTextModule:
        assert name == "fasttext.FastText"
        assert fromlist == ["_FastText"]
        return _FakeFastTextModule()

    monkeypatch.setattr("builtins.__import__", fake_import)
    factory = lid._get_fasttext_model_factory()
    model = factory(model_path="/fake/path.bin")
    # Verify factory works by calling predict
    labels, _probs = model.predict("test", k=1)
    assert labels == ("__label__en",)


def test_extract_prob_empty_array() -> None:
    """Test _extract_prob returns 0.0 for empty arrays."""
    empty_list: list[np.float64] = []
    empty_probs: NDArray[np.float64] = np.array(empty_list, dtype=np.float64)
    assert lid._extract_prob(empty_probs) == 0.0


def test_extract_prob_with_value() -> None:
    """Test _extract_prob returns the first value for non-empty arrays."""
    assert lid._extract_prob(make_probs(0.95, 0.05)) == 0.95
