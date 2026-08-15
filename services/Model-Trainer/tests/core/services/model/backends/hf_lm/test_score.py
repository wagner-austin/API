"""Tests for HuggingFace LM score module."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pytest
from platform_core.errors import AppError

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import PreparedLMModel
from model_trainer.core.services.model.backends.hf_lm._test_hooks import (
    Hooks,
    reset_hooks,
)
from model_trainer.core.services.model.backends.hf_lm.score import (
    _read_text_or_path,
    score_hf_lm,
)

from .testing import FakeEncoder, FakeScoreModel, FakeSinglePositionScoreModel, make_score_config


class _SettingsFactory(Protocol):
    def __call__(
        self,
        *,
        artifacts_root: str | None = None,
        data_root: str | None = None,
    ) -> Settings: ...


class TestReadTextOrPath:
    """Tests for _read_text_or_path function."""

    def setup_method(self) -> None:
        """Reset hooks before each test."""
        reset_hooks()

    def teardown_method(self) -> None:
        """Clean up hooks after each test."""
        reset_hooks()

    def test_returns_text_when_provided(
        self, tmp_path: Path, settings_factory: _SettingsFactory
    ) -> None:
        """Test that text is returned directly."""
        cfg = make_score_config(text="Hello world")
        settings = settings_factory(
            artifacts_root=str(tmp_path / "artifacts"),
            data_root=str(tmp_path / "data"),
        )
        result = _read_text_or_path(cfg, settings)
        assert result == "Hello world"

    def test_raises_when_neither_text_nor_path(
        self, tmp_path: Path, settings_factory: _SettingsFactory
    ) -> None:
        """Test error when neither text nor path provided."""
        cfg = make_score_config(text=None, path=None)
        settings = settings_factory(
            artifacts_root=str(tmp_path / "artifacts"),
            data_root=str(tmp_path / "data"),
        )
        with pytest.raises(AppError, match="either text or path"):
            _read_text_or_path(cfg, settings)

    def test_raises_when_path_outside_data_root(
        self, tmp_path: Path, settings_factory: _SettingsFactory
    ) -> None:
        """Test error when path is outside data_root."""
        cfg = make_score_config(text=None, path="/etc/passwd")
        settings = settings_factory(
            artifacts_root=str(tmp_path / "artifacts"),
            data_root=str(tmp_path / "data"),
        )
        with pytest.raises(AppError, match="path must be under data_root"):
            _read_text_or_path(cfg, settings)

    def test_reads_file_via_hook(self, tmp_path: Path, settings_factory: _SettingsFactory) -> None:
        """Test that file is read via hook."""
        test_file = tmp_path / "text.txt"
        test_file.write_text("Test text content")

        read_calls: list[Path] = []

        def fake_read(path: Path) -> str:
            read_calls.append(path)
            return "Content from hook"

        Hooks.read_text_file = fake_read

        cfg = make_score_config(text=None, path=str(test_file))
        settings = settings_factory(
            artifacts_root=str(tmp_path),
            data_root=str(tmp_path),
        )

        result = _read_text_or_path(cfg, settings)
        assert result == "Content from hook"
        assert len(read_calls) == 1
        assert read_calls[0].name == "text.txt"


class TestScoreHfLm:
    """Tests for score_hf_lm function."""

    def setup_method(self) -> None:
        """Reset hooks before each test."""
        reset_hooks()

    def teardown_method(self) -> None:
        """Clean up hooks after each test."""
        reset_hooks()

    def _make_settings(self, tmp_path: Path, settings_factory: _SettingsFactory) -> Settings:
        """Create test settings."""
        return settings_factory(
            artifacts_root=str(tmp_path / "artifacts"),
            data_root=str(tmp_path / "data"),
        )

    def test_scores_text_with_summary_level(
        self, tmp_path: Path, settings_factory: _SettingsFactory
    ) -> None:
        """Test scoring with summary detail level."""
        Hooks.read_text_file = lambda p: "unused"

        model = FakeScoreModel()
        prepared = PreparedLMModel(
            model=model,
            tokenizer_id="test-tok",
            eos_id=0,
            pad_id=1,
            max_seq_len=128,
            tok_for_dataset=FakeEncoder(),
        )

        cfg = make_score_config(text="Hello world", detail_level="summary")
        settings = self._make_settings(tmp_path, settings_factory)

        result = score_hf_lm(prepared=prepared, cfg=cfg, settings=settings)

        assert result["loss"] >= 0.0
        assert result["perplexity"] >= 1.0
        assert result["surprisal"] is None
        assert result["tokens"] is None

    def test_scores_text_with_per_char_level(
        self, tmp_path: Path, settings_factory: _SettingsFactory
    ) -> None:
        """Test scoring with per_char detail level."""
        Hooks.read_text_file = lambda p: "unused"

        model = FakeScoreModel()
        prepared = PreparedLMModel(
            model=model,
            tokenizer_id="test-tok",
            eos_id=0,
            pad_id=1,
            max_seq_len=128,
            tok_for_dataset=FakeEncoder(),
        )

        cfg = make_score_config(text="Hello world", detail_level="per_char")
        settings = self._make_settings(tmp_path, settings_factory)

        result = score_hf_lm(prepared=prepared, cfg=cfg, settings=settings)

        surprisal = result["surprisal"]
        tokens = result["tokens"]
        # Verify we got surprisal values by checking the first element
        surprisal_list = list(surprisal) if surprisal else []
        tokens_list = list(tokens) if tokens else []
        first_surprisal = surprisal_list[0]
        first_token = tokens_list[0]
        assert first_surprisal >= 0.0
        assert first_token != ""

    def test_scores_text_with_topk(
        self, tmp_path: Path, settings_factory: _SettingsFactory
    ) -> None:
        """Test scoring with top-k predictions."""
        Hooks.read_text_file = lambda p: "unused"

        model = FakeScoreModel()
        prepared = PreparedLMModel(
            model=model,
            tokenizer_id="test-tok",
            eos_id=0,
            pad_id=1,
            max_seq_len=128,
            tok_for_dataset=FakeEncoder(),
        )

        cfg = make_score_config(text="Hello", top_k=5)
        settings = self._make_settings(tmp_path, settings_factory)

        result = score_hf_lm(prepared=prepared, cfg=cfg, settings=settings)

        topk = result["topk"]
        # Verify we got topk values by checking the first element
        topk_list = list(topk) if topk else []
        first_topk = topk_list[0]
        first_prediction = first_topk[0]
        _, prob = first_prediction
        assert prob >= 0.0
        assert prob <= 1.0

    def test_returns_default_for_single_token(
        self, tmp_path: Path, settings_factory: _SettingsFactory
    ) -> None:
        """Test that single token text returns default values."""
        Hooks.read_text_file = lambda p: "unused"

        model = FakeScoreModel()

        class _SingleTokenEncoder:
            """Encoder that always returns single token."""

            def encode(self, text: str) -> _Enc:
                return _Enc([42])

            def token_to_id(self, token: str) -> int | None:
                return 42

            def get_vocab_size(self) -> int:
                return 100

            def decode(self, ids: list[int]) -> str:
                return "x"

        class _Enc:
            """Fake encoded output."""

            def __init__(self, ids: list[int]) -> None:
                self._ids = ids

            @property
            def ids(self) -> list[int]:
                return self._ids

        prepared = PreparedLMModel(
            model=model,
            tokenizer_id="test-tok",
            eos_id=0,
            pad_id=1,
            max_seq_len=128,
            tok_for_dataset=_SingleTokenEncoder(),
        )

        cfg = make_score_config(text="x")
        settings = self._make_settings(tmp_path, settings_factory)

        result = score_hf_lm(prepared=prepared, cfg=cfg, settings=settings)

        assert result["loss"] == 0.0
        assert result["perplexity"] == 1.0

    def test_truncates_to_max_seq_len(
        self, tmp_path: Path, settings_factory: _SettingsFactory
    ) -> None:
        """Test that long text is truncated."""
        Hooks.read_text_file = lambda p: "unused"

        model = FakeScoreModel()
        prepared = PreparedLMModel(
            model=model,
            tokenizer_id="test-tok",
            eos_id=0,
            pad_id=1,
            max_seq_len=10,
            tok_for_dataset=FakeEncoder(),
        )

        cfg = make_score_config(text="A" * 100)
        settings = self._make_settings(tmp_path, settings_factory)

        result = score_hf_lm(prepared=prepared, cfg=cfg, settings=settings)
        assert result["loss"] >= 0.0

    def test_respects_seed(self, tmp_path: Path, settings_factory: _SettingsFactory) -> None:
        """Test that seed is set when provided."""
        Hooks.read_text_file = lambda p: "unused"

        model = FakeScoreModel()
        prepared = PreparedLMModel(
            model=model,
            tokenizer_id="test-tok",
            eos_id=0,
            pad_id=1,
            max_seq_len=128,
            tok_for_dataset=FakeEncoder(),
        )

        cfg = make_score_config(text="Hello", seed=12345)
        settings = self._make_settings(tmp_path, settings_factory)

        result = score_hf_lm(prepared=prepared, cfg=cfg, settings=settings)
        assert result["loss"] >= 0.0

    def test_without_seed(self, tmp_path: Path, settings_factory: _SettingsFactory) -> None:
        """Test scoring without seed (seed=None)."""
        Hooks.read_text_file = lambda p: "unused"

        model = FakeScoreModel()
        prepared = PreparedLMModel(
            model=model,
            tokenizer_id="test-tok",
            eos_id=0,
            pad_id=1,
            max_seq_len=128,
            tok_for_dataset=FakeEncoder(),
        )

        cfg = make_score_config(text="Hello", seed=None)
        settings = self._make_settings(tmp_path, settings_factory)

        result = score_hf_lm(prepared=prepared, cfg=cfg, settings=settings)
        assert result["loss"] >= 0.0

    def test_model_returns_single_position_logits(
        self, tmp_path: Path, settings_factory: _SettingsFactory
    ) -> None:
        """Test scoring when model returns logits with only 1 position (edge case)."""
        Hooks.read_text_file = lambda p: "unused"

        model = FakeSinglePositionScoreModel()
        prepared = PreparedLMModel(
            model=model,
            tokenizer_id="test-tok",
            eos_id=0,
            pad_id=1,
            max_seq_len=128,
            tok_for_dataset=FakeEncoder(),
        )

        cfg = make_score_config(text="Hello world")
        settings = self._make_settings(tmp_path, settings_factory)

        result = score_hf_lm(prepared=prepared, cfg=cfg, settings=settings)
        # With single position logits, loss should be 0.0 and perplexity 1.0
        assert result["loss"] == 0.0
        assert result["perplexity"] == 1.0
