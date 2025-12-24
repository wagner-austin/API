"""Tests for HuggingFace LM generate module."""

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
from model_trainer.core.services.model.backends.hf_lm.generate import (
    _read_prompt,
    generate_hf_lm,
)

from .testing import FakeEncoder, FakeGenerateModel, make_generate_config


class _SettingsFactory(Protocol):
    def __call__(
        self,
        *,
        artifacts_root: str | None = None,
        data_root: str | None = None,
    ) -> Settings: ...


class TestReadPrompt:
    """Tests for _read_prompt function."""

    def setup_method(self) -> None:
        """Reset hooks before each test."""
        reset_hooks()

    def teardown_method(self) -> None:
        """Clean up hooks after each test."""
        reset_hooks()

    def test_returns_prompt_text_when_provided(
        self, tmp_path: Path, settings_factory: _SettingsFactory
    ) -> None:
        """Test that prompt_text is returned directly."""
        cfg = make_generate_config(prompt_text="Hello world")
        settings = settings_factory(
            artifacts_root=str(tmp_path / "artifacts"),
            data_root=str(tmp_path / "data"),
        )
        result = _read_prompt(cfg, settings)
        assert result == "Hello world"

    def test_raises_when_neither_text_nor_path(
        self, tmp_path: Path, settings_factory: _SettingsFactory
    ) -> None:
        """Test error when neither text nor path provided."""
        cfg = make_generate_config(prompt_text=None, prompt_path=None)
        settings = settings_factory(
            artifacts_root=str(tmp_path / "artifacts"),
            data_root=str(tmp_path / "data"),
        )
        with pytest.raises(AppError, match="either prompt_text or prompt_path"):
            _read_prompt(cfg, settings)

    def test_raises_when_path_outside_data_root(
        self, tmp_path: Path, settings_factory: _SettingsFactory
    ) -> None:
        """Test error when path is outside data_root."""
        cfg = make_generate_config(prompt_text=None, prompt_path="/etc/passwd")
        settings = settings_factory(
            artifacts_root=str(tmp_path / "artifacts"),
            data_root=str(tmp_path / "data"),
        )
        with pytest.raises(AppError, match="prompt_path must be under data_root"):
            _read_prompt(cfg, settings)

    def test_raises_when_hook_not_initialized(
        self, tmp_path: Path, settings_factory: _SettingsFactory
    ) -> None:
        """Test error when read_text_file hook not set."""
        test_file = tmp_path / "prompt.txt"
        test_file.write_text("Test prompt")

        cfg = make_generate_config(prompt_text=None, prompt_path=str(test_file))
        settings = settings_factory(
            artifacts_root=str(tmp_path),
            data_root=str(tmp_path),
        )
        with pytest.raises(RuntimeError, match=r"Hooks\.read_text_file not initialized"):
            _read_prompt(cfg, settings)

    def test_reads_file_via_hook(self, tmp_path: Path, settings_factory: _SettingsFactory) -> None:
        """Test that file is read via hook."""
        test_file = tmp_path / "prompt.txt"
        test_file.write_text("Test prompt content")

        read_calls: list[Path] = []

        def fake_read(path: Path) -> str:
            read_calls.append(path)
            return "Content from hook"

        Hooks.read_text_file = fake_read

        cfg = make_generate_config(prompt_text=None, prompt_path=str(test_file))
        settings = settings_factory(
            artifacts_root=str(tmp_path),
            data_root=str(tmp_path),
        )

        result = _read_prompt(cfg, settings)
        assert result == "Content from hook"
        assert len(read_calls) == 1
        assert read_calls[0].name == "prompt.txt"


class TestGenerateHfLm:
    """Tests for generate_hf_lm function."""

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

    def test_generates_text_with_prompt_text(
        self, tmp_path: Path, settings_factory: _SettingsFactory
    ) -> None:
        """Test generation with direct prompt text."""
        Hooks.read_text_file = lambda p: "unused"

        model = FakeGenerateModel(eos_id=0)
        prepared = PreparedLMModel(
            model=model,
            tokenizer_id="test-tok",
            eos_id=0,
            pad_id=1,
            max_seq_len=128,
            tok_for_dataset=FakeEncoder(),
        )

        cfg = make_generate_config(
            prompt_text="Hello",
            temperature=0.0,
            max_new_tokens=5,
        )
        settings = self._make_settings(tmp_path, settings_factory)

        result = generate_hf_lm(prepared=prepared, cfg=cfg, settings=settings)

        outputs = list(result["outputs"])
        assert len(outputs) == 1
        assert result["steps"] >= 1

    def test_generates_multiple_sequences(
        self, tmp_path: Path, settings_factory: _SettingsFactory
    ) -> None:
        """Test generation with multiple return sequences."""
        Hooks.read_text_file = lambda p: "unused"

        model = FakeGenerateModel(eos_id=0)
        prepared = PreparedLMModel(
            model=model,
            tokenizer_id="test-tok",
            eos_id=0,
            pad_id=1,
            max_seq_len=128,
            tok_for_dataset=FakeEncoder(),
        )

        cfg = make_generate_config(
            prompt_text="Hello",
            num_return_sequences=3,
            temperature=0.7,
        )
        settings = self._make_settings(tmp_path, settings_factory)

        result = generate_hf_lm(prepared=prepared, cfg=cfg, settings=settings)

        outputs = list(result["outputs"])
        eos_terminated = list(result["eos_terminated"])
        assert len(outputs) == 3
        assert len(eos_terminated) == 3

    def test_respects_seed(self, tmp_path: Path, settings_factory: _SettingsFactory) -> None:
        """Test that seed is set when provided."""
        Hooks.read_text_file = lambda p: "unused"

        model = FakeGenerateModel(eos_id=0)
        prepared = PreparedLMModel(
            model=model,
            tokenizer_id="test-tok",
            eos_id=0,
            pad_id=1,
            max_seq_len=128,
            tok_for_dataset=FakeEncoder(),
        )

        cfg = make_generate_config(prompt_text="Hello", seed=12345)
        settings = self._make_settings(tmp_path, settings_factory)

        result = generate_hf_lm(prepared=prepared, cfg=cfg, settings=settings)
        outputs = list(result["outputs"])
        first_output = outputs[0]
        # Verify we got an output
        assert type(first_output) is str

    def test_truncates_prompt_if_too_long(
        self, tmp_path: Path, settings_factory: _SettingsFactory
    ) -> None:
        """Test that long prompts are truncated."""
        Hooks.read_text_file = lambda p: "unused"

        model = FakeGenerateModel(eos_id=0)
        prepared = PreparedLMModel(
            model=model,
            tokenizer_id="test-tok",
            eos_id=0,
            pad_id=1,
            max_seq_len=10,
            tok_for_dataset=FakeEncoder(),
        )

        cfg = make_generate_config(
            prompt_text="A" * 100,
            max_new_tokens=5,
        )
        settings = self._make_settings(tmp_path, settings_factory)

        result = generate_hf_lm(prepared=prepared, cfg=cfg, settings=settings)
        outputs = list(result["outputs"])
        first_output = outputs[0]
        # Verify we got an output
        assert type(first_output) is str

    def test_without_eos_in_output(
        self, tmp_path: Path, settings_factory: _SettingsFactory
    ) -> None:
        """Test generation when EOS is not in generated tokens."""
        Hooks.read_text_file = lambda p: "unused"

        model = FakeGenerateModel(eos_id=0, include_eos=False)
        prepared = PreparedLMModel(
            model=model,
            tokenizer_id="test-tok",
            eos_id=0,
            pad_id=1,
            max_seq_len=128,
            tok_for_dataset=FakeEncoder(),
        )

        cfg = make_generate_config(prompt_text="Hello", temperature=0.7)
        settings = self._make_settings(tmp_path, settings_factory)

        result = generate_hf_lm(prepared=prepared, cfg=cfg, settings=settings)

        eos_terminated = list(result["eos_terminated"])
        first_terminated = eos_terminated[0]
        assert first_terminated is False

    def test_with_stop_sequences(self, tmp_path: Path, settings_factory: _SettingsFactory) -> None:
        """Test generation with stop sequences."""
        Hooks.read_text_file = lambda p: "unused"

        model = FakeGenerateModel(eos_id=0, include_eos=False)
        prepared = PreparedLMModel(
            model=model,
            tokenizer_id="test-tok",
            eos_id=0,
            pad_id=1,
            max_seq_len=128,
            tok_for_dataset=FakeEncoder(),
        )

        cfg = make_generate_config(
            prompt_text="Hello",
            temperature=0.7,
            stop_sequences=["stop", "end"],
        )
        settings = self._make_settings(tmp_path, settings_factory)

        result = generate_hf_lm(prepared=prepared, cfg=cfg, settings=settings)

        outputs = list(result["outputs"])
        first_output = outputs[0]
        assert type(first_output) is str

    def test_without_seed(self, tmp_path: Path, settings_factory: _SettingsFactory) -> None:
        """Test generation without seed (seed=None)."""
        Hooks.read_text_file = lambda p: "unused"

        model = FakeGenerateModel(eos_id=0)
        prepared = PreparedLMModel(
            model=model,
            tokenizer_id="test-tok",
            eos_id=0,
            pad_id=1,
            max_seq_len=128,
            tok_for_dataset=FakeEncoder(),
        )

        cfg = make_generate_config(prompt_text="Hello", seed=None)
        settings = self._make_settings(tmp_path, settings_factory)

        result = generate_hf_lm(prepared=prepared, cfg=cfg, settings=settings)
        outputs = list(result["outputs"])
        first_output = outputs[0]
        assert type(first_output) is str

    def test_with_eos_and_stop_on_eos_false(
        self, tmp_path: Path, settings_factory: _SettingsFactory
    ) -> None:
        """Test generation with EOS in output but stop_on_eos=False."""
        Hooks.read_text_file = lambda p: "unused"

        model = FakeGenerateModel(eos_id=0, include_eos=True)
        prepared = PreparedLMModel(
            model=model,
            tokenizer_id="test-tok",
            eos_id=0,
            pad_id=1,
            max_seq_len=128,
            tok_for_dataset=FakeEncoder(),
        )

        cfg = make_generate_config(prompt_text="Hello", stop_on_eos=False)
        settings = self._make_settings(tmp_path, settings_factory)

        result = generate_hf_lm(prepared=prepared, cfg=cfg, settings=settings)

        eos_terminated = list(result["eos_terminated"])
        first_terminated = eos_terminated[0]
        # EOS was found but we didn't stop
        assert first_terminated is True

    def test_stop_sequence_found_in_text(
        self, tmp_path: Path, settings_factory: _SettingsFactory
    ) -> None:
        """Test generation stops at stop sequence when found in text."""
        Hooks.read_text_file = lambda p: "unused"

        model = FakeGenerateModel(eos_id=0, include_eos=False)
        # Use encoder that returns text containing the stop sequence
        encoder = FakeEncoder(decode_result="hello world stop here")
        prepared = PreparedLMModel(
            model=model,
            tokenizer_id="test-tok",
            eos_id=0,
            pad_id=1,
            max_seq_len=128,
            tok_for_dataset=encoder,
        )

        cfg = make_generate_config(
            prompt_text="Hello",
            temperature=0.7,
            stop_sequences=["stop"],
        )
        settings = self._make_settings(tmp_path, settings_factory)

        result = generate_hf_lm(prepared=prepared, cfg=cfg, settings=settings)

        outputs = list(result["outputs"])
        first_output = outputs[0]
        # Text should be truncated at "stop"
        assert first_output == "hello world "
