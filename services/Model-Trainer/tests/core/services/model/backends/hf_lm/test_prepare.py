"""Tests for HuggingFace LM prepare module."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from model_trainer.core.contracts.model import QuantizationConfig
from model_trainer.core.services.finetuning.strategies._test_hooks import (
    reset_hooks as reset_ft_hooks,
)
from model_trainer.core.services.model.backends.hf_lm._test_hooks import (
    HFTokenizerProto,
    Hooks,
    reset_hooks,
)
from model_trainer.core.services.model.backends.hf_lm.prepare import (
    HFTokenizerEncoder,
    _require_finetuning_strategy,
    _require_hub_model_id,
    _token_ids_from_hf_tokenizer,
    prepare_hf_lm_with_handle,
)
from model_trainer.core.types import LMModelProto

from .testing import FakeHFModel, FakeHFTokenizer, make_test_config


class _FakeModelLoader:
    """Fake model loader for testing."""

    def __call__(
        self, model_id_or_path: str, quantization: QuantizationConfig | None
    ) -> LMModelProto:
        return FakeHFModel(model_id_or_path)


class _FakeTokenizerLoader:
    """Fake tokenizer loader for testing."""

    def __call__(self, model_id_or_path: str) -> HFTokenizerProto:
        return FakeHFTokenizer()


class _CapturingModelLoader:
    """Model loader that captures called model IDs."""

    def __init__(self) -> None:
        self.captured: list[str] = []

    def __call__(
        self, model_id_or_path: str, quantization: QuantizationConfig | None
    ) -> LMModelProto:
        self.captured.append(model_id_or_path)
        return FakeHFModel(model_id_or_path)


class _CapturingTokenizerLoader:
    """Tokenizer loader that captures called model IDs."""

    def __init__(self) -> None:
        self.captured: list[str] = []

    def __call__(self, model_id_or_path: str) -> HFTokenizerProto:
        self.captured.append(model_id_or_path)
        return FakeHFTokenizer()


@pytest.fixture(autouse=True)
def _reset_all_hooks() -> Generator[None, None, None]:
    """Reset hooks before and after each test."""
    reset_hooks()
    reset_ft_hooks()
    yield
    reset_hooks()
    reset_ft_hooks()


class TestRequireHubModelId:
    """Tests for _require_hub_model_id function."""

    def test_returns_hub_model_id_when_present(self) -> None:
        """Test extraction of hub_model_id from config."""
        cfg = make_test_config(hub_model_id="test/model")
        result = _require_hub_model_id(cfg)
        assert result == "test/model"

    def test_raises_when_hub_model_id_is_none(self) -> None:
        """Test that ValueError is raised when hub_model_id is None."""
        cfg = make_test_config(hub_model_id=None)
        with pytest.raises(ValueError, match="hub_model_id is required"):
            _require_hub_model_id(cfg)


class TestRequireFineTuningStrategy:
    """Tests for _require_finetuning_strategy function."""

    def test_returns_strategy_when_present(self) -> None:
        """Test extraction of finetuning_strategy from config."""
        cfg = make_test_config(finetuning_strategy="lora")
        result = _require_finetuning_strategy(cfg)
        assert result == "lora"

    def test_returns_all_valid_strategies(self) -> None:
        """Test that all valid strategy names are accepted."""
        for strategy in ("full", "lora", "qlora"):
            cfg = make_test_config(finetuning_strategy=strategy)
            result = _require_finetuning_strategy(cfg)
            assert result == strategy


class TestTokenIdsFromHFTokenizer:
    """Tests for _token_ids_from_hf_tokenizer function."""

    def test_extracts_token_ids(self) -> None:
        """Test extraction of eos, pad, vocab_size from tokenizer."""
        tok = FakeHFTokenizer(vocab_size=1000)
        eos_id, pad_id, vocab_size = _token_ids_from_hf_tokenizer(tok)
        assert eos_id == 0
        assert pad_id == 1
        assert vocab_size == 1000


class TestHFTokenizerEncoder:
    """Tests for HFTokenizerEncoder class."""

    def test_encode_returns_encoded_with_ids(self) -> None:
        """Test that encode returns Encoded object with ids."""
        tok = FakeHFTokenizer()
        encoder = HFTokenizerEncoder(tok)
        result = encoder.encode("hello")
        # Access ids directly - will raise AttributeError if missing
        assert len(result.ids) == 5

    def test_decode_returns_string(self) -> None:
        """Test that decode returns a string."""
        tok = FakeHFTokenizer()
        encoder = HFTokenizerEncoder(tok)
        result = encoder.decode([32, 33, 34])
        assert type(result) is str

    def test_token_to_id_returns_int(self) -> None:
        """Test that token_to_id returns an integer."""
        tok = FakeHFTokenizer()
        encoder = HFTokenizerEncoder(tok)
        result = encoder.token_to_id("a")
        assert type(result) is int

    def test_get_vocab_size_returns_int(self) -> None:
        """Test that get_vocab_size returns vocabulary size."""
        tok = FakeHFTokenizer(vocab_size=5000)
        encoder = HFTokenizerEncoder(tok)
        result = encoder.get_vocab_size()
        assert result == 5000


class TestPrepareHFLMWithHandle:
    """Tests for prepare_hf_lm_with_handle function."""

    def test_returns_prepared_model_with_full_strategy(self) -> None:
        """Test that prepare_hf_lm_with_handle returns PreparedLMModel."""
        from model_trainer.core.contracts.tokenizer import TokenizerHandle

        class _FakeTokHandle(TokenizerHandle):
            def encode(self, text: str) -> list[int]:
                return []

            def decode(self, ids: list[int]) -> str:
                return ""

            def token_to_id(self, token: str) -> int | None:
                return 0

            def get_vocab_size(self) -> int:
                return 100

        model_loader = _CapturingModelLoader()
        tok_loader = _CapturingTokenizerLoader()

        Hooks.load_hf_model = model_loader
        Hooks.load_hf_tokenizer = tok_loader

        cfg = make_test_config(finetuning_strategy="full", hub_model_id="test/base-model")
        tok = _FakeTokHandle()

        result = prepare_hf_lm_with_handle(tok, cfg)

        assert len(model_loader.captured) == 1
        assert model_loader.captured[0] == "test/base-model"
        assert len(tok_loader.captured) == 1
        assert tok_loader.captured[0] == "test/base-model"

        assert result.strategy_name == "full"
        assert result.hub_model_id == "test/base-model"
        assert result.is_peft is False
        assert result.tokenizer_id == "test-tok"
