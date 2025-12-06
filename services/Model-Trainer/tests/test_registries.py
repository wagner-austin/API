from __future__ import annotations

import pytest
from platform_core.errors import AppError

from model_trainer.core.contracts.dataset import DatasetBuilder
from model_trainer.core.contracts.model import ModelBackend
from model_trainer.core.services.dataset.local_text_builder import LocalTextDatasetBuilder
from model_trainer.core.services.model.backend_factory import (
    CHAR_LSTM_CAPABILITIES,
    GPT2_CAPABILITIES,
    create_char_lstm_backend,
    create_gpt2_backend,
)
from model_trainer.core.services.model.unavailable_backend import (
    UNAVAILABLE_CAPABILITIES,
    UnavailableBackend,
)
from model_trainer.core.services.registries import (
    BackendRegistration,
    ModelRegistry,
    TokenizerRegistry,
)


def test_model_registry_get_and_missing() -> None:
    ds = LocalTextDatasetBuilder()
    reg = ModelRegistry(
        registrations={
            "llama": BackendRegistration(
                factory=lambda _: UnavailableBackend("llama"),
                capabilities=UNAVAILABLE_CAPABILITIES,
            ),
        },
        dataset_builder=ds,
    )
    b = reg.get("llama")
    assert b.name() == "llama"
    with pytest.raises(AppError):
        _ = reg.get("nope")


def test_model_registry_list_backends() -> None:
    ds = LocalTextDatasetBuilder()
    reg = ModelRegistry(
        registrations={
            "gpt2": BackendRegistration(
                factory=create_gpt2_backend,
                capabilities=GPT2_CAPABILITIES,
            ),
            "char_lstm": BackendRegistration(
                factory=create_char_lstm_backend,
                capabilities=CHAR_LSTM_CAPABILITIES,
            ),
        },
        dataset_builder=ds,
    )
    backends = reg.list_backends()
    assert "gpt2" in backends
    assert "char_lstm" in backends
    assert len(backends) == 2


def test_model_registry_get_capabilities_without_instantiation() -> None:
    """Test that capabilities can be queried without instantiating the backend."""
    instantiation_count = 0

    def counting_factory(ds: DatasetBuilder) -> ModelBackend:
        nonlocal instantiation_count
        instantiation_count += 1
        return UnavailableBackend("test")

    ds = LocalTextDatasetBuilder()
    reg = ModelRegistry(
        registrations={
            "test": BackendRegistration(
                factory=counting_factory,
                capabilities=UNAVAILABLE_CAPABILITIES,
            ),
        },
        dataset_builder=ds,
    )
    # Get capabilities - should NOT instantiate
    caps = reg.get_capabilities("test")
    assert caps["supports_train"] is False
    assert instantiation_count == 0

    # Now get the backend - should instantiate
    _ = reg.get("test")
    assert instantiation_count == 1


def test_model_registry_lazy_loading_and_caching() -> None:
    """Test that backends are lazily loaded and cached."""
    instantiation_count = 0

    def counting_factory(ds: DatasetBuilder) -> ModelBackend:
        nonlocal instantiation_count
        instantiation_count += 1
        return UnavailableBackend("test")

    ds = LocalTextDatasetBuilder()
    reg = ModelRegistry(
        registrations={
            "test": BackendRegistration(
                factory=counting_factory,
                capabilities=UNAVAILABLE_CAPABILITIES,
            ),
        },
        dataset_builder=ds,
    )
    # No instantiation yet
    assert instantiation_count == 0

    # First get - should instantiate
    b1 = reg.get("test")
    assert instantiation_count == 1

    # Second get - should return cached instance
    b2 = reg.get("test")
    assert instantiation_count == 1
    assert b1 is b2


def test_model_registry_get_capabilities_missing() -> None:
    ds = LocalTextDatasetBuilder()
    reg = ModelRegistry(registrations={}, dataset_builder=ds)
    with pytest.raises(AppError):
        _ = reg.get_capabilities("nope")


def test_backend_capabilities_values() -> None:
    """Test that capability constants have expected values."""
    # GPT2 capabilities
    assert GPT2_CAPABILITIES["supports_train"] is True
    assert GPT2_CAPABILITIES["supports_evaluate"] is True
    assert GPT2_CAPABILITIES["supports_score"] is True
    assert GPT2_CAPABILITIES["supports_generate"] is True
    assert GPT2_CAPABILITIES["supports_distributed"] is False
    assert GPT2_CAPABILITIES["supported_sizes"] == ("tiny", "small", "medium", "large")

    # CharLSTM capabilities
    assert CHAR_LSTM_CAPABILITIES["supports_train"] is True
    assert CHAR_LSTM_CAPABILITIES["supports_evaluate"] is True
    assert CHAR_LSTM_CAPABILITIES["supports_score"] is True
    assert CHAR_LSTM_CAPABILITIES["supports_generate"] is True
    assert CHAR_LSTM_CAPABILITIES["supports_distributed"] is False
    assert CHAR_LSTM_CAPABILITIES["supported_sizes"] == ("small",)

    # Unavailable capabilities
    assert UNAVAILABLE_CAPABILITIES["supports_train"] is False
    assert UNAVAILABLE_CAPABILITIES["supports_evaluate"] is False
    assert UNAVAILABLE_CAPABILITIES["supports_score"] is False
    assert UNAVAILABLE_CAPABILITIES["supports_generate"] is False
    assert UNAVAILABLE_CAPABILITIES["supports_distributed"] is False
    assert UNAVAILABLE_CAPABILITIES["supported_sizes"] == ()


def test_unavailable_backend_capabilities() -> None:
    """Test that UnavailableBackend.capabilities() returns correct values."""
    backend = UnavailableBackend("test_backend")
    caps = backend.capabilities()
    assert caps["supports_train"] is False
    assert caps["supports_evaluate"] is False
    assert caps["supports_score"] is False
    assert caps["supports_generate"] is False
    assert caps["supports_distributed"] is False
    assert caps["supported_sizes"] == ()


def test_factory_backend_capabilities() -> None:
    """Test that factory-created backends expose capabilities correctly."""
    ds = LocalTextDatasetBuilder()
    backend = create_gpt2_backend(ds)
    caps = backend.capabilities()
    assert caps["supports_train"] is True
    assert caps["supports_evaluate"] is True
    assert caps["supports_score"] is True
    assert caps["supports_generate"] is True
    assert caps["supports_distributed"] is False
    assert caps["supported_sizes"] == ("tiny", "small", "medium", "large")


def test_tokenizer_registry_missing() -> None:
    reg = TokenizerRegistry(backends={})
    with pytest.raises(AppError):
        _ = reg.get("nope")
