from __future__ import annotations

import pytest
from platform_core.errors import AppError

from model_trainer.core.contracts.dataset import DatasetBuilder
from model_trainer.core.contracts.model import ModelBackend
from model_trainer.core.services.dataset.local_text_builder import LocalTextDatasetBuilder
from model_trainer.core.services.model.backend_factory import (
    CHAR_LSTM_CAPABILITIES,
    GPT2_CAPABILITIES,
    HF_LM_CAPABILITIES,
    create_char_lstm_backend,
    create_gpt2_backend,
)
from model_trainer.core.services.model.backends.char_lstm.prepare import _size_to_dims
from model_trainer.core.services.model.backends.gpt2.hf_gpt2 import create_gpt2_config
from model_trainer.core.services.model.model_sizes import CHAR_LSTM_MODEL_SIZES, GPT2_MODEL_SIZES
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
    # Asserted as a RELATION against the implementation, not as a copy of the
    # literal. The previous form (== a hand-typed tuple) proved only that the
    # constant was what someone typed, and passed happily while the registry
    # advertised a "tiny" the table did not implement and hid an "xl" it did.
    assert GPT2_CAPABILITIES["supported_sizes"] == tuple(GPT2_MODEL_SIZES)

    # CharLSTM capabilities
    assert CHAR_LSTM_CAPABILITIES["supports_train"] is True
    assert CHAR_LSTM_CAPABILITIES["supports_evaluate"] is True
    assert CHAR_LSTM_CAPABILITIES["supports_score"] is True
    assert CHAR_LSTM_CAPABILITIES["supports_generate"] is True
    assert CHAR_LSTM_CAPABILITIES["supports_distributed"] is False
    assert CHAR_LSTM_CAPABILITIES["supported_sizes"] == tuple(CHAR_LSTM_MODEL_SIZES)

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
    assert caps["supported_sizes"] == tuple(GPT2_MODEL_SIZES)


def test_tokenizer_registry_missing() -> None:
    reg = TokenizerRegistry(backends={})
    with pytest.raises(AppError):
        _ = reg.get("nope")


# --- Capability declarations must be true, not merely well-formed -------------
#
# The tests above assert each capability against the implementation it describes.
# These assert the advertisement END TO END: every size a backend advertises must
# actually resolve. That is the check the original defect needed -- the registry
# advertised a GPT-2 "tiny" that was absent from the size table, so the only way
# to find out was to ask for it and take a bare KeyError.


def test_every_advertised_gpt2_size_resolves() -> None:
    """Each advertised GPT-2 size builds a config with the table's architecture."""
    advertised = GPT2_CAPABILITIES["supported_sizes"]
    assert advertised, "GPT-2 must advertise at least one size"
    for size in advertised:
        cfg = create_gpt2_config(vocab_size=128, max_seq_len=16, model_size=size)
        expected = GPT2_MODEL_SIZES[size]
        assert cfg.n_embd == expected["hidden_size"]
        assert cfg.n_layer == expected["n_layer"]
        assert cfg.n_head == expected["n_head"]


def test_unknown_gpt2_size_raises_apperror() -> None:
    """An unadvertised size is a typed rejection, not a bare KeyError."""
    with pytest.raises(AppError):
        _ = create_gpt2_config(vocab_size=128, max_seq_len=16, model_size="gargantuan")


def test_every_advertised_char_lstm_size_resolves() -> None:
    """Each advertised char-LSTM size resolves to the table's dimensions."""
    advertised = CHAR_LSTM_CAPABILITIES["supported_sizes"]
    assert advertised, "char_lstm must advertise at least one size"
    for size in advertised:
        expected = CHAR_LSTM_MODEL_SIZES[size]
        assert _size_to_dims(size) == (
            expected["embed_dim"],
            expected["hidden_dim"],
            expected["num_layers"],
            expected["dropout"],
        )


def test_unknown_char_lstm_size_raises_apperror() -> None:
    with pytest.raises(AppError):
        _ = _size_to_dims("gargantuan")


def test_sizeless_backends_advertise_no_sizes() -> None:
    """hf_lm takes its size from hub_model_id, so an empty tuple is the honest claim."""
    assert HF_LM_CAPABILITIES["supported_sizes"] == ()
    assert UNAVAILABLE_CAPABILITIES["supported_sizes"] == ()
