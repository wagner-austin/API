"""Tests for backend_factory module."""

from __future__ import annotations

from collections.abc import Generator

import pytest

from model_trainer.core.services.finetuning.strategies._test_hooks import (
    reset_hooks as reset_ft_hooks,
)
from model_trainer.core.services.model.backend_factory import (
    create_hf_lm_backend,
    hf_lm_backend_funcs,
)
from model_trainer.core.services.model.backends.hf_lm._test_hooks import (
    reset_hooks as reset_hf_lm_hooks,
)


@pytest.fixture(autouse=True)
def _reset_all_hooks() -> Generator[None, None, None]:
    """Reset hooks before and after each test."""
    reset_hf_lm_hooks()
    reset_ft_hooks()
    yield
    reset_hf_lm_hooks()
    reset_ft_hooks()


class TestHfLmBackendFuncs:
    """Tests for hf_lm_backend_funcs function."""

    def test_returns_backend_funcs_with_correct_name(self) -> None:
        """Test that hf_lm_backend_funcs returns BackendFuncs with correct name."""
        funcs = hf_lm_backend_funcs()
        assert funcs["name"] == "hf_lm"

    def test_returns_backend_funcs_with_callable_prepare(self) -> None:
        """Test that hf_lm_backend_funcs returns BackendFuncs with callable prepare."""
        funcs = hf_lm_backend_funcs()
        assert callable(funcs["prepare"])

    def test_returns_backend_funcs_with_callable_save(self) -> None:
        """Test that hf_lm_backend_funcs returns BackendFuncs with callable save."""
        funcs = hf_lm_backend_funcs()
        assert callable(funcs["save"])

    def test_returns_backend_funcs_with_callable_load(self) -> None:
        """Test that hf_lm_backend_funcs returns BackendFuncs with callable load."""
        funcs = hf_lm_backend_funcs()
        assert callable(funcs["load"])

    def test_returns_backend_funcs_with_callable_train(self) -> None:
        """Test that hf_lm_backend_funcs returns BackendFuncs with callable train."""
        funcs = hf_lm_backend_funcs()
        assert callable(funcs["train"])

    def test_returns_backend_funcs_with_callable_evaluate(self) -> None:
        """Test that hf_lm_backend_funcs returns BackendFuncs with callable evaluate."""
        funcs = hf_lm_backend_funcs()
        assert callable(funcs["evaluate"])

    def test_returns_backend_funcs_with_callable_score(self) -> None:
        """Test that hf_lm_backend_funcs returns BackendFuncs with callable score."""
        funcs = hf_lm_backend_funcs()
        assert callable(funcs["score"])

    def test_returns_backend_funcs_with_callable_generate(self) -> None:
        """Test that hf_lm_backend_funcs returns BackendFuncs with callable generate."""
        funcs = hf_lm_backend_funcs()
        assert callable(funcs["generate"])


class TestCreateHfLmBackend:
    """Tests for create_hf_lm_backend function."""

    def test_creates_model_backend_with_capabilities(self) -> None:
        """Test that create_hf_lm_backend returns a ModelBackend with capabilities."""
        from model_trainer.core.contracts.dataset import DatasetConfig

        class _FakeDatasetBuilder:
            """Fake dataset builder for testing."""

            def split(self, cfg: DatasetConfig) -> tuple[list[str], list[str], list[str]]:
                return [], [], []

        backend = create_hf_lm_backend(_FakeDatasetBuilder())
        caps = backend.capabilities()
        assert caps["supports_score"] is True
        assert caps["supports_generate"] is True
        assert caps["supports_evaluate"] is True
