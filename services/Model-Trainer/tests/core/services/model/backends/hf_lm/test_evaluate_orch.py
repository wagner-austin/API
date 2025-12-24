"""Tests for HuggingFace LM evaluate module."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Protocol

import pytest

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.dataset import DatasetConfig
from model_trainer.core.contracts.model import PreparedLMModel
from model_trainer.core.contracts.tokenizer import TokenizerHandle
from model_trainer.core.encoding import Encoder
from model_trainer.core.services.model.backends.hf_lm._test_hooks import (
    CausalLMDatasetProto,
    DataLoaderProto,
    Hooks,
    reset_hooks,
)
from model_trainer.core.services.model.backends.hf_lm.evaluate import (
    EvalResult,
    _get_autocast_context,
    evaluate_hf_lm,
)

from .testing import (
    FakeDataLoader,
    FakeDataset,
    FakeEncoder,
    FakeEvalModel,
    FakeHFModel,
    FakeTokenizerHandle,
    make_test_config,
)


class _SettingsFactory(Protocol):
    def __call__(
        self,
        *,
        artifacts_root: str | None = None,
        data_root: str | None = None,
    ) -> Settings: ...


class _FakeDatasetBuilder:
    """Fake dataset builder for testing."""

    def __init__(self, val_files: list[str]) -> None:
        """Initialize.

        Args:
            val_files: Validation file paths.
        """
        self._val_files = val_files

    def split(self, cfg: DatasetConfig) -> tuple[list[str], list[str], list[str]]:
        """Split corpus.

        Args:
            cfg: Dataset config.

        Returns:
            Train, val, test file lists.
        """
        return [], self._val_files, []


class TestGetAutocastContext:
    """Tests for _get_autocast_context function."""

    def test_returns_nullcontext_for_fp32(self) -> None:
        """Test that fp32 returns nullcontext."""
        ctx = _get_autocast_context("fp32", "cpu")
        assert type(ctx) is type(nullcontext())

    def test_returns_nullcontext_for_cpu_fp16(self) -> None:
        """Test that CPU with fp16 returns nullcontext."""
        ctx = _get_autocast_context("fp16", "cpu")
        assert type(ctx) is type(nullcontext())

    def test_returns_nullcontext_for_cpu_bf16(self) -> None:
        """Test that CPU with bf16 returns nullcontext."""
        ctx = _get_autocast_context("bf16", "cpu")
        assert type(ctx) is type(nullcontext())

    def test_returns_autocast_for_cuda_fp16(self) -> None:
        """Test that CUDA with fp16 returns autocast context."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ctx = _get_autocast_context("fp16", "cuda")
            # Should not be nullcontext - it's a torch.amp.autocast
            assert type(ctx) is not type(nullcontext())

    def test_returns_autocast_for_cuda_bf16(self) -> None:
        """Test that CUDA with bf16 returns autocast context."""
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ctx = _get_autocast_context("bf16", "cuda")
            # Should not be nullcontext - it's a torch.amp.autocast
            assert type(ctx) is not type(nullcontext())


class TestEvalResult:
    """Tests for EvalResult class."""

    def test_stores_loss_and_perplexity(self) -> None:
        """Test that values are stored."""
        result = EvalResult(loss=1.5, perplexity=4.48)
        assert result.loss == 1.5
        assert result.perplexity == 4.48


class TestEvaluateHfLm:
    """Tests for evaluate_hf_lm function."""

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

    def test_raises_when_load_bpe_hook_not_initialized(
        self, tmp_path: Path, settings_factory: _SettingsFactory
    ) -> None:
        """Test error when load_bpe_tokenizer hook is None."""
        cfg = make_test_config()
        settings = self._make_settings(tmp_path, settings_factory)
        dataset_builder = _FakeDatasetBuilder([])

        with pytest.raises(RuntimeError, match=r"Hooks\.load_bpe_tokenizer not initialized"):
            evaluate_hf_lm(
                run_id="test-run",
                cfg=cfg,
                settings=settings,
                dataset_builder=dataset_builder,
            )

    def test_raises_when_load_prepared_model_hook_not_initialized(
        self, tmp_path: Path, settings_factory: _SettingsFactory
    ) -> None:
        """Test error when load_prepared_model hook is None."""
        Hooks.load_bpe_tokenizer = lambda p: FakeTokenizerHandle()

        cfg = make_test_config()
        settings = self._make_settings(tmp_path, settings_factory)
        dataset_builder = _FakeDatasetBuilder([])

        with pytest.raises(RuntimeError, match=r"Hooks\.load_prepared_model not initialized"):
            evaluate_hf_lm(
                run_id="test-run",
                cfg=cfg,
                settings=settings,
                dataset_builder=dataset_builder,
            )

    def test_raises_when_get_model_dir_hook_not_initialized(
        self, tmp_path: Path, settings_factory: _SettingsFactory
    ) -> None:
        """Test error when get_model_dir hook is None."""
        Hooks.load_bpe_tokenizer = lambda p: FakeTokenizerHandle()
        Hooks.load_prepared_model = lambda p, t: PreparedLMModel(
            model=FakeHFModel(),
            tokenizer_id="test",
            eos_id=0,
            pad_id=1,
            max_seq_len=128,
            tok_for_dataset=FakeEncoder(),
        )

        cfg = make_test_config()
        settings = self._make_settings(tmp_path, settings_factory)
        dataset_builder = _FakeDatasetBuilder([])

        with pytest.raises(RuntimeError, match=r"Hooks\.get_model_dir not initialized"):
            evaluate_hf_lm(
                run_id="test-run",
                cfg=cfg,
                settings=settings,
                dataset_builder=dataset_builder,
            )

    def test_raises_when_get_eval_dir_hook_not_initialized(
        self, tmp_path: Path, settings_factory: _SettingsFactory
    ) -> None:
        """Test error when get_eval_dir hook is None."""
        Hooks.load_bpe_tokenizer = lambda p: FakeTokenizerHandle()
        Hooks.load_prepared_model = lambda p, t: PreparedLMModel(
            model=FakeHFModel(),
            tokenizer_id="test",
            eos_id=0,
            pad_id=1,
            max_seq_len=128,
            tok_for_dataset=FakeEncoder(),
        )
        Hooks.get_model_dir = lambda s, r: Path("/tmp/model")

        cfg = make_test_config()
        settings = self._make_settings(tmp_path, settings_factory)
        dataset_builder = _FakeDatasetBuilder([])

        with pytest.raises(RuntimeError, match=r"Hooks\.get_eval_dir not initialized"):
            evaluate_hf_lm(
                run_id="test-run",
                cfg=cfg,
                settings=settings,
                dataset_builder=dataset_builder,
            )

    def test_raises_when_create_causal_dataset_hook_not_initialized(
        self, tmp_path: Path, settings_factory: _SettingsFactory
    ) -> None:
        """Test error when create_causal_dataset hook is None."""
        Hooks.load_bpe_tokenizer = lambda p: FakeTokenizerHandle()
        Hooks.load_prepared_model = lambda p, t: PreparedLMModel(
            model=FakeHFModel(),
            tokenizer_id="test",
            eos_id=0,
            pad_id=1,
            max_seq_len=128,
            tok_for_dataset=FakeEncoder(),
        )
        Hooks.get_model_dir = lambda s, r: Path("/tmp/model")
        Hooks.get_eval_dir = lambda s, r: Path("/tmp/eval")

        cfg = make_test_config()
        settings = self._make_settings(tmp_path, settings_factory)
        dataset_builder = _FakeDatasetBuilder([])

        with pytest.raises(RuntimeError, match=r"Hooks\.create_causal_dataset not initialized"):
            evaluate_hf_lm(
                run_id="test-run",
                cfg=cfg,
                settings=settings,
                dataset_builder=dataset_builder,
            )

    def test_raises_when_create_dataloader_hook_not_initialized(
        self, tmp_path: Path, settings_factory: _SettingsFactory
    ) -> None:
        """Test error when create_dataloader hook is None."""
        Hooks.load_bpe_tokenizer = lambda p: FakeTokenizerHandle()
        Hooks.load_prepared_model = lambda p, t: PreparedLMModel(
            model=FakeHFModel(),
            tokenizer_id="test",
            eos_id=0,
            pad_id=1,
            max_seq_len=128,
            tok_for_dataset=FakeEncoder(),
        )
        Hooks.get_model_dir = lambda s, r: Path("/tmp/model")
        Hooks.get_eval_dir = lambda s, r: Path("/tmp/eval")
        Hooks.create_causal_dataset = (
            lambda *, files, tokenizer, max_len, eos_id, pad_id: FakeDataset()
        )

        cfg = make_test_config()
        settings = self._make_settings(tmp_path, settings_factory)
        dataset_builder = _FakeDatasetBuilder([])

        with pytest.raises(RuntimeError, match=r"Hooks\.create_dataloader not initialized"):
            evaluate_hf_lm(
                run_id="test-run",
                cfg=cfg,
                settings=settings,
                dataset_builder=dataset_builder,
            )

    def test_evaluates_model_and_returns_result(
        self, tmp_path: Path, settings_factory: _SettingsFactory
    ) -> None:
        """Test full evaluation with all hooks set."""
        eval_dir = tmp_path / "eval"

        class _FakeModelLoader:
            """Callable class that implements PreparedModelLoader Protocol."""

            def __call__(
                self, model_path: str, tokenizer_handle: TokenizerHandle | None
            ) -> PreparedLMModel:
                return PreparedLMModel(
                    model=FakeEvalModel(loss_value=1.0),
                    tokenizer_id="test",
                    eos_id=0,
                    pad_id=1,
                    max_seq_len=128,
                    tok_for_dataset=FakeEncoder(),
                )

        def fake_create_dataset(
            *,
            files: Sequence[str],
            tokenizer: Encoder,
            max_len: int,
            eos_id: int,
            pad_id: int,
        ) -> CausalLMDatasetProto:
            return FakeDataset(num_samples=4)

        def fake_create_loader(
            dataset: CausalLMDatasetProto,
            *,
            batch_size: int,
            shuffle: bool,
            num_workers: int,
            pin_memory: bool,
        ) -> DataLoaderProto:
            return FakeDataLoader(dataset, batch_size)

        Hooks.load_bpe_tokenizer = lambda p: FakeTokenizerHandle()
        Hooks.load_prepared_model = _FakeModelLoader()
        Hooks.get_model_dir = lambda s, r: Path("/tmp/model")
        Hooks.get_eval_dir = lambda s, r: eval_dir
        Hooks.create_causal_dataset = fake_create_dataset
        Hooks.create_dataloader = fake_create_loader

        cfg = make_test_config()
        settings = settings_factory(
            artifacts_root=str(tmp_path),
            data_root=str(tmp_path),
        )
        dataset_builder = _FakeDatasetBuilder(["val_file.txt"])

        result = evaluate_hf_lm(
            run_id="test-run",
            cfg=cfg,
            settings=settings,
            dataset_builder=dataset_builder,
        )

        assert result.loss == 1.0
        assert abs(result.perplexity - 2.718) < 0.1

        metrics_file = eval_dir / "metrics.json"
        assert metrics_file.exists()

    def test_handles_empty_dataset(
        self, tmp_path: Path, settings_factory: _SettingsFactory
    ) -> None:
        """Test evaluation with empty dataset."""
        eval_dir = tmp_path / "eval"

        def fake_create_dataset(
            *,
            files: Sequence[str],
            tokenizer: Encoder,
            max_len: int,
            eos_id: int,
            pad_id: int,
        ) -> CausalLMDatasetProto:
            return FakeDataset(num_samples=0)

        def fake_create_loader(
            dataset: CausalLMDatasetProto,
            *,
            batch_size: int,
            shuffle: bool,
            num_workers: int,
            pin_memory: bool,
        ) -> DataLoaderProto:
            return FakeDataLoader(dataset, batch_size)

        Hooks.load_bpe_tokenizer = lambda p: FakeTokenizerHandle()
        Hooks.load_prepared_model = lambda p, t: PreparedLMModel(
            model=FakeEvalModel(),
            tokenizer_id="test",
            eos_id=0,
            pad_id=1,
            max_seq_len=128,
            tok_for_dataset=FakeEncoder(),
        )
        Hooks.get_model_dir = lambda s, r: Path("/tmp/model")
        Hooks.get_eval_dir = lambda s, r: eval_dir
        Hooks.create_causal_dataset = fake_create_dataset
        Hooks.create_dataloader = fake_create_loader

        cfg = make_test_config()
        settings = settings_factory(
            artifacts_root=str(tmp_path),
            data_root=str(tmp_path),
        )
        dataset_builder = _FakeDatasetBuilder([])

        result = evaluate_hf_lm(
            run_id="test-run",
            cfg=cfg,
            settings=settings,
            dataset_builder=dataset_builder,
        )

        assert result.loss == 0.0
        assert result.perplexity == 1.0

    def test_evaluates_model_with_tokenizer_id_none(
        self, tmp_path: Path, settings_factory: _SettingsFactory
    ) -> None:
        """Test evaluation with tokenizer_id=None (hf_lm case)."""
        eval_dir = tmp_path / "eval"

        class _FakeModelLoader:
            """Callable class that implements PreparedModelLoader Protocol."""

            def __call__(
                self, model_path: str, tokenizer_handle: TokenizerHandle | None
            ) -> PreparedLMModel:
                # Verify tokenizer_handle is None when tokenizer_id is None
                assert tokenizer_handle is None
                return PreparedLMModel(
                    model=FakeEvalModel(loss_value=0.5),
                    tokenizer_id=None,
                    eos_id=0,
                    pad_id=1,
                    max_seq_len=128,
                    tok_for_dataset=FakeEncoder(),
                )

        def fake_create_dataset(
            *,
            files: Sequence[str],
            tokenizer: Encoder,
            max_len: int,
            eos_id: int,
            pad_id: int,
        ) -> CausalLMDatasetProto:
            return FakeDataset(num_samples=2)

        def fake_create_loader(
            dataset: CausalLMDatasetProto,
            *,
            batch_size: int,
            shuffle: bool,
            num_workers: int,
            pin_memory: bool,
        ) -> DataLoaderProto:
            return FakeDataLoader(dataset, batch_size)

        # load_bpe_tokenizer should not be called when tokenizer_id is None
        class _FailingTokenizerLoader:
            """Loader that fails if called."""

            def __call__(self: _FailingTokenizerLoader, path: str) -> TokenizerHandle:
                raise AssertionError("load_bpe_tokenizer should not be called")

        Hooks.load_bpe_tokenizer = _FailingTokenizerLoader()
        Hooks.load_prepared_model = _FakeModelLoader()
        Hooks.get_model_dir = lambda s, r: Path("/tmp/model")
        Hooks.get_eval_dir = lambda s, r: eval_dir
        Hooks.create_causal_dataset = fake_create_dataset
        Hooks.create_dataloader = fake_create_loader

        # Use tokenizer_id=None to exercise the else branch at line 139
        cfg = make_test_config(tokenizer_id=None)
        settings = settings_factory(
            artifacts_root=str(tmp_path),
            data_root=str(tmp_path),
        )
        dataset_builder = _FakeDatasetBuilder(["val_file.txt"])

        result = evaluate_hf_lm(
            run_id="test-run",
            cfg=cfg,
            settings=settings,
            dataset_builder=dataset_builder,
        )

        assert result.loss == 0.5
        assert abs(result.perplexity - 1.648) < 0.1
