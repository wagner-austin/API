"""Tests for HuggingFace LM train module."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from platform_core.determinism_record import DeterminismRecord
from platform_ml.wandb_publisher import WandbPublisher
from tests.conftest import UNPINNED

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import ModelTrainConfig, PreparedLMModel, TrainOutcome
from model_trainer.core.services.model.backends.hf_lm._test_hooks import (
    Hooks,
    ProgressCallback,
    TrainerProto,
    reset_hooks,
)
from model_trainer.core.services.model.backends.hf_lm.train import train_prepared_hf_lm

from .testing import FakeEncoder, FakeHFModel, make_test_config


class _SettingsFactory(Protocol):
    def __call__(
        self,
        *,
        artifacts_root: str | None = None,
        data_root: str | None = None,
    ) -> Settings: ...


class _FakeTrainer:
    """Fake trainer for testing."""

    def __init__(
        self,
        prepared: PreparedLMModel,
        cfg: ModelTrainConfig,
        settings: Settings,
        *,
        run_id: str,
        redis_hb: Callable[[float], None],
        cancelled: Callable[[], bool],
        progress: ProgressCallback | None,
        service_name: str,
        wandb_publisher: WandbPublisher | None,
    ) -> None:
        """Store all args for verification."""
        self.prepared = prepared
        self.cfg = cfg
        self.settings = settings
        self.run_id = run_id
        self.redis_hb = redis_hb
        self.cancelled = cancelled
        self.progress = progress
        self.service_name = service_name
        self.wandb_publisher = wandb_publisher
        self.train_called = False

    def train(self) -> TrainOutcome:
        """Return fake outcome."""
        self.train_called = True
        return TrainOutcome(
            loss=0.5,
            perplexity=1.65,
            steps=100,
            out_dir="/tmp/model",
            cancelled=False,
            test_loss=0.6,
            test_perplexity=1.82,
            best_val_loss=0.55,
            early_stopped=False,
        )


class _CapturingTrainerFactory:
    """Factory that captures args and returns fake trainer."""

    def __init__(self) -> None:
        """Initialize."""
        self.captured_trainer: _FakeTrainer | None = None

    def __call__(
        self,
        prepared: PreparedLMModel,
        cfg: ModelTrainConfig,
        settings: Settings,
        *,
        run_id: str,
        redis_hb: Callable[[float], None],
        cancelled: Callable[[], bool],
        resume: bool,
        progress: ProgressCallback | None,
        service_name: str,
        wandb_publisher: WandbPublisher | None,
        determinism: DeterminismRecord | None,
    ) -> TrainerProto:
        """Create and capture trainer."""
        trainer = _FakeTrainer(
            prepared,
            cfg,
            settings,
            run_id=run_id,
            redis_hb=redis_hb,
            cancelled=cancelled,
            progress=progress,
            service_name=service_name,
            wandb_publisher=wandb_publisher,
        )
        self.captured_trainer = trainer
        return trainer


def _require_trainer(factory: _CapturingTrainerFactory) -> _FakeTrainer:
    """Get captured trainer, raising if None.

    Args:
        factory: Factory with captured trainer.

    Returns:
        Captured trainer.

    Raises:
        AssertionError: If no trainer was captured.
    """
    trainer = factory.captured_trainer
    if trainer is None:
        raise AssertionError("No trainer was captured")
    return trainer


class TestTrainPreparedHfLm:
    """Tests for train_prepared_hf_lm function."""

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

    def test_calls_trainer_train(self, tmp_path: Path, settings_factory: _SettingsFactory) -> None:
        """Test that trainer.train() is called and result returned."""
        factory = _CapturingTrainerFactory()
        Hooks.create_trainer = factory

        prepared = PreparedLMModel(
            model=FakeHFModel(),
            tokenizer_id="test-tok",
            eos_id=0,
            pad_id=1,
            max_seq_len=128,
            tok_for_dataset=FakeEncoder(),
        )
        cfg = make_test_config()
        settings = self._make_settings(tmp_path, settings_factory)

        def heartbeat(ts: float) -> None:
            pass

        def cancelled() -> bool:
            return False

        result = train_prepared_hf_lm(
            prepared,
            cfg,
            settings,
            run_id="run-123",
            redis_hb=heartbeat,
            cancelled=cancelled,
            resume=False,
            determinism=UNPINNED,
        )

        trainer = _require_trainer(factory)
        assert trainer.train_called is True
        assert trainer.run_id == "run-123"
        assert trainer.service_name == "hf-lm-train"
        assert result["loss"] == 0.5
        assert result["perplexity"] == 1.65
        assert result["steps"] == 100

    def test_passes_progress_callback(
        self, tmp_path: Path, settings_factory: _SettingsFactory
    ) -> None:
        """Test that progress callback is passed through."""
        factory = _CapturingTrainerFactory()
        Hooks.create_trainer = factory

        prepared = PreparedLMModel(
            model=FakeHFModel(),
            tokenizer_id="test-tok",
            eos_id=0,
            pad_id=1,
            max_seq_len=128,
            tok_for_dataset=FakeEncoder(),
        )
        cfg = make_test_config()
        settings = self._make_settings(tmp_path, settings_factory)

        progress_calls: list[
            tuple[int, int, float, float, float, float, float | None, float | None]
        ] = []

        def progress(
            step: int,
            epoch: int,
            loss: float,
            grad_norm: float,
            samples_per_sec: float,
            lr: float,
            val_loss: float | None,
            val_ppl: float | None,
        ) -> None:
            progress_calls.append(
                (step, epoch, loss, grad_norm, samples_per_sec, lr, val_loss, val_ppl)
            )

        train_prepared_hf_lm(
            prepared,
            cfg,
            settings,
            run_id="run-456",
            redis_hb=lambda _: None,
            cancelled=lambda: False,
            resume=False,
            progress=progress,
            determinism=UNPINNED,
        )

        trainer = _require_trainer(factory)
        assert trainer.progress is progress

    def test_passes_wandb_publisher_none(
        self, tmp_path: Path, settings_factory: _SettingsFactory
    ) -> None:
        """Test that None wandb publisher is passed through."""
        factory = _CapturingTrainerFactory()
        Hooks.create_trainer = factory

        prepared = PreparedLMModel(
            model=FakeHFModel(),
            tokenizer_id="test-tok",
            eos_id=0,
            pad_id=1,
            max_seq_len=128,
            tok_for_dataset=FakeEncoder(),
        )
        cfg = make_test_config()
        settings = self._make_settings(tmp_path, settings_factory)

        train_prepared_hf_lm(
            prepared,
            cfg,
            settings,
            run_id="run-789",
            redis_hb=lambda _: None,
            cancelled=lambda: False,
            resume=False,
            wandb_publisher=None,
            determinism=UNPINNED,
        )

        trainer = _require_trainer(factory)
        assert trainer.wandb_publisher is None
