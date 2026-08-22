"""char_lstm backend impl branches."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from tests.core.services.model.backends.char_lstm._train_branches_support import (
    _LM,
    _make_cfg,
    _make_fake_wandb_module,
    _make_prepared,
    _make_settings,
    _MiniEnc,
    _setup_fake_wandb_hooks,
)

from model_trainer.core.contracts.model import ModelTrainConfig, PreparedLMModel
from model_trainer.core.services.training import base_trainer as bt
from model_trainer.core.services.training import trainer_grad_utils as bt_grad
from model_trainer.core.services.training.dataloader import DataLoader
from model_trainer.core.types import (
    ForwardOutProto,
)


def test_early_stopping_triggers_after_patience_exceeded() -> None:
    """Test early stopping triggers when epochs_no_improve >= patience (lines 516-531)."""

    class _DS:
        def __len__(self: _DS) -> int:
            return 1

        def __getitem__(self: _DS, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            vals: list[int] = [1, 1]
            ids = torch.tensor(vals, dtype=torch.long)
            return (ids, ids)

    class _ConstantLossLM(_LM):
        """Model that returns same loss every forward pass to prevent improvement."""

        def forward(
            self: _ConstantLossLM, *, input_ids: torch.Tensor, labels: torch.Tensor
        ) -> ForwardOutProto:
            class _Out(ForwardOutProto):
                @property
                def loss(self: _Out) -> torch.Tensor:
                    return torch.tensor(1.0, requires_grad=True)

            return _Out()

    cfg: ModelTrainConfig = {
        **_make_cfg(),
        "num_epochs": 10,
        "early_stopping_patience": 2,
    }

    prepared = PreparedLMModel(
        model=_ConstantLossLM(),
        tokenizer_id="tok",
        eos_id=1,
        pad_id=0,
        max_seq_len=8,
        tok_for_dataset=_MiniEnc(),
    )

    trainer = bt.BaseTrainer(
        prepared,
        cfg,
        _make_settings(),
        run_id="test-early-stop",
        redis_hb=lambda _: None,
        cancelled=lambda: False,
        resume=False,
        progress=None,
        service_name="char-lstm-train",
    )

    trainer._device = torch.device("cpu")
    trainer._es_state = {"best_val_loss": 0.5, "epochs_no_improve": 0}

    val_loader = DataLoader(_DS(), batch_size=1, shuffle=False)
    trainer._val_loader = val_loader

    train_loader = DataLoader(_DS(), batch_size=1, shuffle=False)

    _last_loss, _step, _was_cancelled, early_stopped = trainer._run_training_loop(
        model=_ConstantLossLM(),
        dataloader=train_loader,
        effective_lr=1e-3,
        start_epoch=0,
        start_step=0,
        initial_last_loss=0.0,
        restored=None,
    )

    assert early_stopped is True, "Expected early stopping to trigger"
    assert trainer._es_state["epochs_no_improve"] >= 2


def test_clip_grad_norm_legacy_function() -> None:
    """Test legacy _clip_grad_norm function (lines 761-762)."""
    vals: list[float] = [1.0, 2.0, 3.0]
    grad_vals: list[float] = [10.0, 20.0, 30.0]
    param = torch.nn.Parameter(torch.tensor(vals))
    param.grad = torch.tensor(grad_vals)

    bt_grad._clip_grad_norm([param], max_norm=1.0)

    grad_tensor: torch.Tensor = param.grad
    norm_tensor: torch.Tensor = torch.linalg.vector_norm(grad_tensor)
    grad_norm = float(norm_tensor.item())
    assert grad_norm <= 1.1, f"Expected gradient norm <= 1.1 after clipping, got {grad_norm}"


def test_apply_lr_cap_no_log_when_lr_below_cap() -> None:
    """Test _apply_lr_cap skips logging when LR is already below cap (line 330->340)."""
    cfg: ModelTrainConfig = {
        **_make_cfg(),
        "learning_rate": 1e-6,
        "pretrained_run_id": "base-run",
        "finetune_lr_cap": 5e-5,
        "loss_mask_prefix_separator": None,
    }

    trainer = bt.BaseTrainer(
        _make_prepared(),
        cfg,
        _make_settings(),
        run_id="test-no-log-cap",
        redis_hb=lambda _: None,
        cancelled=lambda: False,
        resume=False,
        progress=None,
        service_name="char-lstm-train",
    )

    effective_lr = trainer._apply_lr_cap()
    # LR is 1e-6 which is already below cap of 5e-5, so no capping occurs
    assert effective_lr == 1e-6, f"Expected LR to remain at 1e-6, got {effective_lr}"


def test_make_loader_returns_none_for_empty_files(tmp_path: Path) -> None:
    """Test internal make_loader returns None when files list is empty (line 364)."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    # Create one file so split_corpus_files works
    (corpus / "a.txt").write_text("test content\n", encoding="utf-8")

    cfg: ModelTrainConfig = {
        **_make_cfg(),
        "corpus_path": str(corpus),
        "holdout_fraction": 0.0,
        "test_split_ratio": 0.0,
    }

    trainer = bt.BaseTrainer(
        _make_prepared(),
        cfg,
        _make_settings(),
        run_id="test-make-loader-none",
        redis_hb=lambda _: None,
        cancelled=lambda: False,
        resume=False,
        progress=None,
        service_name="char-lstm-train",
    )

    train_loader, val_loader, test_loader = trainer._build_all_loaders()
    # With holdout_fraction=0 and test_split_ratio=0, val and test loaders should be None
    assert train_loader  # non-empty
    assert val_loader is None, "Val loader should be None when holdout_fraction=0"
    assert test_loader is None, "Test loader should be None when test_split_ratio=0"


def test_build_all_loaders_raises_when_no_train_data(tmp_path: Path) -> None:
    """Test _build_all_loaders raises RuntimeError when no training data (line 379)."""
    from model_trainer.core import _test_hooks
    from model_trainer.core.contracts.dataset import CorpusSplit
    from model_trainer.core.contracts.dataset import DatasetConfig as DS_Cfg

    def _fake_split(cfg: DS_Cfg) -> CorpusSplit:
        return CorpusSplit(train=(), validation=(), test=())

    _test_hooks.split_corpus = _fake_split

    cfg: ModelTrainConfig = {
        **_make_cfg(),
        "corpus_path": str(tmp_path),
    }

    trainer = bt.BaseTrainer(
        _make_prepared(),
        cfg,
        _make_settings(),
        run_id="test-no-train-data",
        redis_hb=lambda _: None,
        cancelled=lambda: False,
        resume=False,
        progress=None,
        service_name="char-lstm-train",
    )

    with pytest.raises(RuntimeError, match="No training data available"):
        _ = trainer._build_all_loaders()


def test_log_wandb_config_called_when_publisher_present() -> None:
    """Test _log_wandb_config logs config when wandb publisher is provided."""
    from platform_ml.wandb_publisher import WandbPublisher

    state, fake_module = _make_fake_wandb_module()
    _setup_fake_wandb_hooks(fake_module)
    wandb_pub = WandbPublisher(project="test", run_name="test", enabled=True)

    trainer = bt.BaseTrainer(
        _make_prepared(),
        _make_cfg(),
        _make_settings(),
        run_id="test-wandb-config",
        redis_hb=lambda _: None,
        cancelled=lambda: False,
        resume=False,
        progress=None,
        service_name="char-lstm-train",
        wandb_publisher=wandb_pub,
    )

    trainer._log_wandb_config()

    assert len(state.config_updates) == 1
    config = state.config_updates[0]
    assert config["run_id"] == "test-wandb-config"
    assert config["model_family"] == "char_lstm"


def test_log_wandb_step_called_when_publisher_present() -> None:
    """Test _log_wandb_step logs step metrics when wandb publisher is provided."""
    from platform_ml.wandb_publisher import WandbPublisher

    state, fake_module = _make_fake_wandb_module()
    _setup_fake_wandb_hooks(fake_module)
    wandb_pub = WandbPublisher(project="test", run_name="test", enabled=True)

    trainer = bt.BaseTrainer(
        _make_prepared(),
        _make_cfg(),
        _make_settings(),
        run_id="test-wandb-step",
        redis_hb=lambda _: None,
        cancelled=lambda: False,
        resume=False,
        progress=None,
        service_name="char-lstm-train",
        wandb_publisher=wandb_pub,
    )

    trainer._log_wandb_step(
        step=10,
        epoch=0,
        train_loss=0.5,
        train_ppl=1.65,
        grad_norm=0.1,
        samples_per_sec=100.0,
    )

    assert len(state.log_calls) == 1
    metrics = state.log_calls[0]
    assert metrics["global_step"] == 10
    assert metrics["train_loss"] == 0.5


def test_log_wandb_epoch_called_when_publisher_present() -> None:
    """Test _log_wandb_epoch logs epoch metrics when wandb publisher is provided."""
    from platform_ml.wandb_publisher import WandbPublisher

    state, fake_module = _make_fake_wandb_module()
    _setup_fake_wandb_hooks(fake_module)
    wandb_pub = WandbPublisher(project="test", run_name="test", enabled=True)

    trainer = bt.BaseTrainer(
        _make_prepared(),
        _make_cfg(),
        _make_settings(),
        run_id="test-wandb-epoch",
        redis_hb=lambda _: None,
        cancelled=lambda: False,
        resume=False,
        progress=None,
        service_name="char-lstm-train",
        wandb_publisher=wandb_pub,
    )

    trainer._log_wandb_epoch(
        epoch=1,
        train_loss=0.3,
        train_ppl=1.35,
        val_loss=0.4,
        val_ppl=1.5,
        best_val_loss=0.35,
        epochs_no_improve=0,
    )

    assert len(state.log_calls) == 1
    metrics = state.log_calls[0]
    assert metrics["epoch"] == 1
    assert metrics["train_loss"] == 0.3
    assert metrics["train_ppl"] == 1.35
    assert metrics["val_loss"] == 0.4


def test_log_wandb_final_called_when_publisher_present() -> None:
    """Test _log_wandb_final logs final metrics when publisher is provided."""
    from platform_ml.wandb_publisher import WandbPublisher

    state, fake_module = _make_fake_wandb_module()
    _setup_fake_wandb_hooks(fake_module)
    wandb_pub = WandbPublisher(project="test", run_name="test", enabled=True)

    trainer = bt.BaseTrainer(
        _make_prepared(),
        _make_cfg(),
        _make_settings(),
        run_id="test-wandb-final",
        redis_hb=lambda _: None,
        cancelled=lambda: False,
        resume=False,
        progress=None,
        service_name="char-lstm-train",
        wandb_publisher=wandb_pub,
    )

    trainer._log_wandb_final(
        test_loss=0.25,
        test_ppl=1.28,
        early_stopped=True,
    )

    assert len(state.log_calls) == 1
    metrics = state.log_calls[0]
    assert metrics["test_loss"] == 0.25
    # early_stopped is converted to int (1) by WandbPublisher.log_final
    assert metrics["early_stopped"] == 1


def test_log_wandb_final_skips_none_values() -> None:
    """Test _log_wandb_final only includes non-None test metrics."""
    from platform_ml.wandb_publisher import WandbPublisher

    state, fake_module = _make_fake_wandb_module()
    _setup_fake_wandb_hooks(fake_module)
    wandb_pub = WandbPublisher(project="test", run_name="test", enabled=True)

    trainer = bt.BaseTrainer(
        _make_prepared(),
        _make_cfg(),
        _make_settings(),
        run_id="test-wandb-final-none",
        redis_hb=lambda _: None,
        cancelled=lambda: False,
        resume=False,
        progress=None,
        service_name="char-lstm-train",
        wandb_publisher=wandb_pub,
    )

    trainer._log_wandb_final(
        test_loss=None,
        test_ppl=None,
        early_stopped=False,
    )

    assert len(state.log_calls) == 1
    metrics = state.log_calls[0]
    assert "test_loss" not in metrics
    assert "test_ppl" not in metrics
    # early_stopped=False converted to 0 by WandbPublisher.log_final
    assert metrics["early_stopped"] == 0


def test_log_wandb_epoch_table_skips_when_no_publisher() -> None:
    """Test _log_wandb_epoch_table does nothing when no wandb publisher."""
    trainer = bt.BaseTrainer(
        _make_prepared(),
        _make_cfg(),
        _make_settings(),
        run_id="test-no-wandb-table",
        redis_hb=lambda _: None,
        cancelled=lambda: False,
        resume=False,
        progress=None,
        service_name="char-lstm-train",
        wandb_publisher=None,
    )

    # Should not raise
    trainer._log_wandb_epoch_table()


def test_log_wandb_epoch_table_skips_when_no_summaries() -> None:
    """Test _log_wandb_epoch_table does nothing when epoch_summaries is empty."""
    from platform_ml.wandb_publisher import WandbPublisher

    state, fake_module = _make_fake_wandb_module()
    _setup_fake_wandb_hooks(fake_module)
    wandb_pub = WandbPublisher(project="test", run_name="test", enabled=True)

    trainer = bt.BaseTrainer(
        _make_prepared(),
        _make_cfg(),
        _make_settings(),
        run_id="test-wandb-table-empty",
        redis_hb=lambda _: None,
        cancelled=lambda: False,
        resume=False,
        progress=None,
        service_name="char-lstm-train",
        wandb_publisher=wandb_pub,
    )

    # Ensure epoch_summaries is empty
    trainer._epoch_summaries = []
    trainer._log_wandb_epoch_table()

    # With empty summaries, no log calls should be made for table
    # log_calls should be empty (no table logged)
    assert len(state.log_calls) == 0


def test_log_wandb_epoch_table_logs_data_when_summaries_exist() -> None:
    """Test _log_wandb_epoch_table logs table when epoch_summaries has data."""
    from platform_ml.wandb_publisher import WandbPublisher

    state, fake_module = _make_fake_wandb_module()
    _setup_fake_wandb_hooks(fake_module)
    wandb_pub = WandbPublisher(project="test", run_name="test", enabled=True)

    trainer = bt.BaseTrainer(
        _make_prepared(),
        _make_cfg(),
        _make_settings(),
        run_id="test-wandb-table-data",
        redis_hb=lambda _: None,
        cancelled=lambda: False,
        resume=False,
        progress=None,
        service_name="char-lstm-train",
        wandb_publisher=wandb_pub,
    )

    # Add epoch summaries: (epoch, train_loss, train_ppl, val_loss, val_ppl)
    trainer._epoch_summaries = [
        (1, 0.5, 1.65, 0.4, 1.49),
        (2, 0.3, 1.35, 0.25, 1.28),
    ]
    trainer._log_wandb_epoch_table()

    # The table is logged via wandb.log with {"epoch_summary": table}
    # Our fake just captures log_calls with the table object
    assert len(state.log_calls) == 1
    log_data = state.log_calls[0]
    assert "epoch_summary" in log_data


def test_finish_wandb_skips_when_no_publisher() -> None:
    """Test _finish_wandb does nothing when no wandb publisher."""
    trainer = bt.BaseTrainer(
        _make_prepared(),
        _make_cfg(),
        _make_settings(),
        run_id="test-no-wandb-finish",
        redis_hb=lambda _: None,
        cancelled=lambda: False,
        resume=False,
        progress=None,
        service_name="char-lstm-train",
        wandb_publisher=None,
    )

    # Should not raise
    trainer._finish_wandb()


def test_finish_wandb_calls_finish_when_publisher_present() -> None:
    """Test _finish_wandb calls finish when wandb publisher is provided."""
    from platform_ml.wandb_publisher import WandbPublisher

    state, fake_module = _make_fake_wandb_module()
    _setup_fake_wandb_hooks(fake_module)
    wandb_pub = WandbPublisher(project="test", run_name="test", enabled=True)

    trainer = bt.BaseTrainer(
        _make_prepared(),
        _make_cfg(),
        _make_settings(),
        run_id="test-wandb-finish",
        redis_hb=lambda _: None,
        cancelled=lambda: False,
        resume=False,
        progress=None,
        service_name="char-lstm-train",
        wandb_publisher=wandb_pub,
    )

    trainer._finish_wandb()

    assert state.finish_called is True
