"""char_lstm trainer branches: mid-loop behavior."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from tests.core.services.model.backends.char_lstm._train_branches_support import (
    _LM,
    _make_cfg,
    _make_prepared,
    _make_settings,
)

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import ModelTrainConfig
from model_trainer.core.services.training import base_trainer as bt
from model_trainer.core.services.training import trainer_grad_utils as bt_grad
from model_trainer.core.services.training.dataloader import DataLoader
from model_trainer.core.types import (
    LMModelProto,
    NamedParameter,
    OptimizerProto,
    TorchStateValue,
)


def test_trainer_train_one_epoch_progress_none_inside_loop() -> None:
    """Test _train_one_epoch when progress is None."""

    class _DS1:
        def __len__(self: _DS1) -> int:
            return 1

        def __getitem__(self: _DS1, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            vals: list[int] = [1, 1]
            ids = torch.tensor(vals, dtype=torch.long)
            return (ids, ids)

    class _Opt3(OptimizerProto):
        def zero_grad(self: _Opt3, *, set_to_none: bool = True) -> None:
            return None

        def step(self: _Opt3) -> None:
            return None

        def state_dict(self: _Opt3) -> dict[str, TorchStateValue]:
            return {}

        def load_state_dict(self: _Opt3, state_dict: dict[str, TorchStateValue]) -> None:
            _ = state_dict

    dl = DataLoader(_DS1(), batch_size=1, shuffle=False)

    trainer = bt.BaseTrainer(
        _make_prepared(),
        _make_cfg(),
        _make_settings(),
        run_id="test-run",
        redis_hb=lambda _: None,
        cancelled=lambda: False,
        resume=False,
        progress=None,
        service_name="char-lstm-train",
    )
    trainer._device = torch.device("cpu")

    out = trainer._train_one_epoch(
        model=_LM(),
        dataloader=dl,
        optim=_Opt3(),
        epoch=0,
        device="cpu",
        start_step=0,
    )
    assert out[2] is False and out[1] >= 1


def test_run_training_loop_progress_called_when_no_batches() -> None:
    """Test that progress is called even when no batches (for empty epoch)."""

    # DataLoader that yields zero batches to keep steps unchanged
    class _DS:
        def __len__(self: _DS) -> int:
            return 0

        def __getitem__(self: _DS, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            raise AssertionError("should not be called")

    prog_calls: list[tuple[int, int, float, float, float]] = []

    def _progress_cb(
        step: int,
        epoch: int,
        loss: float,
        train_ppl: float,
        grad_norm: float,
        samples_per_sec: float,
        val_loss: float | None,
        val_ppl: float | None,
    ) -> None:
        prog_calls.append((step, epoch, loss, grad_norm, samples_per_sec))

    trainer = bt.BaseTrainer(
        _make_prepared(),
        _make_cfg(),
        _make_settings(),
        run_id="test-run",
        redis_hb=lambda _: None,
        cancelled=lambda: False,
        resume=False,
        progress=_progress_cb,
        service_name="char-lstm-train",
    )

    trainer._device = torch.device("cpu")
    trainer._es_state = {"best_val_loss": float("inf"), "epochs_no_improve": 0}
    trainer._val_loader = None

    # Create empty dataloader directly - no need to patch _build_all_loaders
    empty_loader = DataLoader(_DS(), batch_size=1, shuffle=False)
    out = trainer._run_training_loop(
        model=_LM(),
        dataloader=empty_loader,
        effective_lr=1e-3,
        start_epoch=0,
        start_step=0,
        initial_last_loss=0.0,
        restored=None,
    )
    # Ensure branch executed: progress called even if no steps advanced
    # out is (loss, steps, cancelled, early_stopped)
    assert isinstance(out, tuple) and len(out) == 4 and len(prog_calls) >= 1


def test_freeze_embeddings_when_enabled() -> None:
    """Test that freeze_embed=True triggers _freeze_embeddings and freezes embedding params."""

    class _EmbedParam(NamedParameter):
        """Fake embedding parameter that tracks if requires_grad was set."""

        def __init__(self: _EmbedParam) -> None:
            self._requires_grad = True
            self._tensor = torch.zeros(1)

        @property
        def requires_grad(self: _EmbedParam) -> bool:
            return self._requires_grad

        @requires_grad.setter
        def requires_grad(self: _EmbedParam, value: bool) -> None:
            self._requires_grad = value

        @property
        def grad(self: _EmbedParam) -> torch.Tensor | None:
            return None

        def detach(self: _EmbedParam) -> torch.Tensor:
            return self._tensor.detach()

        def clone(self: _EmbedParam) -> torch.Tensor:
            return self._tensor.clone()

    class _LMWithEmbeddings(_LM):
        """Model with embedding-like named parameters."""

        def __init__(self: _LMWithEmbeddings) -> None:
            super().__init__()
            self._embed_param = _EmbedParam()
            self._other_param = _EmbedParam()

        def named_parameters(
            self: _LMWithEmbeddings,
        ) -> Sequence[tuple[str, NamedParameter]]:
            # Return params with embedding-like names that should be frozen
            return [
                ("transformer.wte.weight", self._embed_param),
                ("linear.weight", self._other_param),
            ]

    model = _LMWithEmbeddings()
    # Call the internal function directly to test freezing logic
    bt_grad._freeze_embeddings(model)

    # Verify embedding param was frozen, other param was not
    assert model._embed_param.requires_grad is False
    assert model._other_param.requires_grad is True


def test_train_with_freeze_embed_enabled(tmp_path: Path) -> None:
    """Test that training with freeze_embed=True calls _freeze_embeddings hook."""
    from model_trainer.core import _test_hooks
    from model_trainer.core.contracts.dataset import CorpusSplit
    from model_trainer.core.contracts.dataset import DatasetConfig as DS_Cfg
    from model_trainer.core.services.training.dataset_builder import read_corpus_lines

    freeze_called = {"count": 0}

    def _tracking_freeze(model: LMModelProto) -> None:
        freeze_called["count"] += 1
        # Still perform the actual freeze via the default implementation
        bt_grad._freeze_embeddings(model)

    _test_hooks.freeze_embeddings = _tracking_freeze

    # Create corpus file
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    corpus_file = corpus_dir / "train.txt"
    corpus_file.write_text("hello world test data\n" * 10, encoding="utf-8")

    # Hook split_corpus to train on our test corpus with no holdout
    def _test_split(cfg: DS_Cfg) -> CorpusSplit:
        return CorpusSplit(train=read_corpus_lines([str(corpus_file)]), validation=(), test=())

    _test_hooks.split_corpus = _test_split

    # Hook model_dir to use tmp_path
    def _test_model_dir(settings: Settings, run_id: str) -> Path:
        return tmp_path / "models" / run_id

    _test_hooks.model_dir = _test_model_dir

    # Create config with freeze_embed=True
    cfg: ModelTrainConfig = {
        **_make_cfg(),
        "freeze_embed": True,
        "corpus_path": str(corpus_dir),
    }

    train_losses: list[float] = []

    def track_loss(
        step: int,
        epoch: int,
        loss: float,
        train_ppl: float,
        grad_norm: float,
        samples_per_sec: float,
        val_loss: float | None,
        val_ppl: float | None,
    ) -> None:
        train_losses.append(loss)

    trainer = bt.BaseTrainer(
        _make_prepared(),
        cfg,
        _make_settings(),
        run_id="test-run-freeze",
        redis_hb=lambda _: None,
        cancelled=lambda: False,
        resume=False,
        progress=track_loss,
        service_name="char-lstm-train",
    )

    _ = trainer.train()

    # Verify _freeze_embeddings hook was called
    assert freeze_called["count"] == 1

    # Verify training ran and produced valid losses
    assert train_losses, "Expected at least one loss record from training"
    # Verify losses are valid float values (not NaN or infinite)
    for loss in train_losses:
        assert loss >= 0.0, f"Loss should be non-negative, got {loss}"
        assert loss < 1e10, f"Loss should be finite, got {loss}"
    # Verify loss decreased or stayed stable (training made progress or converged)
    if len(train_losses) >= 2:
        initial_loss = train_losses[0]
        final_loss = train_losses[-1]
        assert final_loss <= initial_loss, (
            f"Expected final loss ({final_loss:.4f}) <= initial loss ({initial_loss:.4f})"
        )


def test_freeze_embeddings_on_real_char_lstm() -> None:
    """Integration test: verify _freeze_embeddings actually freezes real CharLSTM embedding."""
    from model_trainer.core.services.model.backends.char_lstm.model import (
        CharLSTM,
        CharLSTMModel,
    )

    # Create a real CharLSTM model
    model = CharLSTM(
        vocab_size=10,
        embed_dim=8,
        hidden_dim=16,
        num_layers=2,
        dropout=0.1,
        max_seq_len=32,
    )
    wrapper = CharLSTMModel(model)

    # Verify embedding params start with requires_grad=True
    embedding_params_before = [
        (name, p.requires_grad)
        for name, p in wrapper.named_parameters()
        if "embedding" in name.lower()
    ]
    assert len(embedding_params_before) == 1, "Expected exactly 1 embedding param"
    # Verify the embedding param is unfrozen by checking the specific value
    embed_name, embed_requires_grad = embedding_params_before[0]
    assert embed_requires_grad is True, (
        f"Expected embedding param '{embed_name}' to start unfrozen "
        f"(requires_grad=True), got {embed_requires_grad}"
    )

    # Apply freeze
    bt_grad._freeze_embeddings(wrapper)

    # Verify embedding params now have requires_grad=False
    embedding_params_after = [
        (name, p.requires_grad)
        for name, p in wrapper.named_parameters()
        if "embedding" in name.lower()
    ]
    # Check the specific frozen state of the embedding param
    embed_name_after, embed_requires_grad_after = embedding_params_after[0]
    assert embed_requires_grad_after is False, (
        f"Expected embedding param '{embed_name_after}' to be frozen "
        f"(requires_grad=False), got {embed_requires_grad_after}"
    )

    # Verify non-embedding params still have requires_grad=True
    other_params = [
        (name, p.requires_grad)
        for name, p in wrapper.named_parameters()
        if "embedding" not in name.lower()
    ]
    # num_layers=2: 8 LSTM params (4 per layer) + 2 projection params = 10
    assert len(other_params) == 10, (
        f"Expected 10 non-embedding params (2-layer LSTM + projection), got {len(other_params)}"
    )
    # Check each non-embedding param individually to ensure they remain unfrozen
    for param_name, param_requires_grad in other_params:
        assert param_requires_grad is True, (
            f"Expected non-embedding param '{param_name}' to remain unfrozen "
            f"(requires_grad=True), got {param_requires_grad}"
        )


def test_apply_lr_cap_when_finetuning() -> None:
    """Test _apply_lr_cap caps learning rate when fine-tuning (lines 328-340)."""
    cfg: ModelTrainConfig = {
        **_make_cfg(),
        "learning_rate": 1e-3,
        "pretrained_run_id": "base-run",
        "finetune_lr_cap": 5e-5,
        "loss_mask_prefix_separator": None,
    }

    trainer = bt.BaseTrainer(
        _make_prepared(),
        cfg,
        _make_settings(),
        run_id="test-lr-cap",
        redis_hb=lambda _: None,
        cancelled=lambda: False,
        resume=False,
        progress=None,
        service_name="char-lstm-train",
    )

    effective_lr = trainer._apply_lr_cap()
    assert effective_lr == 5e-5, f"Expected LR to be capped at 5e-5, got {effective_lr}"


def test_apply_lr_cap_no_cap_when_not_finetuning() -> None:
    """Test _apply_lr_cap does not cap LR when not fine-tuning."""
    cfg: ModelTrainConfig = {
        **_make_cfg(),
        "learning_rate": 1e-3,
        "pretrained_run_id": None,
        "finetune_lr_cap": 5e-5,
        "loss_mask_prefix_separator": None,
    }

    trainer = bt.BaseTrainer(
        _make_prepared(),
        cfg,
        _make_settings(),
        run_id="test-no-cap",
        redis_hb=lambda _: None,
        cancelled=lambda: False,
        resume=False,
        progress=None,
        service_name="char-lstm-train",
    )

    effective_lr = trainer._apply_lr_cap()
    assert effective_lr == 1e-3, f"Expected LR to remain at 1e-3, got {effective_lr}"
