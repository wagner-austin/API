"""char_lstm trainer branches: setup and loop entry."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import ModelTrainConfig, PreparedLMModel
from model_trainer.core.encoding import Encoder, ListEncoded
from model_trainer.core.services.model.backends.char_lstm.train import train_prepared_char_lstm
from model_trainer.core.services.training import base_trainer as bt
from model_trainer.core.services.training import base_trainer_core as bt_core
from model_trainer.core.services.training.dataloader import DataLoader
from model_trainer.core.types import (
    ConfigLike,
    ForwardOutProto,
    LMModelProto,
    LoadStateDictResultProto,
    NamedParameter,
    OptimizerProto,
    ParameterLike,
    TorchStateValue,
)
from model_trainer.infra.persistence.models import TrainingManifestVersions


class _LM(LMModelProto):
    def __init__(self: _LM) -> None:
        self._p = torch.nn.Parameter(torch.zeros(1))

    def train(self: _LM) -> None:
        return None

    def eval(self: _LM) -> None:
        return None

    def forward(self: _LM, *, input_ids: torch.Tensor, labels: torch.Tensor) -> ForwardOutProto:
        # Capture parameter reference for use in nested class
        param = self._p

        class _Out(ForwardOutProto):
            @property
            def loss(self: _Out) -> torch.Tensor:
                # Loss must depend on model parameter for gradients to flow
                return (param * 0.0).sum() + 0.1

        return _Out()

    def forward_logits(self: _LM, *, input_ids: torch.Tensor) -> torch.Tensor:
        """Return dummy logits for inference."""
        batch_size = int(input_ids.size(0))
        seq_len = int(input_ids.size(1))
        vocab_size = 4
        return torch.zeros(batch_size, seq_len, vocab_size)

    def parameters(self: _LM) -> Sequence[ParameterLike]:
        return [self._p]

    def named_parameters(self: _LM) -> Sequence[tuple[str, NamedParameter]]:
        return []

    def to(self: _LM, device: str) -> LMModelProto:
        return self

    def save_pretrained(self: _LM, out_dir: str) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

    def gradient_checkpointing_enable(self: _LM) -> None:
        return None

    @property
    def config(self: _LM) -> ConfigLike:
        class _C(ConfigLike):
            n_positions = 8

        return _C()

    @classmethod
    def from_pretrained(cls: type[_LM], path: str) -> LMModelProto:
        return cls()

    def state_dict(self: _LM) -> dict[str, torch.Tensor]:
        return {}

    def load_state_dict(self: _LM, state_dict: dict[str, torch.Tensor]) -> LoadStateDictResultProto:
        _ = state_dict
        return self


class _MiniEnc(Encoder):
    def encode(self: _MiniEnc, text: str) -> ListEncoded:
        return ListEncoded([1, 2])

    def token_to_id(self: _MiniEnc, token: str) -> int | None:
        return 0

    def get_vocab_size(self: _MiniEnc) -> int:
        return 4

    def decode(self: _MiniEnc, ids: list[int]) -> str:
        return "".join(str(i) for i in ids)


def _make_prepared() -> PreparedLMModel:
    return PreparedLMModel(
        model=_LM(),
        tokenizer_id="tok",
        eos_id=1,
        pad_id=0,
        max_seq_len=8,
        tok_for_dataset=_MiniEnc(),
    )


def _make_cfg() -> ModelTrainConfig:
    return {
        "model_family": "char_lstm",
        "model_size": "tiny",
        "max_seq_len": 8,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "tokenizer_id": "tok",
        "corpus_path": "",
        "holdout_fraction": 0.01,
        "seed": 42,
        "pretrained_run_id": None,
        "freeze_embed": False,
        "gradient_clipping": 1.0,
        "optimizer": "adamw",
        "device": "cpu",
        "data_num_workers": 0,
        "data_pin_memory": False,
        "early_stopping_patience": 0,
        "test_split_ratio": 0.0,
        "finetune_lr_cap": 0.0,
        "loss_mask_prefix_separator": None,
        "precision": "fp32",
        "finetuning_strategy": "full",
        "hub_model_id": None,
        "lora": None,
        "quantization": None,
        "gguf_export": None,
    }


def test_gather_versions_handles_missing() -> None:
    from model_trainer.core import _test_hooks

    def _always_unknown(name: str) -> str:
        return "unknown"

    _test_hooks.pkg_version = _always_unknown

    vers: TrainingManifestVersions = bt_core._gather_lib_versions("char-lstm-train")
    assert set(vers.keys()) == {"torch", "transformers", "tokenizers", "datasets"}
    assert all(v == "unknown" for v in vers.values())


class _EvalDS:
    """Dataset of a fixed number of identical two-token rows."""

    def __init__(self: _EvalDS, rows: int) -> None:
        self._rows = rows

    def __len__(self: _EvalDS) -> int:
        return self._rows

    def __getitem__(self: _EvalDS, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        vals: list[int] = [1, 1]
        ids = torch.tensor(vals, dtype=torch.long)
        return (ids, ids)


def _eval_trainer(*, cancelled: bool) -> bt.BaseTrainer:
    """Build a trainer whose evaluation runs on CPU with a fixed cancel answer.

    Args:
        cancelled: What the cancellation callback reports on every call.

    Returns:
        A trainer ready for ``_run_evaluation``.
    """
    trainer = bt.BaseTrainer(
        _make_prepared(),
        _make_cfg(),
        _make_settings(),
        run_id="test-eval",
        redis_hb=lambda _: None,
        cancelled=lambda: cancelled,
        resume=False,
        progress=None,
        service_name="char-lstm-train",
    )
    trainer._device = torch.device("cpu")
    return trainer


def test_run_evaluation_returns_partial_metrics_when_cancelled() -> None:
    """Cancelling mid-evaluation returns what was measured, not a whole pass.

    This path used to be reached only because a single-file corpus was split
    into itself three times, which handed every such run a validation loader.
    Now that the split is disjoint, the path is exercised directly.

    Cancellation is checked before the first batch is scored, so no batch
    contributes and the reported loss is the zero-batch average rather than the
    model's 0.1 per-batch loss.
    """
    loader = DataLoader(_EvalDS(4), batch_size=1, shuffle=False)

    metrics = _eval_trainer(cancelled=True)._run_evaluation(loader, eval_type="validation")

    assert metrics["val_loss"] == 0.0
    assert metrics["val_ppl"] == 1.0


def test_run_evaluation_logs_progress_on_an_interval_not_every_batch() -> None:
    """Twenty batches give ``log_interval = 2``, so half the batches skip logging.

    That exercises the progress check in both directions; with fewer than ten
    batches the interval collapses to 1 and the skip branch never runs.
    """
    loader = DataLoader(_EvalDS(20), batch_size=1, shuffle=False)

    metrics = _eval_trainer(cancelled=False)._run_evaluation(loader, eval_type="test")

    # _LM returns a constant 0.1 loss per batch, so the average is that loss.
    assert abs(metrics["val_loss"] - 0.1) < 1e-6


def test_trainer_train_one_epoch_cancelled_early_triggers_return() -> None:
    """Test that _train_one_epoch returns immediately when cancelled."""

    class _DS:
        def __len__(self: _DS) -> int:
            return 1

        def __getitem__(self: _DS, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            vals: list[int] = [1, 1]
            ids = torch.tensor(vals, dtype=torch.long)
            return (ids, ids)

    dl = DataLoader(_DS(), batch_size=1, shuffle=False)

    class _Opt(OptimizerProto):
        def zero_grad(self: _Opt, *, set_to_none: bool = True) -> None:
            return None

        def step(self: _Opt) -> None:
            return None

        def state_dict(self: _Opt) -> dict[str, TorchStateValue]:
            return {}

        def load_state_dict(self: _Opt, state_dict: dict[str, TorchStateValue]) -> None:
            _ = state_dict

    trainer = bt.BaseTrainer(
        _make_prepared(),
        _make_cfg(),
        _make_settings(),
        run_id="test-run",
        redis_hb=lambda _: None,
        cancelled=lambda: True,
        resume=False,
        progress=None,
        service_name="char-lstm-train",
    )
    trainer._device = torch.device("cpu")

    out = trainer._train_one_epoch(
        model=_LM(),
        dataloader=dl,
        optim=_Opt(),
        epoch=0,
        device="cpu",
        start_step=0,
    )
    assert out[2] is True


def test_trainer_train_one_epoch_progress_and_heartbeat() -> None:
    """Test that _train_one_epoch calls progress and heartbeat."""

    class _DS10:
        def __len__(self: _DS10) -> int:
            return 10

        def __getitem__(self: _DS10, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            vals: list[int] = [1, 1]
            ids = torch.tensor(vals, dtype=torch.long)
            return (ids, ids)

    dl = DataLoader(_DS10(), batch_size=1, shuffle=False)
    hb_calls: list[float] = []
    prog_calls: list[tuple[int, int, float, float, float]] = []

    class _Opt2(OptimizerProto):
        def zero_grad(self: _Opt2, *, set_to_none: bool = True) -> None:
            return None

        def step(self: _Opt2) -> None:
            return None

        def state_dict(self: _Opt2) -> dict[str, TorchStateValue]:
            return {}

        def load_state_dict(self: _Opt2, state_dict: dict[str, TorchStateValue]) -> None:
            _ = state_dict

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
        redis_hb=lambda t: hb_calls.append(t),
        cancelled=lambda: False,
        resume=False,
        progress=_progress_cb,
        service_name="char-lstm-train",
    )
    trainer._device = torch.device("cpu")

    out = trainer._train_one_epoch(
        model=_LM(),
        dataloader=dl,
        optim=_Opt2(),
        epoch=0,
        device="cpu",
        start_step=0,
    )
    assert hb_calls and prog_calls and out[2] is False


def test_trainer_run_training_loop_breaks_on_cancelled() -> None:
    """Test that _run_training_loop breaks when cancelled callback returns True."""

    class _DS1:
        def __len__(self: _DS1) -> int:
            return 10  # More items to ensure loop would continue without cancel

        def __getitem__(self: _DS1, i: int) -> tuple[torch.Tensor, torch.Tensor]:
            vals: list[int] = [1, 1]
            ids = torch.tensor(vals, dtype=torch.long)
            return (ids, ids)

    ds = _DS1()

    # Cancelled returns True immediately - the loop should exit on first batch
    trainer = bt.BaseTrainer(
        _make_prepared(),
        _make_cfg(),
        _make_settings(),
        run_id="test-run",
        redis_hb=lambda _: None,
        cancelled=lambda: True,
        resume=False,  # Always cancelled
        progress=None,
        service_name="char-lstm-train",
    )

    trainer._device = torch.device("cpu")
    trainer._es_state = {"best_val_loss": float("inf"), "epochs_no_improve": 0}
    trainer._val_loader = None

    out = trainer._run_training_loop(
        model=_LM(),
        dataloader=DataLoader(ds, batch_size=1, shuffle=False),
        effective_lr=1e-3,
        start_epoch=0,
        start_step=0,
        initial_last_loss=0.0,
        restored=None,
    )
    # out is (loss, steps, cancelled, early_stopped)
    assert out[2] is True


def _make_settings() -> Settings:
    """Create minimal test settings."""
    from model_trainer.core.config.settings import load_settings

    return load_settings()


def test_train_prepared_calls_save_when_not_cancelled(
    tmp_path: Path, settings_with_paths: Settings
) -> None:
    from model_trainer.core import _test_hooks
    from model_trainer.core.contracts.dataset import CorpusSplit
    from model_trainer.core.contracts.dataset import DatasetConfig as DS_Cfg
    from model_trainer.core.services.training.dataset_builder import read_corpus_lines

    # Create corpus file
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    corpus_file = corpus_dir / "train.txt"
    corpus_file.write_text("hello world test data\n" * 10, encoding="utf-8")

    # Hook split_corpus to train on our test corpus with no holdout
    def _test_split(cfg: DS_Cfg) -> CorpusSplit:
        return CorpusSplit(train=read_corpus_lines([str(corpus_file)]), validation=(), test=())

    _test_hooks.split_corpus = _test_split

    class _RecorderLM(_LM):
        def __init__(self: _RecorderLM) -> None:
            super().__init__()
            self.saved: list[str] = []

        def save_pretrained(self: _RecorderLM, out_dir: str) -> None:
            self.saved.append(out_dir)
            Path(out_dir).mkdir(parents=True, exist_ok=True)

    rec = _RecorderLM()
    prepared = PreparedLMModel(
        model=rec,
        tokenizer_id="tok",
        eos_id=1,
        pad_id=0,
        max_seq_len=8,
        tok_for_dataset=_MiniEnc(),
    )

    cfg: ModelTrainConfig = {
        "model_family": "char_lstm",
        "model_size": "tiny",
        "max_seq_len": 8,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "tokenizer_id": "tok",
        "corpus_path": str(corpus_dir),
        "holdout_fraction": 0.0,
        "seed": 42,
        "pretrained_run_id": None,
        "freeze_embed": False,
        "gradient_clipping": 1.0,
        "optimizer": "adamw",
        "device": "cpu",
        "data_num_workers": 0,
        "data_pin_memory": False,
        "early_stopping_patience": 0,
        "test_split_ratio": 0.0,
        "finetune_lr_cap": 0.0,
        "loss_mask_prefix_separator": None,
        "precision": "fp32",
        "finetuning_strategy": "full",
        "hub_model_id": None,
        "lora": None,
        "quantization": None,
        "gguf_export": None,
    }

    train_prepared_char_lstm(
        prepared,
        cfg,
        settings_with_paths,
        run_id="rid2",
        redis_hb=lambda _: None,
        cancelled=lambda: False,
        resume=False,
        progress=None,
    )
    expected_dir = Path(settings_with_paths["app"]["artifacts_root"]) / "models" / "rid2"
    assert expected_dir.exists()
    assert rec.saved == [str(expected_dir)], (
        f"Expected model to be saved to {expected_dir}, got {rec.saved}"
    )


def test_train_prepared_skips_save_when_cancelled(
    tmp_path: Path, settings_with_paths: Settings
) -> None:
    from model_trainer.core import _test_hooks
    from model_trainer.core.contracts.dataset import CorpusSplit
    from model_trainer.core.contracts.dataset import DatasetConfig as DS_Cfg
    from model_trainer.core.services.training.dataset_builder import read_corpus_lines

    # Create corpus file
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    corpus_file = corpus_dir / "train.txt"
    corpus_file.write_text("hello world test data\n" * 10, encoding="utf-8")

    # Hook split_corpus to train on our test corpus with no holdout
    def _test_split(cfg: DS_Cfg) -> CorpusSplit:
        return CorpusSplit(train=read_corpus_lines([str(corpus_file)]), validation=(), test=())

    _test_hooks.split_corpus = _test_split

    class _RecorderLM(_LM):
        def __init__(self: _RecorderLM) -> None:
            super().__init__()
            self.saved: list[str] = []

        def save_pretrained(self: _RecorderLM, out_dir: str) -> None:
            self.saved.append(out_dir)
            Path(out_dir).mkdir(parents=True, exist_ok=True)

    rec2 = _RecorderLM()
    prepared = PreparedLMModel(
        model=rec2,
        tokenizer_id="tok",
        eos_id=1,
        pad_id=0,
        max_seq_len=8,
        tok_for_dataset=_MiniEnc(),
    )

    cfg: ModelTrainConfig = {
        "model_family": "char_lstm",
        "model_size": "tiny",
        "max_seq_len": 8,
        "num_epochs": 1,
        "batch_size": 1,
        "learning_rate": 1e-3,
        "tokenizer_id": "tok",
        "corpus_path": str(corpus_dir),
        "holdout_fraction": 0.0,
        "seed": 42,
        "pretrained_run_id": None,
        "freeze_embed": False,
        "gradient_clipping": 1.0,
        "optimizer": "adamw",
        "device": "cpu",
        "data_num_workers": 0,
        "data_pin_memory": False,
        "early_stopping_patience": 0,
        "test_split_ratio": 0.0,
        "finetune_lr_cap": 0.0,
        "loss_mask_prefix_separator": None,
        "precision": "fp32",
        "finetuning_strategy": "full",
        "hub_model_id": None,
        "lora": None,
        "quantization": None,
        "gguf_export": None,
    }

    train_prepared_char_lstm(
        prepared,
        cfg,
        settings_with_paths,
        run_id="rid3",
        redis_hb=lambda _: None,
        cancelled=lambda: True,
        resume=False,  # Always cancelled - save should be skipped
        progress=None,
    )
    # Save should be skipped when cancelled=True
    assert rec2.saved == []


def test_trainer_run_training_loop_progress_none_branch() -> None:
    """Test _run_training_loop when progress is None."""

    class _DSEmpty:
        def __len__(self: _DSEmpty) -> int:
            return 0

        def __getitem__(self: _DSEmpty, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            raise AssertionError

    dl = DataLoader(_DSEmpty(), batch_size=1, shuffle=False)

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
    trainer._es_state = {"best_val_loss": float("inf"), "epochs_no_improve": 0}
    trainer._val_loader = None

    out = trainer._run_training_loop(
        model=_LM(),
        dataloader=dl,
        effective_lr=1e-3,
        start_epoch=0,
        start_step=0,
        initial_last_loss=0.0,
        restored=None,
    )
    # Verify the return values: (final_loss, total_steps, was_cancelled, early_stopped)
    assert out[0] >= 0.0, f"Expected non-negative loss, got {out[0]}"
    assert out[1] >= 0, f"Expected non-negative steps, got {out[1]}"
    assert out[2] is False, f"Expected not cancelled, got {out[2]}"
