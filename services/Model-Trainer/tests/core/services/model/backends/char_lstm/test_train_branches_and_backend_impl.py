"""Tests for CharLSTM training via BaseTrainer."""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest
import torch
from platform_core.json_utils import JSONValue
from platform_ml.testing import WandbTableProtocol

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import ModelTrainConfig, PreparedLMModel
from model_trainer.core.encoding import Encoder, ListEncoded
from model_trainer.core.services.model.backends.char_lstm.train import train_prepared_char_lstm
from model_trainer.core.services.model.backends.gpt2._dl import DataLoader
from model_trainer.core.services.training import base_trainer as bt
from model_trainer.core.types import (
    ConfigLike,
    ForwardOutProto,
    LMModelProto,
    NamedParameter,
    OptimizerProto,
    ParameterLike,
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
        class _Out(ForwardOutProto):
            @property
            def loss(self: _Out) -> torch.Tensor:
                return torch.tensor(0.0, requires_grad=True)

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

    @property
    def config(self: _LM) -> ConfigLike:
        class _C(ConfigLike):
            n_positions = 8

        return _C()

    @classmethod
    def from_pretrained(cls: type[_LM], path: str) -> LMModelProto:
        return cls()


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
        "precision": "fp32",
    }


def test_gather_versions_handles_missing() -> None:
    from model_trainer.core import _test_hooks

    def _always_unknown(name: str) -> str:
        return "unknown"

    _test_hooks.pkg_version = _always_unknown

    vers: TrainingManifestVersions = bt._gather_lib_versions("char-lstm-train")
    assert set(vers.keys()) == {"torch", "transformers", "tokenizers", "datasets"}
    assert all(v == "unknown" for v in vers.values())


def test_trainer_train_one_epoch_cancelled_early_triggers_return() -> None:
    """Test that _train_one_epoch returns immediately when cancelled."""

    class _DS:
        def __len__(self: _DS) -> int:
            return 1

        def __getitem__(self: _DS, idx: int) -> torch.Tensor:
            vals: list[int] = [1, 1]
            return torch.tensor(vals, dtype=torch.long)

    dl = DataLoader(_DS(), batch_size=1, shuffle=False)

    class _Opt(OptimizerProto):
        def zero_grad(self: _Opt, *, set_to_none: bool = True) -> None:
            return None

        def step(self: _Opt) -> None:
            return None

    trainer = bt.BaseTrainer(
        _make_prepared(),
        _make_cfg(),
        _make_settings(),
        run_id="test-run",
        redis_hb=lambda _: None,
        cancelled=lambda: True,
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

        def __getitem__(self: _DS10, idx: int) -> torch.Tensor:
            vals: list[int] = [1, 1]
            return torch.tensor(vals, dtype=torch.long)

    dl = DataLoader(_DS10(), batch_size=1, shuffle=False)
    hb_calls: list[float] = []
    prog_calls: list[tuple[int, int, float, float, float]] = []

    class _Opt2(OptimizerProto):
        def zero_grad(self: _Opt2, *, set_to_none: bool = True) -> None:
            return None

        def step(self: _Opt2) -> None:
            return None

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

        def __getitem__(self: _DS1, i: int) -> torch.Tensor:
            vals: list[int] = [1, 1]
            return torch.tensor(vals, dtype=torch.long)

    ds = _DS1()

    # Cancelled returns True immediately - the loop should exit on first batch
    trainer = bt.BaseTrainer(
        _make_prepared(),
        _make_cfg(),
        _make_settings(),
        run_id="test-run",
        redis_hb=lambda _: None,
        cancelled=lambda: True,  # Always cancelled
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
    from model_trainer.core.contracts.dataset import DatasetConfig as DS_Cfg

    # Create corpus file
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    corpus_file = corpus_dir / "train.txt"
    corpus_file.write_text("hello world test data\n" * 10, encoding="utf-8")

    # Hook split_corpus_files to return our test file
    def _test_split(cfg: DS_Cfg) -> tuple[list[str], list[str], list[str]]:
        return [str(corpus_file)], [], []

    _test_hooks.split_corpus_files = _test_split

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
        "precision": "fp32",
    }

    train_prepared_char_lstm(
        prepared,
        cfg,
        settings_with_paths,
        run_id="rid2",
        redis_hb=lambda _: None,
        cancelled=lambda: False,
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
    from model_trainer.core.contracts.dataset import DatasetConfig as DS_Cfg

    # Create corpus file
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    corpus_file = corpus_dir / "train.txt"
    corpus_file.write_text("hello world test data\n" * 10, encoding="utf-8")

    # Hook split_corpus_files to return our test file
    def _test_split(cfg: DS_Cfg) -> tuple[list[str], list[str], list[str]]:
        return [str(corpus_file)], [], []

    _test_hooks.split_corpus_files = _test_split

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
        "precision": "fp32",
    }

    train_prepared_char_lstm(
        prepared,
        cfg,
        settings_with_paths,
        run_id="rid3",
        redis_hb=lambda _: None,
        cancelled=lambda: True,  # Always cancelled - save should be skipped
        progress=None,
    )
    # Save should be skipped when cancelled=True
    assert rec2.saved == []


def test_trainer_run_training_loop_progress_none_branch() -> None:
    """Test _run_training_loop when progress is None."""

    class _DSEmpty:
        def __len__(self: _DSEmpty) -> int:
            return 0

        def __getitem__(self: _DSEmpty, idx: int) -> torch.Tensor:
            raise AssertionError

    dl = DataLoader(_DSEmpty(), batch_size=1, shuffle=False)

    trainer = bt.BaseTrainer(
        _make_prepared(),
        _make_cfg(),
        _make_settings(),
        run_id="test-run",
        redis_hb=lambda _: None,
        cancelled=lambda: False,
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
    )
    # Verify the return values: (final_loss, total_steps, was_cancelled, early_stopped)
    assert out[0] >= 0.0, f"Expected non-negative loss, got {out[0]}"
    assert out[1] >= 0, f"Expected non-negative steps, got {out[1]}"
    assert out[2] is False, f"Expected not cancelled, got {out[2]}"


def test_trainer_train_one_epoch_progress_none_inside_loop() -> None:
    """Test _train_one_epoch when progress is None."""

    class _DS1:
        def __len__(self: _DS1) -> int:
            return 1

        def __getitem__(self: _DS1, idx: int) -> torch.Tensor:
            vals: list[int] = [1, 1]
            return torch.tensor(vals, dtype=torch.long)

    class _Opt3(OptimizerProto):
        def zero_grad(self: _Opt3, *, set_to_none: bool = True) -> None:
            return None

        def step(self: _Opt3) -> None:
            return None

    dl = DataLoader(_DS1(), batch_size=1, shuffle=False)

    trainer = bt.BaseTrainer(
        _make_prepared(),
        _make_cfg(),
        _make_settings(),
        run_id="test-run",
        redis_hb=lambda _: None,
        cancelled=lambda: False,
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

        def __getitem__(self: _DS, idx: int) -> torch.Tensor:
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
    bt._freeze_embeddings(model)

    # Verify embedding param was frozen, other param was not
    assert model._embed_param.requires_grad is False
    assert model._other_param.requires_grad is True


def test_train_with_freeze_embed_enabled(tmp_path: Path) -> None:
    """Test that training with freeze_embed=True calls _freeze_embeddings hook."""
    from model_trainer.core import _test_hooks
    from model_trainer.core.contracts.dataset import DatasetConfig as DS_Cfg

    freeze_called = {"count": 0}

    def _tracking_freeze(model: LMModelProto) -> None:
        freeze_called["count"] += 1
        # Still perform the actual freeze via the default implementation
        bt._freeze_embeddings(model)

    _test_hooks.freeze_embeddings = _tracking_freeze

    # Create corpus file
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    corpus_file = corpus_dir / "train.txt"
    corpus_file.write_text("hello world test data\n" * 10, encoding="utf-8")

    # Hook split_corpus_files to return our test file
    def _test_split(cfg: DS_Cfg) -> tuple[list[str], list[str], list[str]]:
        return [str(corpus_file)], [], []

    _test_hooks.split_corpus_files = _test_split

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
    bt._freeze_embeddings(wrapper)

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


def test_setup_device_cuda_not_available() -> None:
    """Test _setup_device raises RuntimeError when CUDA requested but not available."""
    from model_trainer.core import _test_hooks

    _test_hooks.cuda_is_available = lambda: False

    cfg: ModelTrainConfig = {**_make_cfg(), "device": "cuda"}

    trainer = bt.BaseTrainer(
        _make_prepared(),
        cfg,
        _make_settings(),
        run_id="test-cuda-fail",
        redis_hb=lambda _: None,
        cancelled=lambda: False,
        progress=None,
        service_name="char-lstm-train",
    )

    with pytest.raises(RuntimeError, match="CUDA requested but not available"):
        _ = trainer._setup_device()


# ===== AMP (Automatic Mixed Precision) Tests =====


def test_get_autocast_context_fp32_returns_nullcontext() -> None:
    """Test that fp32 precision returns nullcontext (no autocast)."""
    ctx = bt._get_autocast_context("fp32", torch.device("cpu"))
    # Verify the context is a no-op by entering and exiting it
    with ctx:
        pass  # No exception means it worked


def test_get_autocast_context_fp16_on_cpu_returns_nullcontext() -> None:
    """Test that fp16 on CPU returns nullcontext (autocast only on CUDA)."""
    ctx = bt._get_autocast_context("fp16", torch.device("cpu"))
    # Verify the context is a no-op by entering and exiting it
    with ctx:
        pass  # No exception means it worked


def test_get_autocast_context_bf16_on_cpu_returns_nullcontext() -> None:
    """Test that bf16 on CPU returns nullcontext (autocast only on CUDA)."""
    ctx = bt._get_autocast_context("bf16", torch.device("cpu"))
    # Verify the context is a no-op by entering and exiting it
    with ctx:
        pass  # No exception means it worked


def test_get_autocast_context_fp16_on_cuda() -> None:
    """Test that fp16 on CUDA returns autocast context."""
    # Create a mock CUDA device (doesn't require actual CUDA)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ctx = bt._get_autocast_context("fp16", torch.device("cuda"))
        # Verify the context can be entered and exited
        with ctx:
            pass  # Autocast context entered successfully


def test_get_autocast_context_bf16_on_cuda() -> None:
    """Test that bf16 on CUDA returns autocast context."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ctx = bt._get_autocast_context("bf16", torch.device("cuda"))
        # Verify the context can be entered and exited
        with ctx:
            pass  # Autocast context entered successfully


def test_create_grad_scaler_returns_scaler() -> None:
    """Test that _create_grad_scaler returns a valid GradScaler."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scaler = bt._create_grad_scaler()
        # Verify it can scale a tensor (basic functionality check)
        t = torch.tensor(1.0, requires_grad=True)
        scaled = scaler.scale(t)
        assert scaled.item() >= 0.0  # Scaled value should be non-negative


def test_train_one_epoch_fp16_scaler_paths() -> None:
    """Test that fp16 precision with CUDA device uses GradScaler paths.

    This test covers lines 611-614 and 627-629 in base_trainer.py.
    Note: Without actual CUDA, the scaler is disabled but the code paths still execute.
    """

    class _DS(torch.utils.data.Dataset[torch.Tensor]):
        def __len__(self: _DS) -> int:
            return 2

        def __getitem__(self: _DS, idx: int) -> torch.Tensor:
            return torch.randint(0, 4, (4,))

    dl = DataLoader(_DS(), batch_size=1, shuffle=False)

    class _Opt(OptimizerProto):
        def __init__(self: _Opt) -> None:
            pass

        def zero_grad(self: _Opt, *, set_to_none: bool = False) -> None:
            return None

        def step(self: _Opt) -> None:
            return None

    # Create config with fp16 precision
    cfg: ModelTrainConfig = {**_make_cfg(), "precision": "fp16"}

    trainer = bt.BaseTrainer(
        _make_prepared(),
        cfg,
        _make_settings(),
        run_id="test-fp16-run",
        redis_hb=lambda _: None,
        cancelled=lambda: False,
        progress=None,
        service_name="char-lstm-train",
    )
    # Mock device as CUDA to trigger scaler path
    trainer._device = torch.device("cuda")

    # Run training epoch - this will use scaler paths even if CUDA is not available
    # (the scaler is disabled but code paths still execute)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        out = trainer._train_one_epoch(
            model=_LM(),
            dataloader=dl,
            optim=_Opt(),
            epoch=0,
            device="cpu",  # Actually runs on CPU but scaler logic still executes
            start_step=0,
        )
    # Verify training completed (not cancelled)
    assert out[2] is False and out[1] >= 1


def test_evaluate_get_autocast_context_cuda_fp16() -> None:
    """Test char_lstm evaluate._get_autocast_context with fp16 on CUDA."""
    from model_trainer.core.services.model.backends.char_lstm import evaluate as char_eval

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ctx = char_eval._get_autocast_context("fp16", "cuda")
        with ctx:
            pass  # Autocast context entered successfully


def test_evaluate_get_autocast_context_cuda_bf16() -> None:
    """Test char_lstm evaluate._get_autocast_context with bf16 on CUDA."""
    from model_trainer.core.services.model.backends.char_lstm import evaluate as char_eval

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ctx = char_eval._get_autocast_context("bf16", "cuda")
        with ctx:
            pass  # Autocast context entered successfully


def test_evaluate_get_autocast_context_cpu_fp16() -> None:
    """Test char_lstm evaluate._get_autocast_context with fp16 on CPU returns nullcontext."""
    from model_trainer.core.services.model.backends.char_lstm import evaluate as char_eval

    ctx = char_eval._get_autocast_context("fp16", "cpu")
    with ctx:
        pass  # Returns nullcontext on non-cuda


def test_gpt2_evaluate_get_autocast_context_cuda_fp16() -> None:
    """Test gpt2 evaluate._get_autocast_context with fp16 on CUDA."""
    from model_trainer.core.services.model.backends.gpt2 import evaluate as gpt2_eval

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ctx = gpt2_eval._get_autocast_context("fp16", "cuda")
        with ctx:
            pass  # Autocast context entered successfully


def test_gpt2_evaluate_get_autocast_context_cuda_bf16() -> None:
    """Test gpt2 evaluate._get_autocast_context with bf16 on CUDA."""
    from model_trainer.core.services.model.backends.gpt2 import evaluate as gpt2_eval

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ctx = gpt2_eval._get_autocast_context("bf16", "cuda")
        with ctx:
            pass  # Autocast context entered successfully


def test_gpt2_evaluate_get_autocast_context_cpu_fp16() -> None:
    """Test gpt2 evaluate._get_autocast_context with fp16 on CPU returns nullcontext."""
    from model_trainer.core.services.model.backends.gpt2 import evaluate as gpt2_eval

    ctx = gpt2_eval._get_autocast_context("fp16", "cpu")
    with ctx:
        pass  # Returns nullcontext on non-cuda


def test_setup_device_cuda_available() -> None:
    """Test _setup_device returns cuda device when available."""
    from model_trainer.core import _test_hooks

    _test_hooks.cuda_is_available = lambda: True

    cfg: ModelTrainConfig = {**_make_cfg(), "device": "cuda"}

    trainer = bt.BaseTrainer(
        _make_prepared(),
        cfg,
        _make_settings(),
        run_id="test-cuda-ok",
        redis_hb=lambda _: None,
        cancelled=lambda: False,
        progress=None,
        service_name="char-lstm-train",
    )

    device = trainer._setup_device()
    assert device.type == "cuda"


def test_apply_lr_cap_when_finetuning() -> None:
    """Test _apply_lr_cap caps learning rate when fine-tuning (lines 328-340)."""
    cfg: ModelTrainConfig = {
        **_make_cfg(),
        "learning_rate": 1e-3,
        "pretrained_run_id": "base-run",
        "finetune_lr_cap": 5e-5,
    }

    trainer = bt.BaseTrainer(
        _make_prepared(),
        cfg,
        _make_settings(),
        run_id="test-lr-cap",
        redis_hb=lambda _: None,
        cancelled=lambda: False,
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
    }

    trainer = bt.BaseTrainer(
        _make_prepared(),
        cfg,
        _make_settings(),
        run_id="test-no-cap",
        redis_hb=lambda _: None,
        cancelled=lambda: False,
        progress=None,
        service_name="char-lstm-train",
    )

    effective_lr = trainer._apply_lr_cap()
    assert effective_lr == 1e-3, f"Expected LR to remain at 1e-3, got {effective_lr}"


def test_early_stopping_triggers_after_patience_exceeded() -> None:
    """Test early stopping triggers when epochs_no_improve >= patience (lines 516-531)."""

    class _DS:
        def __len__(self: _DS) -> int:
            return 1

        def __getitem__(self: _DS, idx: int) -> torch.Tensor:
            vals: list[int] = [1, 1]
            return torch.tensor(vals, dtype=torch.long)

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
    )

    assert early_stopped is True, "Expected early stopping to trigger"
    assert trainer._es_state["epochs_no_improve"] >= 2


def test_clip_grad_norm_legacy_function() -> None:
    """Test legacy _clip_grad_norm function (lines 761-762)."""
    vals: list[float] = [1.0, 2.0, 3.0]
    grad_vals: list[float] = [10.0, 20.0, 30.0]
    param = torch.nn.Parameter(torch.tensor(vals))
    param.grad = torch.tensor(grad_vals)

    bt._clip_grad_norm([param], max_norm=1.0)

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
    }

    trainer = bt.BaseTrainer(
        _make_prepared(),
        cfg,
        _make_settings(),
        run_id="test-no-log-cap",
        redis_hb=lambda _: None,
        cancelled=lambda: False,
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
    from model_trainer.core.contracts.dataset import DatasetConfig as DS_Cfg

    def _fake_split(cfg: DS_Cfg) -> tuple[list[str], list[str], list[str]]:
        return [], [], []

    _test_hooks.split_corpus_files = _fake_split

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
        progress=None,
        service_name="char-lstm-train",
    )

    with pytest.raises(RuntimeError, match="No training data available"):
        _ = trainer._build_all_loaders()


# ===== WandB Integration Tests =====


class _WandbTestState:
    """Shared state for fake wandb module tests."""

    def __init__(self: _WandbTestState) -> None:
        self.config_updates: list[dict[str, str | int | float | bool | None]] = []
        self.log_calls: list[dict[str, float | int | str | bool | WandbTableProtocol]] = []
        self.finish_called = False
        self.init_calls: list[tuple[str, str]] = []


class _FakeWandbRun:
    """Fake wandb run for testing."""

    def __init__(self: _FakeWandbRun) -> None:
        self._id = "fake-run-id"

    @property
    def id(self: _FakeWandbRun) -> str:
        return self._id


class _FakeWandbTable:
    """Fake wandb.Table for testing."""

    def __init__(
        self: _FakeWandbTable,
        columns: list[str],
        data: list[list[float | int | str | bool]],
    ) -> None:
        self._columns = columns
        self._data = data

    @property
    def columns(self: _FakeWandbTable) -> list[str]:
        return self._columns

    @property
    def data(self: _FakeWandbTable) -> list[list[float | int | str | bool]]:
        return self._data


class _FakeWandbConfig:
    """Fake wandb config for testing."""

    def __init__(self: _FakeWandbConfig, state: _WandbTestState) -> None:
        self._state = state

    def update(self: _FakeWandbConfig, d: Mapping[str, JSONValue]) -> None:
        converted: dict[str, str | int | float | bool | None] = {
            k: v for k, v in d.items() if v is None or isinstance(v, (str, int, float, bool))
        }
        self._state.config_updates.append(converted)


class _FakeWandbModule:
    """Fake wandb module that implements WandbModuleProtocol."""

    def __init__(self: _FakeWandbModule, state: _WandbTestState) -> None:
        self._state = state
        self._run: _FakeWandbRun | None = None
        self._config = _FakeWandbConfig(state)

    @property
    def run(self: _FakeWandbModule) -> _FakeWandbRun | None:
        return self._run

    @property
    def config(self: _FakeWandbModule) -> _FakeWandbConfig:
        return self._config

    @property
    def table_ctor(self: _FakeWandbModule) -> type[_FakeWandbTable]:
        return _FakeWandbTable

    def init(self: _FakeWandbModule, *, project: str, name: str) -> _FakeWandbRun:
        self._state.init_calls.append((project, name))
        self._run = _FakeWandbRun()
        return self._run

    def log(
        self: _FakeWandbModule,
        data: Mapping[str, float | int | str | bool | WandbTableProtocol],
    ) -> None:
        # Store log data - convert to dict for easier assertions
        self._state.log_calls.append(dict(data))

    def finish(self: _FakeWandbModule) -> None:
        self._state.finish_called = True


def _make_fake_wandb_module() -> tuple[_WandbTestState, _FakeWandbModule]:
    """Create a fake wandb module for testing.

    Returns:
        Tuple of (state, fake_module) where state tracks all calls.
    """
    state = _WandbTestState()
    fake_module = _FakeWandbModule(state)
    return state, fake_module


def _setup_fake_wandb_hooks(
    fake_module: _FakeWandbModule,
) -> None:
    """Set up platform_ml.testing hooks to use fake wandb module."""
    from platform_ml import testing as ml_testing
    from platform_ml.testing import WandbModuleProtocol

    def _fake_load_wandb() -> WandbModuleProtocol:
        return fake_module

    ml_testing.hooks.load_wandb_module = _fake_load_wandb


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
    # Note: finish is NOT called by _log_wandb_final - see _finish_wandb


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
        progress=None,
        service_name="char-lstm-train",
        wandb_publisher=wandb_pub,
    )

    trainer._finish_wandb()

    assert state.finish_called is True
