"""char_lstm trainer branches: mid-loop behavior."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from pathlib import Path

import pytest
import torch

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import ModelTrainConfig, PreparedLMModel
from model_trainer.core.encoding import Encoder, ListEncoded
from model_trainer.core.services.training import base_trainer as bt
from model_trainer.core.services.training import trainer_grad_utils as bt_grad
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


class _MiniEnc(Encoder):
    def encode(self: _MiniEnc, text: str) -> ListEncoded:
        return ListEncoded([1, 2])

    def token_to_id(self: _MiniEnc, token: str) -> int | None:
        return 0

    def get_vocab_size(self: _MiniEnc) -> int:
        return 4

    def decode(self: _MiniEnc, ids: list[int]) -> str:
        return "".join(str(i) for i in ids)


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


def _make_prepared() -> PreparedLMModel:
    return PreparedLMModel(
        model=_LM(),
        tokenizer_id="tok",
        eos_id=1,
        pad_id=0,
        max_seq_len=8,
        tok_for_dataset=_MiniEnc(),
    )


def _make_settings() -> Settings:
    """Create minimal test settings."""
    from model_trainer.core.config.settings import load_settings

    return load_settings()


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
        resume=False,
        progress=None,
        service_name="char-lstm-train",
    )

    with pytest.raises(RuntimeError, match="CUDA requested but not available"):
        _ = trainer._setup_device()


# ===== AMP (Automatic Mixed Precision) Tests =====


def test_get_autocast_context_fp32_returns_nullcontext() -> None:
    """Test that fp32 precision returns nullcontext (no autocast)."""
    ctx = bt_grad._get_autocast_context("fp32", torch.device("cpu"))
    # Verify the context is a no-op by entering and exiting it
    with ctx:
        pass  # No exception means it worked


def test_get_autocast_context_fp16_on_cpu_returns_nullcontext() -> None:
    """Test that fp16 on CPU returns nullcontext (autocast only on CUDA)."""
    ctx = bt_grad._get_autocast_context("fp16", torch.device("cpu"))
    # Verify the context is a no-op by entering and exiting it
    with ctx:
        pass  # No exception means it worked


def test_get_autocast_context_bf16_on_cpu_returns_nullcontext() -> None:
    """Test that bf16 on CPU returns nullcontext (autocast only on CUDA)."""
    ctx = bt_grad._get_autocast_context("bf16", torch.device("cpu"))
    # Verify the context is a no-op by entering and exiting it
    with ctx:
        pass  # No exception means it worked


def test_get_autocast_context_fp16_on_cuda() -> None:
    """Test that fp16 on CUDA returns autocast context."""
    # Create a mock CUDA device (doesn't require actual CUDA)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ctx = bt_grad._get_autocast_context("fp16", torch.device("cuda"))
        # Verify the context can be entered and exited
        with ctx:
            pass  # Autocast context entered successfully


def test_get_autocast_context_bf16_on_cuda() -> None:
    """Test that bf16 on CUDA returns autocast context."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        ctx = bt_grad._get_autocast_context("bf16", torch.device("cuda"))
        # Verify the context can be entered and exited
        with ctx:
            pass  # Autocast context entered successfully


def test_create_grad_scaler_returns_scaler() -> None:
    """Test that _create_grad_scaler returns a valid GradScaler."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        scaler = bt_grad._create_grad_scaler()
        # Verify it can scale a tensor (basic functionality check)
        t = torch.tensor(1.0, requires_grad=True)
        scaled = scaler.scale(t)
        assert scaled.item() >= 0.0  # Scaled value should be non-negative


def test_train_one_epoch_fp16_scaler_paths() -> None:
    """Test that fp16 precision with CUDA device uses GradScaler paths.

    This test covers lines 611-614 and 627-629 in base_trainer.py.
    Note: Without actual CUDA, the scaler is disabled but the code paths still execute.
    """

    class _DS(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
        def __len__(self: _DS) -> int:
            return 2

        def __getitem__(self: _DS, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
            ids = torch.randint(0, 4, (4,))
            return (ids, ids)

    dl = DataLoader(_DS(), batch_size=1, shuffle=False)

    # Create a real model with trainable parameters for proper GradScaler integration
    model = _LM()

    # Use a real optimizer that GradScaler can properly interact with
    # Access _p directly since model.parameters() returns ParameterLike protocol
    optim = torch.optim.SGD([model._p], lr=0.01)

    # Create config with fp16 precision
    cfg: ModelTrainConfig = {**_make_cfg(), "precision": "fp16"}

    trainer = bt.BaseTrainer(
        _make_prepared(),
        cfg,
        _make_settings(),
        run_id="test-fp16-run",
        redis_hb=lambda _: None,
        cancelled=lambda: False,
        resume=False,
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
            model=model,
            dataloader=dl,
            optim=optim,
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
        resume=False,
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
