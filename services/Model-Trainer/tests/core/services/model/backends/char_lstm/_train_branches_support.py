"""Shared fakes for the char_lstm train-branch tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from platform_core.determinism_record import UNPINNED_STACK, determinism_record
from platform_core.json_utils import JSONValue
from platform_ml.testing import WandbTableProtocol

from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.model import ModelTrainConfig, PreparedLMModel
from model_trainer.core.encoding import Encoder, ListEncoded
from model_trainer.core.services.training import base_trainer as bt
from model_trainer.core.types import (
    ConfigLike,
    ForwardOutProto,
    LMModelProto,
    LoadStateDictResultProto,
    NamedParameter,
    ParameterLike,
)

UNPINNED = determinism_record(UNPINNED_STACK, {})
"""What these tests actually ran under, not a placeholder.

A unit test constructing a trainer pins nothing, so "deliberately not
pinned" is the true posture and the record says exactly that. Passing it
explicitly is the point of `determinism` having no default: a test that
supplied nothing would have written a manifest claiming a posture no process
ever put in force.
"""


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
        "corpus_format": "lines",
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
        "cartridge": None,
        "quantization": None,
        "gguf_export": None,
    }


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
        determinism=UNPINNED,
    )
    trainer._device = torch.device("cpu")
    return trainer


def _make_settings() -> Settings:
    """Create minimal test settings."""
    from model_trainer.core.config.settings import load_settings

    return load_settings()


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
