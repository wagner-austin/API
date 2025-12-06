from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import torch
import torch.nn as _nn
import torch.nn.functional as _functional
from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for
from platform_core.json_utils import JSONValue, dump_json_str, load_json_str

from model_trainer.core.types import ForwardOutProto, LMModelProto, NamedParameter, ParameterLike


class _ForwardOut(ForwardOutProto):
    def __init__(self: _ForwardOut, loss: torch.Tensor) -> None:
        self._loss = loss

    @property
    def loss(self: _ForwardOut) -> torch.Tensor:
        return self._loss


class _Config:
    def __init__(
        self: _Config,
        *,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        max_seq_len: int,
    ) -> None:
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.n_positions = max_seq_len  # align with GPT-2 config field for max length


class CharLSTM(_nn.Module):
    def __init__(
        self: CharLSTM,
        *,
        vocab_size: int,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.config = _Config(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self.embedding = _nn.Embedding(vocab_size, embed_dim)
        self.lstm = _nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.proj = _nn.Linear(hidden_dim, vocab_size)
        # No weight tying to keep dimensions independent of hidden vs embed sizes.

    def forward(
        self: CharLSTM, *, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> ForwardOutProto:
        # input_ids: [B, T], labels: [B, T]
        emb: torch.Tensor = self.embedding(input_ids)
        out_tuple: tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]] = self.lstm(emb)
        out: torch.Tensor = out_tuple[0]
        logits: torch.Tensor = self.proj(out)
        # Shift for next-token prediction
        # Flatten to [b*t, v]
        b = int(logits.size(0))
        t = int(logits.size(1))
        v = int(logits.size(2))
        logits_flat: torch.Tensor = logits.reshape(b * t, v)
        labels_flat: torch.Tensor = labels.reshape(b * t)
        loss: torch.Tensor = _functional.cross_entropy(logits_flat, labels_flat, reduction="mean")
        return _ForwardOut(loss)

    # Inherit parameters()/train()/eval()/to() from nn.Module

    # Save/load in a minimal, explicit format compatible with LMModelProto
    def save_pretrained(self: CharLSTM, out_dir: str) -> None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        class _HasStateDict(Protocol):
            def state_dict(self: _HasStateDict) -> dict[str, torch.Tensor]: ...

        self_typed: _HasStateDict = self
        sd_obj: dict[str, torch.Tensor] = self_typed.state_dict()
        torch.save(sd_obj, os.path.join(out_dir, "model.pt"))
        cfg: dict[str, JSONValue] = {
            "vocab_size": int(self.config.vocab_size),
            "embed_dim": int(self.config.embed_dim),
            "hidden_dim": int(self.config.hidden_dim),
            "num_layers": int(self.config.num_layers),
            "dropout": float(self.config.dropout),
            "max_seq_len": int(self.config.n_positions),
        }
        Path(out_dir).joinpath("config.json").write_text(dump_json_str(cfg), encoding="utf-8")

    @classmethod
    def from_pretrained(cls: type[CharLSTM], path: str) -> CharLSTM:
        base = Path(path)
        cfg_text = base.joinpath("config.json").read_text(encoding="utf-8")
        cfg_obj = load_json_str(cfg_text)
        if not isinstance(cfg_obj, dict):
            raise AppError(
                ModelTrainerErrorCode.MODEL_LOAD_FAILED,
                "invalid char_lstm config.json",
                model_trainer_status_for(ModelTrainerErrorCode.MODEL_LOAD_FAILED),
            )

        def _get_int(name: str) -> int:
            v: JSONValue = cfg_obj.get(name)
            if isinstance(v, int):
                return int(v)
            raise AppError(
                ModelTrainerErrorCode.MODEL_LOAD_FAILED,
                f"invalid field {name} in char_lstm config",
                model_trainer_status_for(ModelTrainerErrorCode.MODEL_LOAD_FAILED),
            )

        def _get_float(name: str) -> float:
            v: JSONValue = cfg_obj.get(name)
            if isinstance(v, float):
                return float(v)
            if isinstance(v, int):
                return float(v)
            raise AppError(
                ModelTrainerErrorCode.MODEL_LOAD_FAILED,
                f"invalid field {name} in char_lstm config",
                model_trainer_status_for(ModelTrainerErrorCode.MODEL_LOAD_FAILED),
            )

        vocab_size_o = _get_int("vocab_size")
        embed_dim_o = _get_int("embed_dim")
        hidden_dim_o = _get_int("hidden_dim")
        layers_o = _get_int("num_layers")
        dropout_o = _get_float("dropout")
        maxlen_o = _get_int("max_seq_len")
        model = cls(
            vocab_size=vocab_size_o,
            embed_dim=embed_dim_o,
            hidden_dim=hidden_dim_o,
            num_layers=layers_o,
            dropout=dropout_o,
            max_seq_len=maxlen_o,
        )

        class _TorchLoad(Protocol):
            def __call__(
                self: _TorchLoad, f: str, *, map_location: str, weights_only: bool
            ) -> dict[str, torch.Tensor]: ...

        torch_mod = __import__("torch")
        load_fn: _TorchLoad = torch_mod.load
        sd_loaded = load_fn(str(base.joinpath("model.pt")), map_location="cpu", weights_only=True)
        model.load_state_dict(sd_loaded)
        return model


class CharLSTMModel(LMModelProto):
    """Typed wrapper exposing LMModelProto over a CharLSTM module."""

    def __init__(self: CharLSTMModel, inner: CharLSTM) -> None:
        self._m = inner

    @classmethod
    def from_pretrained(cls: type[CharLSTMModel], path: str) -> LMModelProto:
        return cls(CharLSTM.from_pretrained(path))

    def train(self: CharLSTMModel) -> None:
        self._m.train()

    def eval(self: CharLSTMModel) -> None:
        self._m.eval()

    def forward(
        self: CharLSTMModel, *, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> ForwardOutProto:
        return self._m.forward(input_ids=input_ids, labels=labels)

    def forward_logits(
        self: CharLSTMModel,
        *,
        input_ids: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass returning logits and hidden state without computing loss.

        Used for inference (scoring and generation). Supports stateful generation
        by accepting and returning LSTM hidden state (h_n, c_n).

        Args:
            input_ids: Input tensor of shape [B, T] containing token indices.
            hidden: Optional tuple of (h_n, c_n) hidden states from previous step.
                    Each tensor has shape [num_layers, B, hidden_dim].

        Returns:
            Tuple of (logits, hidden_state) where:
            - logits: Shape [B, T, V] vocabulary logits
            - hidden_state: Tuple of (h_n, c_n) for next step
        """
        emb: torch.Tensor = self._m.embedding(input_ids)
        out_tuple: tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]] = self._m.lstm(
            emb, hidden
        )
        out: torch.Tensor = out_tuple[0]
        new_hidden: tuple[torch.Tensor, torch.Tensor] = out_tuple[1]
        logits: torch.Tensor = self._m.proj(out)
        return logits, new_hidden

    def parameters(self: CharLSTMModel) -> Sequence[ParameterLike]:
        # Convert iterator to a concrete sequence for typing contract
        return list(self._m.parameters())

    def named_parameters(self: CharLSTMModel) -> Sequence[tuple[str, NamedParameter]]:
        # Convert iterator to a concrete sequence for typing contract
        return list(self._m.named_parameters())

    def to(self: CharLSTMModel, device: str) -> LMModelProto:
        self._m.to(device)
        return self

    def save_pretrained(self: CharLSTMModel, out_dir: str) -> None:
        self._m.save_pretrained(out_dir)

    @property
    def config(self: CharLSTMModel) -> _Config:
        return self._m.config
