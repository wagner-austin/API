from __future__ import annotations

from typing import Protocol

from model_trainer.core.contracts.model import PreparedLMModel
from model_trainer.core.contracts.tokenizer import TokenizerHandle
from model_trainer.core.encoding import Encoder, HandleEncoder
from model_trainer.core.types import ConfigLike

from .model import CharLSTM, CharLSTMModel


def encoder_from_handle(handle: TokenizerHandle) -> Encoder:
    return HandleEncoder(handle)


class _TokenInfoProto(Protocol):
    def token_to_id(self: _TokenInfoProto, token: str) -> int | None: ...
    def get_vocab_size(self: _TokenInfoProto) -> int: ...


def token_ids(tokenizer: _TokenInfoProto) -> tuple[int, int, int]:
    eos_id_opt = tokenizer.token_to_id("[EOS]")
    eos_id = int(eos_id_opt) if eos_id_opt is not None else 0
    pad_id_opt = tokenizer.token_to_id("[PAD]")
    pad_id = int(pad_id_opt) if pad_id_opt is not None else 0
    vocab_size = int(tokenizer.get_vocab_size())
    return eos_id, pad_id, vocab_size


def save_prepared_char_lstm(prepared: PreparedLMModel, out_dir: str) -> None:
    prepared.model.save_pretrained(out_dir)


def get_model_max_seq_len(model: CharLSTM) -> int:
    cfg: ConfigLike = model.config
    val: int | None = getattr(cfg, "n_positions", None)
    if isinstance(val, int):
        return val
    return 256


def load_prepared_char_lstm_from_handle(
    artifact_path: str, tokenizer: TokenizerHandle
) -> PreparedLMModel:
    # Load raw model then wrap into LMModelProto implementation
    raw = CharLSTM.from_pretrained(artifact_path)
    eos_id, pad_id, _ = token_ids(tokenizer)
    max_seq_len = get_model_max_seq_len(raw)
    return PreparedLMModel(
        model=CharLSTMModel(raw),
        tokenizer_id="unknown",
        eos_id=eos_id,
        pad_id=pad_id,
        max_seq_len=max_seq_len,
        tok_for_dataset=encoder_from_handle(tokenizer),
    )
