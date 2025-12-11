from __future__ import annotations

from pathlib import Path
from typing import Protocol

from model_trainer.core.contracts.model import PreparedLMModel
from model_trainer.core.contracts.tokenizer import TokenizerHandle
from model_trainer.core.encoding import Encoder, HandleEncoder
from model_trainer.core.types import ConfigLike, LMModelProto


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


def load_encoder_for_dataset(tokenizer_path: str) -> Encoder:
    # Leverage BPE backend loader to get a TokenizerHandle, then adapt.
    from model_trainer.core.services.tokenizer.bpe_backend import BPEBackend

    handle = BPEBackend().load(tokenizer_path)
    return HandleEncoder(handle)


def save_prepared_gpt2(prepared: PreparedLMModel, out_dir: str) -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    prepared.model.save_pretrained(out_dir)


def get_model_max_seq_len(model: LMModelProto) -> int:
    # Default behavior: look for config.n_positions attribute, else 512
    cfg: ConfigLike = model.config
    val: int | None = getattr(cfg, "n_positions", None)
    if isinstance(val, int):
        return val
    return 512


def load_prepared_gpt2_from_handle(
    artifact_path: str, tokenizer: TokenizerHandle
) -> PreparedLMModel:
    from model_trainer.core import _test_hooks

    eos_id, pad_id, _ = token_ids(tokenizer)
    model = _test_hooks.load_gpt2_model(artifact_path)
    max_seq_len = get_model_max_seq_len(model)
    return PreparedLMModel(
        model=model,
        tokenizer_id="unknown",
        eos_id=eos_id,
        pad_id=pad_id,
        max_seq_len=max_seq_len,
        tok_for_dataset=encoder_from_handle(tokenizer),
    )


class _TokWrapper:
    # Minimal adapter used in tests to provide token info interface
    def __init__(self: _TokWrapper, handle: _TokenInfoProto) -> None:
        self._h: _TokenInfoProto = handle

    def token_to_id(self: _TokWrapper, token: str) -> int | None:
        return self._h.token_to_id(token)

    def get_vocab_size(self: _TokWrapper) -> int:
        return self._h.get_vocab_size()
