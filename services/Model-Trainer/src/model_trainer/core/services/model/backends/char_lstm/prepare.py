from __future__ import annotations

from platform_core.errors import AppError, ModelTrainerErrorCode, model_trainer_status_for

from model_trainer.core.contracts.model import ModelTrainConfig, PreparedLMModel
from model_trainer.core.contracts.tokenizer import TokenizerHandle
from model_trainer.core.encoding import Encoder

from .io import encoder_from_handle, token_ids
from .model import CharLSTM, CharLSTMModel


def _size_to_dims(size: str) -> tuple[int, int, int, float]:
    # Returns (embed_dim, hidden_dim, num_layers, dropout)
    if size == "tiny":
        return 128, 256, 2, 0.10
    if size == "small":
        return 256, 512, 2, 0.10
    if size == "medium":
        return 384, 768, 3, 0.10
    raise AppError(
        ModelTrainerErrorCode.INVALID_MODEL_SIZE,
        "invalid model_size for char_lstm",
        model_trainer_status_for(ModelTrainerErrorCode.INVALID_MODEL_SIZE),
    )


def _encoder_for_dataset(tok: TokenizerHandle) -> Encoder:
    return encoder_from_handle(tok)


def prepare_char_lstm_with_handle(
    tokenizer: TokenizerHandle, cfg: ModelTrainConfig
) -> PreparedLMModel:
    eos_id, pad_id, vocab_size = token_ids(tokenizer)
    embed_dim, hidden_dim, num_layers, dropout = _size_to_dims(cfg["model_size"])
    raw = CharLSTM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        max_seq_len=cfg["max_seq_len"],
    )
    return PreparedLMModel(
        model=CharLSTMModel(raw),
        tokenizer_id=cfg["tokenizer_id"],
        eos_id=eos_id,
        pad_id=pad_id,
        max_seq_len=cfg["max_seq_len"],
        tok_for_dataset=_encoder_for_dataset(tokenizer),
    )
