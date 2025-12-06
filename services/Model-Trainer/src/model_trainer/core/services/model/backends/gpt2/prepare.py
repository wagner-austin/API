from __future__ import annotations

from model_trainer.core.contracts.model import ModelTrainConfig, PreparedLMModel
from model_trainer.core.contracts.tokenizer import TokenizerHandle

from .hf_gpt2 import create_gpt2_model
from .io import encoder_from_handle, token_ids


def prepare_gpt2_with_handle(tokenizer: TokenizerHandle, cfg: ModelTrainConfig) -> PreparedLMModel:
    """Prepare a GPT-2 model for training using a tokenizer handle.

    Creates a new GPT2LMHeadModel with the specified configuration.

    Args:
        tokenizer: TokenizerHandle for encoding text.
        cfg: GPT-2 training configuration including model_size and max_seq_len.

    Returns:
        PreparedLMModel containing the model and tokenizer information.

    Raises:
        KeyError: If model_size in cfg is not valid.
    """
    eos_id, pad_id, vocab_size = token_ids(tokenizer)
    model = create_gpt2_model(
        vocab_size=vocab_size,
        max_seq_len=cfg["max_seq_len"],
        model_size=cfg["model_size"],
    )
    return PreparedLMModel(
        model=model,
        tokenizer_id=cfg["tokenizer_id"],
        eos_id=eos_id,
        pad_id=pad_id,
        max_seq_len=cfg["max_seq_len"],
        tok_for_dataset=encoder_from_handle(tokenizer),
    )
