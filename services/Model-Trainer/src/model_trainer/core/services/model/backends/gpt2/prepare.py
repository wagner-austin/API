from __future__ import annotations

from model_trainer.core.contracts.model import ModelTrainConfig, PreparedLMModel
from model_trainer.core.contracts.tokenizer import TokenizerHandle

from .hf_gpt2 import create_gpt2_model
from .io import encoder_from_handle, token_ids


def prepare_gpt2_with_handle(
    tokenizer: TokenizerHandle | None, cfg: ModelTrainConfig
) -> PreparedLMModel:
    """Prepare a GPT-2 model for training using a tokenizer handle.

    Creates a new GPT2LMHeadModel with the specified configuration.

    Args:
        tokenizer: TokenizerHandle for encoding text. Required for GPT-2.
        cfg: GPT-2 training configuration including model_size and max_seq_len.

    Returns:
        PreparedLMModel containing the model and tokenizer information.

    Raises:
        ValueError: If tokenizer is None or tokenizer_id is None.
        AppError: If model_size in cfg is not valid. Propagated from
            create_gpt2_config, which resolves the size.
    """
    if tokenizer is None:
        raise ValueError("tokenizer is required for gpt2 backend")
    tokenizer_id = cfg["tokenizer_id"]
    if tokenizer_id is None:
        raise ValueError("tokenizer_id is required for gpt2 backend")

    eos_id, pad_id, vocab_size = token_ids(tokenizer)
    model = create_gpt2_model(
        vocab_size=vocab_size,
        max_seq_len=cfg["max_seq_len"],
        model_size=cfg["model_size"],
    )
    return PreparedLMModel(
        model=model,
        tokenizer_id=tokenizer_id,
        eos_id=eos_id,
        pad_id=pad_id,
        max_seq_len=cfg["max_seq_len"],
        tok_for_dataset=encoder_from_handle(tokenizer),
    )
