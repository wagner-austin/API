"""HuggingFace LM model preparation with finetuning strategy support.

Loads pretrained models from HuggingFace Hub and applies the configured
finetuning strategy (full, lora, qlora, unsloth).
"""

from __future__ import annotations

from model_trainer.core.contracts.finetuning import StrategyName
from model_trainer.core.contracts.model import ModelTrainConfig, PreparedLMModel
from model_trainer.core.contracts.tokenizer import TokenizerHandle
from model_trainer.core.encoding import Encoded, ListEncoded
from model_trainer.core.services.finetuning import default_registry
from model_trainer.core.services.model.backends.hf_lm._test_hooks import (
    HFTokenizerProto,
    Hooks,
)


class HFTokenizerEncoder:
    """Encoder adapter for HuggingFace tokenizers.

    Adapts HFTokenizerProto to the Encoder protocol expected by
    model training components.
    """

    def __init__(self, tokenizer: HFTokenizerProto) -> None:
        """Initialize with HuggingFace tokenizer.

        Args:
            tokenizer: HuggingFace tokenizer instance.
        """
        self._tok = tokenizer

    def encode(self, text: str) -> Encoded:
        """Encode text to token IDs.

        Args:
            text: Input text string.

        Returns:
            Encoded object with ids property.
        """
        ids = self._tok.encode(text)
        return ListEncoded(ids)

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            ids: Token IDs to decode.

        Returns:
            Decoded text string.
        """
        return self._tok.decode(ids)

    def token_to_id(self, token: str) -> int | None:
        """Convert a token to its ID.

        Args:
            token: Token string to convert.

        Returns:
            Token ID, or None if token not in vocabulary.
        """
        # HuggingFace returns unk_token_id for unknown tokens, not None
        # We return the ID as-is since the token exists in vocab
        return self._tok.convert_tokens_to_ids(token)

    def get_vocab_size(self) -> int:
        """Get the vocabulary size.

        Returns:
            Number of tokens in vocabulary.
        """
        return len(self._tok)


def _require_hub_model_id(cfg: ModelTrainConfig) -> str:
    """Extract and validate hub_model_id from config.

    Args:
        cfg: Model training configuration.

    Returns:
        The hub_model_id string.

    Raises:
        ValueError: If hub_model_id is None.
    """
    hub_model_id = cfg["hub_model_id"]
    if hub_model_id is None:
        raise ValueError("hub_model_id is required for hf_lm backend")
    return hub_model_id


def _require_finetuning_strategy(cfg: ModelTrainConfig) -> StrategyName:
    """Extract finetuning_strategy from config.

    Args:
        cfg: Model training configuration.

    Returns:
        The finetuning_strategy as StrategyName.
    """
    return cfg["finetuning_strategy"]


def _token_ids_from_hf_tokenizer(
    tokenizer: HFTokenizerProto,
) -> tuple[int, int, int]:
    """Extract special token IDs from HuggingFace tokenizer.

    Args:
        tokenizer: HuggingFace tokenizer instance.

    Returns:
        Tuple of (eos_id, pad_id, vocab_size).
    """
    eos_id_opt = tokenizer.eos_token_id
    eos_id = eos_id_opt if eos_id_opt is not None else 0
    pad_id_opt = tokenizer.pad_token_id
    pad_id = pad_id_opt if pad_id_opt is not None else eos_id
    vocab_size = len(tokenizer)
    return eos_id, pad_id, vocab_size


def prepare_hf_lm_with_handle(
    tokenizer: TokenizerHandle | None,
    cfg: ModelTrainConfig,
) -> PreparedLMModel:
    """Prepare a HuggingFace LM model for training.

    Loads a pretrained model from HuggingFace Hub and applies the configured
    finetuning strategy (full, lora, qlora, unsloth).

    The tokenizer parameter is optional for HF LM models because the tokenizer
    is loaded from hub_model_id. The parameter is accepted for protocol
    compatibility but not used - the HF tokenizer is used for dataset encoding.

    Args:
        tokenizer: Optional TokenizerHandle (unused - HF tokenizer from hub is used).
        cfg: Training configuration including hub_model_id and finetuning_strategy.

    Returns:
        PreparedLMModel ready for training.

    Raises:
        ValueError: If hub_model_id or finetuning_strategy is missing.
        RuntimeError: If required hooks are not configured.
    """
    # Note: tokenizer parameter is unused - HF LM uses tokenizer from hub_model_id
    del tokenizer  # Explicitly mark as unused to satisfy linters
    hub_model_id = _require_hub_model_id(cfg)
    strategy_name = _require_finetuning_strategy(cfg)

    # Load HF model and tokenizer via hooks
    load_model = Hooks.load_hf_model
    load_tokenizer = Hooks.load_hf_tokenizer

    base_model = load_model(hub_model_id)
    hf_tokenizer = load_tokenizer(hub_model_id)

    # Get finetuning strategy and adapt the model
    registry = default_registry()
    strategy = registry.get(strategy_name)
    adapted = strategy.adapt(base_model, hub_model_id, cfg)

    # Extract token IDs from HF tokenizer
    eos_id, pad_id, _ = _token_ids_from_hf_tokenizer(hf_tokenizer)

    return PreparedLMModel(
        model=adapted.model,
        tokenizer_id=cfg["tokenizer_id"],
        eos_id=eos_id,
        pad_id=pad_id,
        max_seq_len=cfg["max_seq_len"],
        tok_for_dataset=HFTokenizerEncoder(hf_tokenizer),
        strategy_name=strategy_name,
        hub_model_id=hub_model_id,
        is_peft=adapted.is_peft_model,
    )


__all__ = [
    "HFTokenizerEncoder",
    "_token_ids_from_hf_tokenizer",
    "prepare_hf_lm_with_handle",
]
