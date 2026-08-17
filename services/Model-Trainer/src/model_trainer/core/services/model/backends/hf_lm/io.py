"""HuggingFace LM save/load with finetuning strategy support.

Handles saving and loading models including LoRA adapters via the
finetuning strategy system.
"""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    optional_str,
    require_bool,
    require_str,
)

from model_trainer.core.contracts.finetuning import StrategyName
from model_trainer.core.contracts.model import PreparedLMModel
from model_trainer.core.contracts.tokenizer import TokenizerHandle
from model_trainer.core.services.finetuning import default_registry
from model_trainer.core.services.model.backends.hf_lm._test_hooks import (
    Hooks as HFHooks,
)
from model_trainer.core.services.model.backends.hf_lm.prepare import (
    HFTokenizerEncoder,
    _token_ids_from_hf_tokenizer,
)
from model_trainer.core.types import LMModelProto


class HFLMMetadata(TypedDict):
    """Metadata stored with HF LM model artifacts.

    For HF LM models, tokenizer_id may be None because the HF tokenizer
    is loaded from hub_model_id directly.
    """

    strategy_name: StrategyName
    hub_model_id: str
    tokenizer_id: str | None  # None for HF LM (uses HF tokenizer from hub_model_id)
    is_peft: bool


# Valid strategy names as a set for validation
_VALID_STRATEGY_NAMES: frozenset[str] = frozenset(["full", "lora", "qlora"])


def _validate_strategy_name(value: str) -> StrategyName:
    """Validate a string is a valid strategy name.

    Args:
        value: String to validate.

    Returns:
        Validated StrategyName.

    Raises:
        ValueError: If value is not a valid strategy name.
    """
    if value == "full":
        return "full"
    if value == "lora":
        return "lora"
    if value == "qlora":
        return "qlora"
    valid = ", ".join(sorted(_VALID_STRATEGY_NAMES))
    raise ValueError(f"Invalid strategy name '{value}', must be one of [{valid}]")


def _encode_metadata(metadata: HFLMMetadata) -> JSONObject:
    """Encode HFLMMetadata to JSON-serializable dict.

    Args:
        metadata: Metadata to encode.

    Returns:
        JSON-serializable dictionary.
    """
    result: JSONObject = {
        "strategy_name": metadata["strategy_name"],
        "hub_model_id": metadata["hub_model_id"],
        "tokenizer_id": metadata["tokenizer_id"],
        "is_peft": metadata["is_peft"],
    }
    return result


def _require_strategy_name(obj: JSONObject, key: str) -> StrategyName:
    """Extract and validate strategy name from JSON object.

    Args:
        obj: JSON object to extract from.
        key: Key to extract.

    Returns:
        Validated StrategyName.

    Raises:
        JSONTypeError: If field is missing or not a valid strategy name.
    """
    from platform_core.json_utils import JSONTypeError

    value = require_str(obj, key)
    # Explicit conditionals to narrow str to StrategyName literal type
    if value == "full":
        return "full"
    if value == "lora":
        return "lora"
    if value == "qlora":
        return "qlora"
    valid = ", ".join(sorted(_VALID_STRATEGY_NAMES))
    raise JSONTypeError(f"Field '{key}' must be one of [{valid}], got '{value}'")


def _decode_metadata(data: JSONObject) -> HFLMMetadata:
    """Decode JSON object to HFLMMetadata.

    Args:
        data: JSON dictionary to decode.

    Returns:
        Validated HFLMMetadata.

    Raises:
        JSONTypeError: If field types are incorrect or invalid.
    """
    strategy_name = _require_strategy_name(data, "strategy_name")
    hub_model_id = require_str(data, "hub_model_id")
    # tokenizer_id is optional for HF LM (uses HF tokenizer from hub_model_id)
    tokenizer_id = optional_str(data, "tokenizer_id")
    is_peft = require_bool(data, "is_peft")

    return HFLMMetadata(
        strategy_name=strategy_name,
        hub_model_id=hub_model_id,
        tokenizer_id=tokenizer_id,
        is_peft=is_peft,
    )


def save_prepared_hf_lm(
    prepared: PreparedLMModel,
    out_dir: str,
) -> None:
    """Save HuggingFace LM model to disk.

    For PEFT models, saves adapter weights via strategy.
    For full models, saves complete weights.

    Reads strategy_name, hub_model_id, and is_peft from PreparedLMModel's
    optional fields (set during prepare_hf_lm_with_handle).

    Args:
        prepared: Prepared model with adapter metadata fields set.
        out_dir: Output directory path.

    Raises:
        ValueError: If required metadata fields are missing.
        RuntimeError: If required save hook is not configured.
    """
    strategy_name_raw = prepared.strategy_name
    hub_model_id = prepared.hub_model_id
    is_peft = prepared.is_peft

    if strategy_name_raw is None:
        raise ValueError("PreparedLMModel.strategy_name is required for hf_lm save")
    if hub_model_id is None:
        raise ValueError("PreparedLMModel.hub_model_id is required for hf_lm save")

    # Validate and narrow strategy_name to StrategyName literal type
    strategy_name = _validate_strategy_name(strategy_name_raw)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Get strategy and delegate save
    registry = default_registry()
    strategy = registry.get(strategy_name)

    # Create minimal adapted model info for save
    from model_trainer.core.contracts.finetuning import AdaptedModel

    adapted = AdaptedModel(
        model=prepared.model,
        base_model_id=hub_model_id,
        strategy_name=strategy_name,
        is_peft_model=is_peft,
        lora_config=None,  # Not needed for save
    )
    strategy.save_adapted(adapted, out_dir)

    # Write metadata
    metadata = HFLMMetadata(
        strategy_name=strategy_name,
        hub_model_id=hub_model_id,
        tokenizer_id=prepared.tokenizer_id,
        is_peft=is_peft,
    )
    metadata_path = out_path / "hf_lm_metadata.json"
    metadata_path.write_text(
        dump_json_str(_encode_metadata(metadata)),
        encoding="utf-8",
    )


def load_prepared_hf_lm_from_handle(
    artifact_path: str,
    tokenizer: TokenizerHandle | None,
) -> PreparedLMModel:
    """Load HuggingFace LM model from saved artifact.

    Reads metadata to determine strategy and loads appropriately.

    The tokenizer parameter is optional for HF LM models because the tokenizer
    is loaded from hub_model_id stored in metadata. The parameter is accepted
    for protocol compatibility but not used.

    Args:
        artifact_path: Path to saved model directory.
        tokenizer: Optional TokenizerHandle (unused - HF tokenizer from hub is used).

    Returns:
        PreparedLMModel ready for inference or continued training.

    Raises:
        FileNotFoundError: If metadata file is missing.
        RuntimeError: If required hooks are not configured.
    """
    # Note: tokenizer parameter is unused - HF LM uses tokenizer from hub_model_id
    del tokenizer  # Explicitly mark as unused to satisfy linters
    artifact = Path(artifact_path)
    metadata_path = artifact / "hf_lm_metadata.json"

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    metadata_json = load_json_str(metadata_path.read_text(encoding="utf-8"))
    metadata_obj = narrow_json_to_dict(metadata_json)
    metadata = _decode_metadata(metadata_obj)

    # Load base model from HuggingFace
    load_model = HFHooks.load_hf_model
    load_hf_tokenizer = HFHooks.load_hf_tokenizer

    base_model = load_model(metadata["hub_model_id"])
    hf_tokenizer = load_hf_tokenizer(metadata["hub_model_id"])

    # Get strategy and load adapted weights
    registry = default_registry()
    strategy = registry.get(metadata["strategy_name"])
    adapted = strategy.load_adapted(base_model, metadata["hub_model_id"], artifact_path)

    # Extract token IDs
    eos_id, pad_id, _ = _token_ids_from_hf_tokenizer(hf_tokenizer)

    return PreparedLMModel(
        model=adapted.model,
        tokenizer_id=metadata["tokenizer_id"],
        eos_id=eos_id,
        pad_id=pad_id,
        max_seq_len=_get_model_max_seq_len(adapted.model),
        tok_for_dataset=HFTokenizerEncoder(hf_tokenizer),
        strategy_name=metadata["strategy_name"],
        hub_model_id=metadata["hub_model_id"],
        is_peft=metadata["is_peft"],
    )


def _get_model_max_seq_len(model: LMModelProto) -> int:
    """Extract max sequence length from model config.

    Args:
        model: Model with config attribute.

    Returns:
        Maximum sequence length, defaults to 2048.
    """
    config = model.config

    # Try max_position_embeddings (common in GPT-2, BERT style)
    _attr_mpe: str = "max_position_embeddings"
    if hasattr(config, _attr_mpe):
        val_mpe: int = getattr(config, _attr_mpe)
        return val_mpe

    # Try n_positions (GPT-2 style)
    _attr_np: str = "n_positions"
    if hasattr(config, _attr_np):
        val_np: int = getattr(config, _attr_np)
        return val_np

    # Try max_seq_length (generic)
    _attr_msl: str = "max_seq_length"
    if hasattr(config, _attr_msl):
        val_msl: int = getattr(config, _attr_msl)
        return val_msl

    return 2048


__all__ = [
    "HFLMMetadata",
    "load_prepared_hf_lm_from_handle",
    "save_prepared_hf_lm",
]
