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

from model_trainer.core.contracts.model import PreparedLMModel, QuantizationConfig
from model_trainer.core.contracts.queue_encoding_configs import (
    _decode_optional_quantization,
    encode_quantization_config,
)
from model_trainer.core.contracts.strategy_names import StrategyName, require_strategy_name
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

METADATA_NAME = "hf_lm_metadata.json"
"""What a saved run's metadata file is called inside its artifact directory.

Named once because three readers now open it: a resumed run, an arm being
reloaded for evaluation, and the paired control that arm is compared
against.
"""


class HFLMMetadata(TypedDict):
    """Metadata stored with HF LM model artifacts.

    For HF LM models, tokenizer_id may be None because the HF tokenizer
    is loaded from hub_model_id directly.
    """

    strategy_name: StrategyName
    hub_model_id: str
    tokenizer_id: str | None  # None for HF LM (uses HF tokenizer from hub_model_id)
    is_peft: bool
    # What the base model was quantized to when this run trained, or None.
    # The adapter on disk is a delta against those weights, so reloading it
    # onto a differently-quantized base reconstructs a different model.
    quantization: QuantizationConfig | None


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
        "quantization": (
            None
            if metadata["quantization"] is None
            else encode_quantization_config(metadata["quantization"])
        ),
    }
    return result


def _require_strategy_name(obj: JSONObject, key: str) -> StrategyName:
    """Read a strategy name out of a saved metadata object.

    Splits the two failures rather than merging them. A missing or non-string
    field is a SHAPE fault in the file and stays a ``JSONTypeError`` from
    ``require_str``; a well-formed string naming no strategy is a VALUE fault
    and carries ``STRATEGY_NAME_UNKNOWN``, the same code the request path
    raises for the same mistake.

    Args:
        obj: JSON object to extract from.
        key: Key to extract.

    Returns:
        The strategy name, typed.

    Raises:
        JSONTypeError: If the field is missing or is not a string.
        AppError: With ``STRATEGY_NAME_UNKNOWN`` if the string names no
            declared strategy.
    """
    return require_strategy_name(require_str(obj, key))


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
    quantization = _decode_optional_quantization(data)

    return HFLMMetadata(
        strategy_name=strategy_name,
        hub_model_id=hub_model_id,
        tokenizer_id=tokenizer_id,
        is_peft=is_peft,
        quantization=quantization,
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

    strategy_name = require_strategy_name(strategy_name_raw)

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
        quantization=prepared.quantization,
    )
    metadata_path = out_path / METADATA_NAME
    metadata_path.write_text(
        dump_json_str(_encode_metadata(metadata)),
        encoding="utf-8",
    )


def read_hf_lm_metadata(artifact_path: str) -> HFLMMetadata:
    """Read what a saved run recorded about how it was loaded.

    Args:
        artifact_path: Path to saved model directory.

    Returns:
        The run's metadata.

    Raises:
        FileNotFoundError: If the metadata file is missing. A directory
            without one is not a saved run, and guessing its strategy or its
            quantization would reconstruct a different model while reporting
            success.
    """
    metadata_path = Path(artifact_path) / METADATA_NAME

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")

    metadata_json = load_json_str(metadata_path.read_text(encoding="utf-8"))
    return _decode_metadata(narrow_json_to_dict(metadata_json))


def _prepared_around(
    model: LMModelProto,
    metadata: HFLMMetadata,
    *,
    strategy_name: StrategyName | None,
    is_peft: bool,
) -> PreparedLMModel:
    """Wrap a loaded model in the tokenizer and token ids its run used.

    Shared by the two loaders below so that an arm and its control differ in
    the model they hold and in nothing else. Duplicating this was how the
    tokenizer, the eos id or the max sequence length could come to differ
    between two things being subtracted from each other.

    Args:
        model: The loaded model, adapted or not.
        metadata: What the run recorded.
        strategy_name: The finetuning strategy applied to ``model``, or None
            when none was.
        is_peft: Whether ``model`` carries adapter modules.

    Returns:
        The prepared model.
    """
    hf_tokenizer = HFHooks.load_hf_tokenizer(metadata["hub_model_id"])
    eos_id, pad_id, _ = _token_ids_from_hf_tokenizer(hf_tokenizer)

    return PreparedLMModel(
        model=model,
        tokenizer_id=metadata["tokenizer_id"],
        eos_id=eos_id,
        pad_id=pad_id,
        max_seq_len=_get_model_max_seq_len(model),
        tok_for_dataset=HFTokenizerEncoder(hf_tokenizer),
        strategy_name=strategy_name,
        hub_model_id=metadata["hub_model_id"],
        is_peft=is_peft,
        quantization=metadata["quantization"],
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
    metadata = read_hf_lm_metadata(artifact_path)

    # Reloading a saved run: the adapter is re-attached to a base model loaded
    # the same way the run loaded it, so a quantized run reloads quantized.
    base_model = HFHooks.load_hf_model(metadata["hub_model_id"], metadata["quantization"])

    registry = default_registry()
    strategy = registry.get(metadata["strategy_name"])
    adapted = strategy.load_adapted(base_model, metadata["hub_model_id"], artifact_path)

    return _prepared_around(
        adapted.model,
        metadata,
        strategy_name=metadata["strategy_name"],
        is_peft=metadata["is_peft"],
    )


def load_base_of_prepared_hf_lm(artifact_path: str) -> PreparedLMModel:
    """Load the base a saved run adapted, WITHOUT reapplying the adapter.

    This is the paired control for :func:`load_prepared_hf_lm_from_handle`,
    and it is a different thing from :func:`load_prepared_hf_lm_from_hub`.
    That one loads unquantized weights, because a baseline exists to be
    compared against and must carry no arm. This one deliberately DOES carry
    the arm's quantization, because it is not a baseline -- it is the control
    for one specific adapter, and the question being asked is what the
    adapter did.

    An adapter trained against NF4 weights and compared against bfloat16
    ones would be a comparison of two changes at once: the adapter and the
    dequantization. Reading the quantization out of the run's own metadata,
    rather than accepting it as an argument, is what makes that impossible
    to get wrong by hand.

    Args:
        artifact_path: Path to the saved model directory whose base is
            wanted.

    Returns:
        PreparedLMModel holding the unadapted base, loaded exactly as the
        run loaded it.

    Raises:
        FileNotFoundError: If metadata file is missing.
        RuntimeError: If required hooks are not configured.
    """
    metadata = read_hf_lm_metadata(artifact_path)
    base_model = HFHooks.load_hf_model(metadata["hub_model_id"], metadata["quantization"])

    # Recorded as having no strategy and no adapter, which is the whole
    # difference between this and the arm -- and it is recorded rather than
    # implied so a run record cannot mistake the control for the arm.
    return _prepared_around(base_model, metadata, strategy_name=None, is_peft=False)


def load_prepared_hf_lm_from_hub(hub_model_id: str) -> PreparedLMModel:
    """Load an untrained model straight from the hub, with no adaptation.

    This is the baseline counterpart of
    :func:`load_prepared_hf_lm_from_handle`. That one reads an artifact
    directory and asks a finetuning strategy to reapply whatever was trained;
    this one deliberately does neither, because a baseline is defined by having
    nothing applied to it. The two share the tokenizer and token-id handling so
    a baseline and an arm are scored through identical machinery.

    Args:
        hub_model_id: HuggingFace model id, for example ``gpt2-medium``.

    Returns:
        PreparedLMModel holding the untouched pretrained weights.
    """
    # Baseline scoring of an untrained model: never quantized, because the
    # baseline exists to be compared against and must not carry an arm.
    base_model = HFHooks.load_hf_model(hub_model_id, None)
    hf_tokenizer = HFHooks.load_hf_tokenizer(hub_model_id)
    eos_id, pad_id, _ = _token_ids_from_hf_tokenizer(hf_tokenizer)

    return PreparedLMModel(
        model=base_model,
        # A baseline has no trained tokenizer of its own; the hub model's own
        # tokenizer is the only one that matches these weights.
        tokenizer_id=None,
        eos_id=eos_id,
        pad_id=pad_id,
        max_seq_len=_get_model_max_seq_len(base_model),
        tok_for_dataset=HFTokenizerEncoder(hf_tokenizer),
        # No finetuning strategy was applied, which is the defining property of
        # a baseline -- recorded as absent rather than as a "none" sentinel.
        strategy_name=None,
        hub_model_id=hub_model_id,
        is_peft=False,
        # Loaded above with quantization=None, and recorded as such so the
        # baseline cannot be mistaken for a quantized comparison arm.
        quantization=None,
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
    "METADATA_NAME",
    "HFLMMetadata",
    "load_base_of_prepared_hf_lm",
    "load_prepared_hf_lm_from_handle",
    "load_prepared_hf_lm_from_hub",
    "read_hf_lm_metadata",
    "save_prepared_hf_lm",
]
