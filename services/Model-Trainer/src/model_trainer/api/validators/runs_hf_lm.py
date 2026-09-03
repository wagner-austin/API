"""Decoding the HuggingFace-backend half of a train request.

Split out of ``runs`` when that module passed the package's file-size ceiling.
The seam is the backend: everything left in ``runs`` decodes fields every model
family has, and this decodes the ones only the ``hf_lm`` backend reads --
which adapter or prefix to train, and how it is configured.
"""

from __future__ import annotations

from typing import Literal

from platform_core.json_utils import JSONValue
from platform_core.validators import validate_optional_literal, validate_str

from model_trainer.api.validators.runs_config import (
    _decode_optional_cartridge,
    _decode_optional_gguf_export,
    _decode_optional_lora,
    _decode_optional_quantization,
    _narrow_finetuning_strategy,
)
from model_trainer.api.validators.runs_cross_fields import _validate_hf_lm_cross_fields
from model_trainer.core.contracts.strategy_names import STRATEGY_NAMES, StrategyName

from ..schemas.runs import (
    CartridgeConfigRequest,
    GgufExportConfigRequest,
    LoraConfigRequest,
    QuantizationConfigRequest,
)

#: Derived from the StrategyName Literal rather than restated, so the HTTP
#: layer cannot accept a strategy the registry has never heard of.
_FINETUNING_STRATEGIES: frozenset[str] = frozenset(STRATEGY_NAMES)


class _HfLmFields:
    """Container for decoded HF LM backend fields."""

    hub_model_id: str | None
    finetuning_strategy: StrategyName
    lora: LoraConfigRequest | None
    cartridge: CartridgeConfigRequest | None
    quantization: QuantizationConfigRequest | None
    gguf_export: GgufExportConfigRequest | None

    def __init__(
        self,
        hub_model_id: str | None,
        finetuning_strategy: StrategyName,
        lora: LoraConfigRequest | None,
        cartridge: CartridgeConfigRequest | None,
        quantization: QuantizationConfigRequest | None,
        gguf_export: GgufExportConfigRequest | None,
    ) -> None:
        self.hub_model_id = hub_model_id
        self.finetuning_strategy = finetuning_strategy
        self.lora = lora
        self.cartridge = cartridge
        self.quantization = quantization
        self.gguf_export = gguf_export


def _decode_hf_lm_fields(
    d: dict[str, JSONValue],
    model_family: Literal["gpt2", "llama", "qwen", "char_lstm", "hf_lm"],
) -> _HfLmFields:
    """Decode and validate HuggingFace LM backend fields.

    Args:
        d: Raw request dictionary.
        model_family: Validated model family.

    Returns:
        _HfLmFields container with decoded values.

    Raises:
        AppError: If validation fails.
    """
    hub_model_id_raw = d.get("hub_model_id")
    hub_model_id: str | None = None
    if hub_model_id_raw is not None:
        hub_model_id = validate_str(hub_model_id_raw, "hub_model_id")

    finetuning_strategy_raw = validate_optional_literal(
        d.get("finetuning_strategy"), "finetuning_strategy", _FINETUNING_STRATEGIES
    )
    finetuning_strategy = _narrow_finetuning_strategy(finetuning_strategy_raw)

    lora = _decode_optional_lora(d)
    cartridge = _decode_optional_cartridge(d)
    quantization = _decode_optional_quantization(d)
    gguf_export = _decode_optional_gguf_export(d)

    _validate_hf_lm_cross_fields(
        model_family,
        hub_model_id,
        finetuning_strategy,
        lora,
        cartridge,
        quantization,
        gguf_export,
    )

    return _HfLmFields(
        hub_model_id=hub_model_id,
        finetuning_strategy=finetuning_strategy,
        lora=lora,
        cartridge=cartridge,
        quantization=quantization,
        gguf_export=gguf_export,
    )
