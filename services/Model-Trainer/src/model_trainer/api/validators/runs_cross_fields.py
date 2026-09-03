"""Rules BETWEEN the fields of a run request, rather than within one.

Split out of ``runs_config`` when that module passed the package's file-size
ceiling, and the seam is a real one: everything there decodes one field in
isolation, and everything here is about two fields that are each valid and
cannot both hold.

Every config-to-strategy rule is stated in BOTH directions -- the strategy
requires its config, and the config requires its strategy. A one-directional
rule accepts a request carrying settings nothing will read, which is how a
caller ends up believing they configured something they did not.
"""

from __future__ import annotations

from typing import Literal

from platform_core.errors import AppError, ErrorCode

from model_trainer.core.contracts.strategy_names import StrategyName

from ..schemas.runs import (
    CartridgeConfigRequest,
    GgufExportConfigRequest,
    LoraConfigRequest,
    QuantizationConfigRequest,
)


def _validate_hf_lm_cross_fields(
    model_family: Literal["gpt2", "llama", "qwen", "char_lstm", "hf_lm"],
    hub_model_id: str | None,
    finetuning_strategy: StrategyName,
    lora: LoraConfigRequest | None,
    cartridge: CartridgeConfigRequest | None,
    quantization: QuantizationConfigRequest | None,
    gguf_export: GgufExportConfigRequest | None,
) -> None:
    """Validate cross-field requirements for HF LM backend.

    Every config-to-strategy rule is stated in BOTH directions: the strategy
    requires its config, and the config requires its strategy. A one-directional
    rule accepts a request carrying settings nothing will read, which is how a
    caller ends up believing they configured something they did not.
    """
    if model_family == "hf_lm" and hub_model_id is None:
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="hub_model_id is required when model_family is 'hf_lm'",
            http_status=400,
        )
    if finetuning_strategy in ("lora", "qlora") and lora is None:
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message=f"lora config is required for finetuning_strategy '{finetuning_strategy}'",
            http_status=400,
        )
    if finetuning_strategy == "cartridge" and cartridge is None:
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message=(
                "cartridge config is required for finetuning_strategy 'cartridge'; the "
                "slot count decides how much the cartridge can hold and there is no "
                "defensible default"
            ),
            http_status=400,
        )
    if cartridge is not None and finetuning_strategy != "cartridge":
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message=(
                f"cartridge config requires finetuning_strategy 'cartridge', got "
                f"'{finetuning_strategy}'. No other strategy reads it, so accepting it "
                f"here would report a slot count that never becomes a cartridge"
            ),
            http_status=400,
        )
    if lora is not None and finetuning_strategy == "cartridge":
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message=(
                "lora config cannot be combined with finetuning_strategy 'cartridge'. A "
                "cartridge trains a key-value prefix and touches no weight, so there is "
                "no adapter for these settings to describe"
            ),
            http_status=400,
        )
    if finetuning_strategy == "qlora" and quantization is None:
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="quantization config is required for finetuning_strategy 'qlora'",
            http_status=400,
        )
    if quantization is not None and finetuning_strategy != "qlora":
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message=(
                f"quantization config requires finetuning_strategy 'qlora', got "
                f"'{finetuning_strategy}'. The loader applies quantization whenever this "
                f"config is present, so accepting it under another strategy would train a "
                f"quantized model while reporting an unquantized one"
            ),
            http_status=400,
        )
    if gguf_export is not None and finetuning_strategy not in ("lora", "qlora"):
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="gguf_export requires finetuning_strategy lora or qlora",
            http_status=400,
        )
