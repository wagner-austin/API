"""Per-base loading and adaptation policy for the LoRA sweeps.

Split from ``cartridge_base_lora_sweep`` when the 7B rung's policy pushed
that module over the size ceiling -- and the split is by role, not by
line count: everything here answers "how does THIS base load and where
does its adapter land", which both sweep CLIs must answer identically and
neither should own.

Both maps are EXPLICIT and refuse unknown bases. A heuristic would
silently adapt the wrong modules on a new architecture, or load an
undeclared 7B in fp32 and OOM a batch job three hours in; a refusal here
costs a line in this file instead.
"""

from __future__ import annotations

from model_trainer.core.contracts.model import QuantizationConfig, StoredBf16Precision

#: The GPT-2 family fuses query, key and value into one projection; adapting
#: it is adapting attention. A constant rather than a plan field because it
#: is a property of the architecture the plan's ``model_id`` names, not a
#: knob anyone sweeps.
LORA_TARGET_MODULES = ("c_attn",)

#: GPT-NeoX's fused projection: the ``c_attn`` analogue by another name.
NEOX_TARGET_MODULES = ("query_key_value",)

#: The 7B rung's precision: NF4 with double quantization and bf16 compute,
#: the QLoRA paper's recovering configuration. Forced by arithmetic -- two
#: fp32 7B models are 56GB against a 40GB card -- and also the
#: deployment-realistic form, which is what the 7B rung exists to measure.
PYTHIA_7B_QUANTIZATION: QuantizationConfig = {
    "load_in_4bit": True,
    "load_in_8bit": False,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": "bfloat16",
    "bnb_4bit_use_double_quant": True,
}


def target_modules_for(model_id: str) -> tuple[str, ...]:
    """Name the attention modules LoRA adapts on this base.

    An explicit map, not a heuristic: an unknown base must refuse here,
    before an adapter lands on the wrong modules and trains a run whose
    record claims something else.

    Args:
        model_id: The plan's base model.

    Returns:
        The module names PEFT targets.

    Raises:
        ValueError: For a base no row here names.
    """
    if model_id.startswith("gpt2"):
        return LORA_TARGET_MODULES
    if model_id.startswith("EleutherAI/pythia"):
        return NEOX_TARGET_MODULES
    raise ValueError(
        f"no LoRA target modules are declared for base {model_id!r}; declare "
        f"the architecture's attention projection here before running it"
    )


def quantization_for(model_id: str) -> QuantizationConfig | None:
    """Decide how this base loads: full precision or the 7B rung's NF4.

    Explicit ids, not a size heuristic: precision is part of what a record
    means, so which bases quantize is declared, and an unknown base refuses
    rather than silently loading fp32.

    Args:
        model_id: The plan's base model.

    Returns:
        The quantization to load with, or None for stored-precision fp32.

    Raises:
        ValueError: For a base no row here names.
    """
    if model_id in ("gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"):
        return None
    if model_id == "EleutherAI/pythia-6.9b":
        return PYTHIA_7B_QUANTIZATION
    raise ValueError(
        f"no loading precision is declared for base {model_id!r}; declare it "
        f"here -- a defaulted fp32 on an undeclared 7B is an OOM in a batch "
        f"job three hours from now"
    )


def stored_bf16_for(model_id: str) -> StoredBf16Precision:
    """Declare the unquantized-bf16 load for a base measured that way.

    The 7B precision-control arm (task ``c4b9a01b``): the NF4 rung's
    training deficit needs a comparison with the quantization removed,
    and 7B fp32 (27.6GB) fits no free card where bf16 (13.8GB) does.
    Explicit ids like every map here: an undeclared base refuses, so a
    control that silently ran at the wrong precision cannot exist.

    Args:
        model_id: The plan's base model.

    Returns:
        The declared stored-bf16 load.

    Raises:
        ValueError: For a base no row here names -- including the whole
            gpt2 family, whose certified records are fp32 and which has
            no declared bf16 arm.
    """
    if model_id == "EleutherAI/pythia-6.9b":
        return {"torch_dtype": "bfloat16"}
    raise ValueError(
        f"no stored-bf16 load is declared for base {model_id!r}; the only "
        f"declared precision-control arm is the 7B one, and an undeclared "
        f"half-precision load would change what a record means silently"
    )


__all__ = [
    "LORA_TARGET_MODULES",
    "NEOX_TARGET_MODULES",
    "PYTHIA_7B_QUANTIZATION",
    "quantization_for",
    "stored_bf16_for",
    "target_modules_for",
]
