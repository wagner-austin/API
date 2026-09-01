"""Building a bitsandbytes config, and loading a model under it.

The three fields this exercises are the ones that decide what a quantized
run measures, and two of them carry upstream defaults that are NOT the QLoRA
paper's arm: ``BitsAndBytesConfig`` defaults ``bnb_4bit_quant_type`` to
``"fp4"`` and ``bnb_4bit_compute_dtype`` to ``torch.float32``. The paper
measures FP4 as roughly a percentage point behind the 16-bit LoRA baseline
where NF4 recovers it, so a loader that inherits either default trains a
different arm than the one requested, silently. These tests assert on the
config that is actually built rather than on the call being made.
"""

from __future__ import annotations

from typing import Literal, Protocol

import pytest
import torch

from model_trainer.core.contracts.model import QuantizationConfig
from model_trainer.core.services.model.backends.hf_lm._hook_protocols import (
    BitsAndBytesConfigProto,
)
from model_trainer.core.services.model.backends.hf_lm._test_hooks import (
    _bits_and_bytes_config,
    _compute_dtype,
    _default_load_hf_model,
    load_arguments,
)

_TINY_GPT2 = "sshleifer/tiny-gpt2"


class _BareConfigClassProto(Protocol):
    """Protocol for constructing a BitsAndBytesConfig with 4-bit alone.

    Deliberately narrower than the production class protocol, which requires
    every field. This one exists to build the config the loader never builds,
    so the defaults it inherits can be asserted.
    """

    def __call__(self, *, load_in_4bit: bool) -> BitsAndBytesConfigProto:
        """Build a config stating only that 4-bit is on.

        Args:
            load_in_4bit: Whether to load in 4-bit.

        Returns:
            The configuration, with every other field defaulted upstream.
        """
        ...


def _require_config(config: BitsAndBytesConfigProto | None) -> BitsAndBytesConfigProto:
    """Narrow a requested config, failing loudly when there is none.

    Args:
        config: What the loader would pass to from_pretrained.

    Returns:
        The config.

    Raises:
        AssertionError: When a quantized load requested no config at all,
            which would silently load the model unquantized.
    """
    if config is None:
        raise AssertionError("a quantized load requested no quantization config")
    return config


def _quant(
    *,
    compute_dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16",
    quant_type: Literal["nf4", "fp4"] = "nf4",
    double_quant: bool = True,
) -> QuantizationConfig:
    """Build a quantization config, defaulting to the paper's arm.

    Args:
        compute_dtype: The dtype 4-bit storage dequantizes to.
        quant_type: Storage data type, nf4 or fp4.
        double_quant: Whether to quantize the quantization constants.

    Returns:
        The configuration.
    """
    return QuantizationConfig(
        load_in_4bit=True,
        load_in_8bit=False,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=double_quant,
    )


class TestComputeDtypeResolution:
    """Every named dtype resolves to the torch dtype, and only those three."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("float16", torch.float16),
            ("bfloat16", torch.bfloat16),
            ("float32", torch.float32),
        ],
    )
    def test_each_name_resolves(
        self, name: Literal["float16", "bfloat16", "float32"], expected: torch.dtype
    ) -> None:
        """The mapping is explicit rather than getattr over the torch module.

        Args:
            name: The dtype name in the config.
            expected: The torch dtype it must resolve to.
        """
        assert _compute_dtype(_quant(compute_dtype=name)) is expected


class TestTheBuiltConfigStatesEveryField:
    """Nothing is left to the library's own defaults."""

    def test_the_papers_arm_survives_into_the_config(self) -> None:
        """NF4 storage, double quantization on, 4-bit selected."""
        built: BitsAndBytesConfigProto = _bits_and_bytes_config(_quant())

        assert built.load_in_4bit is True
        assert built.load_in_8bit is False
        assert built.bnb_4bit_quant_type == "nf4"
        assert built.bnb_4bit_use_double_quant is True

    def test_fp4_is_carried_rather_than_corrected(self) -> None:
        """The loader states what was asked for; it does not prefer NF4."""
        built = _bits_and_bytes_config(_quant(quant_type="fp4"))

        assert built.bnb_4bit_quant_type == "fp4"

    def test_double_quant_off_is_carried(self) -> None:
        """Off is a choice and has to survive to the library."""
        built = _bits_and_bytes_config(_quant(double_quant=False))

        assert built.bnb_4bit_use_double_quant is False

    def test_nf4_is_not_the_libraries_default(self) -> None:
        """Guards the reason every field is stated.

        A config built stating only ``load_in_4bit`` comes back as fp4,
        which is exactly why the loader never builds one that way. If
        upstream ever changes that default, this test fails and the comment
        explaining the loader stops being true at the same moment.
        """
        transformers = __import__("transformers", fromlist=["BitsAndBytesConfig"])
        config_cls: _BareConfigClassProto = transformers.BitsAndBytesConfig
        bare = config_cls(load_in_4bit=True)

        assert bare.bnb_4bit_quant_type == "fp4"


class TestTheArgumentsAQuantizationChoiceImplies:
    """What would be requested, asserted without needing a CUDA device."""

    def test_no_quantization_requests_no_config_and_float32(self) -> None:
        """The unquantized path states fp32 rather than inheriting it."""
        config, dtype = load_arguments(None)

        assert config is None
        assert dtype is torch.float32

    def test_quantization_requests_its_own_config_and_compute_dtype(self) -> None:
        """The dtype handed to the loader is the config's COMPUTE dtype.

        Not the storage type and not the library default: the paper pairs
        4-bit storage with a 16-bit compute type, and the compute type is
        what the non-quantized layers are loaded under.
        """
        config, dtype = load_arguments(_quant(compute_dtype="bfloat16"))

        assert _require_config(config).bnb_4bit_quant_type == "nf4"
        assert dtype is torch.bfloat16

    def test_a_float16_config_requests_float16(self) -> None:
        """The compute dtype is carried through, not normalised."""
        _, dtype = load_arguments(_quant(compute_dtype="float16"))

        assert dtype is torch.float16


class TestLoadingUnquantized:
    """The None path states its dtype rather than inheriting one."""

    def test_an_unquantized_load_produces_float32_parameters(self) -> None:
        """fp32 is what transformers documents as its own default.

        Passing it explicitly keeps existing unquantized runs byte-identical
        while removing the silent decision.
        """
        model = _default_load_hf_model(_TINY_GPT2, None)

        # Read dtypes off the state dict rather than off parameters(): the
        # parameter protocol describes gradients, not storage types.
        state: dict[str, torch.Tensor] = model.state_dict()
        assert state
        assert all(tensor.dtype is torch.float32 for tensor in state.values())
