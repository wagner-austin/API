"""Tests for quantization configuration validation in runs.py.

EVERY QUANTIZATION FIELD IS REQUIRED, and most of this file exists to hold
that. The decoder used to default load_in_4bit to True, load_in_8bit to
False, bnb_4bit_compute_dtype to "float16" and bnb_4bit_quant_type to "nf4",
so a caller could post ``"quantization": {}`` and receive a fully specified
arm it never chose. The storage and compute data types are exactly what a
quantized run's numbers depend on, so each is now stated or refused.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, ErrorCode
from platform_core.json_utils import JSONValue

from model_trainer.api.schemas.runs import QuantizationConfigRequest
from model_trainer.api.validators.runs import _decode_train_request

_QUANT_FIELDS = (
    "load_in_4bit",
    "load_in_8bit",
    "bnb_4bit_compute_dtype",
    "bnb_4bit_quant_type",
    "bnb_4bit_use_double_quant",
)


def _base_qlora_payload() -> dict[str, JSONValue]:
    """Return base payload for qlora tests."""
    return {
        "model_family": "hf_lm",
        "model_size": "base",
        "max_seq_len": 128,
        "num_epochs": 1,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "corpus_file_id": "cid",
        "corpus_format": "lines",
        "hub_model_id": "bert-base",
        "finetuning_strategy": "qlora",
        "lora": {"r": 16},
        "user_id": 0,
    }


def _full_quant() -> dict[str, JSONValue]:
    """Return a complete quantization config, the paper's arm.

    Returns:
        Every field stated: NF4 storage, bfloat16 compute, double quant on.
    """
    return {
        "load_in_4bit": True,
        "load_in_8bit": False,
        "bnb_4bit_compute_dtype": "bfloat16",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
    }


def _decode_with(quant: dict[str, JSONValue]) -> QuantizationConfigRequest:
    """Decode a qlora request carrying the given quantization config.

    Args:
        quant: The quantization config to attach.

    Returns:
        The decoded quantization section.
    """
    payload = _base_qlora_payload()
    payload["quantization"] = quant
    out = _decode_train_request(payload)
    decoded = out["quantization"]
    assert decoded is not None
    return decoded


class TestEveryFieldIsRequired:
    """A quantization config that omits anything is refused."""

    def test_an_empty_config_is_refused(self) -> None:
        """The old behaviour returned a full arm here; now it is a 400."""
        payload = _base_qlora_payload()
        payload["quantization"] = {}

        with pytest.raises(AppError) as exc:
            _ = _decode_train_request(payload)

        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400

    @pytest.mark.parametrize("omitted", _QUANT_FIELDS)
    def test_omitting_any_single_field_is_refused_by_name(self, omitted: str) -> None:
        """Each field is individually required, and the error names it.

        Parametrised over the field tuple rather than written out five
        times, so adding a sixth field to the config without adding it to
        the decoder fails here.
        """
        quant = _full_quant()
        del quant[omitted]
        payload = _base_qlora_payload()
        payload["quantization"] = quant

        with pytest.raises(AppError) as exc:
            _ = _decode_train_request(payload)

        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400
        assert omitted in str(err.message)


class TestAcceptedValues:
    """A complete config decodes to exactly what it stated."""

    def test_a_full_config_round_trips(self) -> None:
        """The paper's arm survives the decoder unchanged."""
        decoded = _decode_with(_full_quant())

        assert decoded == _full_quant()

    def test_load_in_4bit_false_is_carried(self) -> None:
        """False is a value, not an absence."""
        quant = _full_quant()
        quant["load_in_4bit"] = False

        assert _decode_with(quant)["load_in_4bit"] is False

    def test_load_in_8bit_true_is_carried(self) -> None:
        """8-bit is selectable through the same config."""
        quant = _full_quant()
        quant["load_in_4bit"] = False
        quant["load_in_8bit"] = True

        assert _decode_with(quant)["load_in_8bit"] is True

    def test_double_quant_false_is_carried(self) -> None:
        """Double quantization is off-able, and off is recorded."""
        quant = _full_quant()
        quant["bnb_4bit_use_double_quant"] = False

        assert _decode_with(quant)["bnb_4bit_use_double_quant"] is False

    @pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32"])
    def test_every_compute_dtype_is_accepted(self, dtype: str) -> None:
        """All three named dtypes decode to themselves."""
        quant = _full_quant()
        quant["bnb_4bit_compute_dtype"] = dtype

        assert _decode_with(quant)["bnb_4bit_compute_dtype"] == dtype

    @pytest.mark.parametrize("quant_type", ["nf4", "fp4"])
    def test_both_quant_types_are_accepted(self, quant_type: str) -> None:
        """FP4 is accepted as well as NF4; the decoder does not prefer one."""
        quant = _full_quant()
        quant["bnb_4bit_quant_type"] = quant_type

        assert _decode_with(quant)["bnb_4bit_quant_type"] == quant_type


class TestRejectedValues:
    """Wrong types and unknown literals are refused."""

    @pytest.mark.parametrize("field", ["load_in_4bit", "load_in_8bit", "bnb_4bit_use_double_quant"])
    def test_a_non_boolean_flag_is_refused(self, field: str) -> None:
        """A string where a flag belongs names the field it came from."""
        quant = _full_quant()
        quant[field] = "yes"
        payload = _base_qlora_payload()
        payload["quantization"] = quant

        with pytest.raises(AppError) as exc:
            _ = _decode_train_request(payload)

        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400
        assert f"quantization.{field} must be a boolean" in str(err.message)

    def test_an_unknown_compute_dtype_is_refused(self) -> None:
        """The accepted set is closed."""
        quant = _full_quant()
        quant["bnb_4bit_compute_dtype"] = "invalid"
        payload = _base_qlora_payload()
        payload["quantization"] = quant

        with pytest.raises(AppError) as exc:
            _ = _decode_train_request(payload)

        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400

    def test_an_unknown_quant_type_is_refused(self) -> None:
        """int8 is not one of the two 4-bit storage types."""
        quant = _full_quant()
        quant["bnb_4bit_quant_type"] = "int8"
        payload = _base_qlora_payload()
        payload["quantization"] = quant

        with pytest.raises(AppError) as exc:
            _ = _decode_train_request(payload)

        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400

    def test_quantization_not_dict(self) -> None:
        """Test quantization config must be dict."""
        payload = _base_qlora_payload()
        payload["quantization"] = "not-a-dict"

        with pytest.raises(AppError) as exc:
            _ = _decode_train_request(payload)

        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400 and "quantization must be an object" in str(err.message)


class TestQuantizationBelongsToQlora:
    """Quantization is only meaningful under the strategy that applies it."""

    @pytest.mark.parametrize("strategy", ["full", "lora"])
    def test_a_non_qlora_strategy_carrying_quantization_is_refused(self, strategy: str) -> None:
        """The loader quantizes whenever the config is present.

        Accepting it under another strategy would train a quantized model
        while the run reported an unquantized one.
        """
        payload = _base_qlora_payload()
        payload["finetuning_strategy"] = strategy
        payload["quantization"] = _full_quant()

        with pytest.raises(AppError) as exc:
            _ = _decode_train_request(payload)

        err: AppError[ErrorCode] = exc.value
        assert err.http_status == 400
        assert "requires finetuning_strategy 'qlora'" in str(err.message)
