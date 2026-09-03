"""The geometry a cartridge is cut to, and what it refuses to describe.

A geometry is the difference between a block of numbers and a cartridge. These
cover the round trip and, more importantly, every shape that must NOT decode:
a zero count reads as valid arithmetic and describes an object with no
parameters, which would train to completion and learn nothing.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, ModelTrainerErrorCode
from platform_core.json_utils import JSONObject, JSONTypeError, dump_json_str, load_json_str

from model_trainer.core.contracts.cartridge import (
    CARTRIDGE_MANIFEST_NAME,
    CARTRIDGE_WEIGHTS_NAME,
    CartridgeGeometry,
    decode_cartridge_geometry,
    encode_cartridge_geometry,
    trainable_parameter_count,
)
from model_trainer.core.contracts.model import CartridgeConfig
from model_trainer.core.contracts.queue_encoding_configs import (
    _decode_optional_cartridge,
    decode_cartridge_config,
    encode_cartridge_config,
)


def make_geometry(
    *,
    num_layers: int = 2,
    num_kv_heads: int = 4,
    head_dim: int = 8,
    num_slots: int = 16,
) -> CartridgeGeometry:
    """Build a geometry for testing.

    Args:
        num_layers: Layers the prefix spans.
        num_kv_heads: Key-value heads per layer.
        head_dim: Width of one head's vectors.
        num_slots: Prefix positions.

    Returns:
        The geometry.
    """
    return CartridgeGeometry(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        num_slots=num_slots,
    )


class TestTheRoundTrip:
    """A geometry written to disk must come back as the same object."""

    def test_it_survives_json(self) -> None:
        """Through a real serialise and parse, not just the two functions."""
        geometry = make_geometry()
        restored = decode_cartridge_geometry(
            load_json_str(dump_json_str(encode_cartridge_geometry(geometry)))
        )
        assert restored == geometry

    def test_the_encoding_carries_exactly_the_declared_fields(self) -> None:
        """An extra field would be written and then silently dropped on read."""
        encoded = encode_cartridge_geometry(make_geometry())
        assert sorted(encoded) == ["head_dim", "num_kv_heads", "num_layers", "num_slots"]


class TestWhatIsRefused:
    """Every count is positive, and zero is the interesting case."""

    @pytest.mark.parametrize(
        "field",
        ["num_layers", "num_kv_heads", "head_dim", "num_slots"],
    )
    def test_a_zero_count_is_refused_in_every_field(self, field: str) -> None:
        """Zero is arithmetically fine and describes nothing trainable."""
        encoded = encode_cartridge_geometry(make_geometry())
        encoded[field] = 0
        with pytest.raises(AppError) as excinfo:
            decode_cartridge_geometry(encoded)
        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_GEOMETRY_INVALID

    def test_a_negative_count_is_refused(self) -> None:
        """Negative would reach torch and fail far from the file that caused it."""
        encoded = encode_cartridge_geometry(make_geometry())
        encoded["num_slots"] = -1
        with pytest.raises(AppError) as excinfo:
            decode_cartridge_geometry(encoded)
        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_GEOMETRY_INVALID

    def test_the_refusal_names_the_field_and_the_value(self) -> None:
        """A manifest with four counts needs to say which one is wrong."""
        encoded = encode_cartridge_geometry(make_geometry())
        encoded["head_dim"] = 0
        with pytest.raises(AppError) as excinfo:
            decode_cartridge_geometry(encoded)
        message = str(excinfo.value)
        assert "'head_dim'" in message
        assert "got 0" in message

    def test_a_non_object_is_refused_with_the_geometry_code(self) -> None:
        """A truncated or replaced manifest is not a generic decode failure."""
        with pytest.raises(AppError) as excinfo:
            decode_cartridge_geometry([1, 2, 3])
        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_GEOMETRY_INVALID

    def test_a_missing_field_is_a_shape_fault(self) -> None:
        """Absent is different from invalid, and stays a decode error."""
        encoded = encode_cartridge_geometry(make_geometry())
        del encoded["num_slots"]
        with pytest.raises(JSONTypeError):
            decode_cartridge_geometry(encoded)

    def test_a_non_integer_count_is_a_shape_fault(self) -> None:
        """A float slot count cannot index a tensor."""
        encoded = encode_cartridge_geometry(make_geometry())
        encoded["num_slots"] = "sixteen"
        with pytest.raises(JSONTypeError):
            decode_cartridge_geometry(encoded)


class TestTheTrainableCount:
    """What a cartridge run actually updates."""

    def test_it_counts_both_blocks_of_every_layer(self) -> None:
        """Keys and values, hence twice the product of the other three."""
        geometry = make_geometry(num_layers=2, num_kv_heads=4, head_dim=8, num_slots=16)
        assert trainable_parameter_count(geometry) == 2 * 2 * 4 * 16 * 8

    def test_it_scales_linearly_with_slots(self) -> None:
        """The slot count is the capacity knob, so this is the cost of turning it."""
        small = trainable_parameter_count(make_geometry(num_slots=16))
        large = trainable_parameter_count(make_geometry(num_slots=32))
        assert large == 2 * small


class TestTheOnDiskNames:
    """The two files a saved cartridge is made of."""

    def test_the_manifest_is_json_and_the_weights_are_not(self) -> None:
        """The manifest is readable without torch, which is why it is separate."""
        assert CARTRIDGE_MANIFEST_NAME.endswith(".json")
        assert CARTRIDGE_WEIGHTS_NAME.endswith(".pt")

    def test_the_two_names_differ(self) -> None:
        """One would overwrite the other."""
        assert CARTRIDGE_MANIFEST_NAME != CARTRIDGE_WEIGHTS_NAME


class TestTheRequestConfigCodec:
    """The cartridge section as it crosses the job queue.

    Distinct from the geometry above: that is what a trained cartridge IS, this
    is what a caller ASKED for. Both travel, and both are validated on the way
    back in, because a worker reading a payload has no more reason to trust it
    than an HTTP handler does.
    """

    def _config(self) -> CartridgeConfig:
        """Build a request config.

        Returns:
            The config.
        """
        return CartridgeConfig(enabled=True, num_slots=512, init_seed=7)

    def test_it_round_trips_through_json(self) -> None:
        """Through a real serialise and parse."""
        parsed = load_json_str(dump_json_str(encode_cartridge_config(self._config())))
        if not isinstance(parsed, dict):
            raise TypeError("an encoded cartridge config must parse back to an object")
        assert decode_cartridge_config(parsed) == self._config()

    def test_the_encoding_carries_exactly_the_declared_fields(self) -> None:
        """An extra field would be written and silently dropped on read."""
        assert sorted(encode_cartridge_config(self._config())) == [
            "enabled",
            "init_seed",
            "num_slots",
        ]

    def test_a_zero_slot_count_is_refused_off_the_queue(self) -> None:
        """The same refusal the HTTP edge makes, because the queue is a second
        entry point and a payload can be written by hand."""
        encoded = encode_cartridge_config(self._config())
        encoded["num_slots"] = 0
        with pytest.raises(AppError) as excinfo:
            decode_cartridge_config(encoded)
        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_GEOMETRY_INVALID

    def test_a_negative_slot_count_is_refused_off_the_queue(self) -> None:
        """Negative would reach torch and fail far from the payload that caused it."""
        encoded = encode_cartridge_config(self._config())
        encoded["num_slots"] = -8
        with pytest.raises(AppError) as excinfo:
            decode_cartridge_config(encoded)
        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_GEOMETRY_INVALID

    def test_a_missing_field_is_a_shape_fault(self) -> None:
        """Absent is different from invalid, and stays a decode error."""
        encoded = encode_cartridge_config(self._config())
        del encoded["init_seed"]
        with pytest.raises(JSONTypeError):
            decode_cartridge_config(encoded)

    def test_an_absent_section_decodes_to_none(self) -> None:
        """Most runs are not cartridge runs."""
        assert _decode_optional_cartridge({"lora": None}) is None

    def test_a_null_section_decodes_to_none(self) -> None:
        """An explicit null and an absent key mean the same thing here."""
        assert _decode_optional_cartridge({"cartridge": None}) is None

    def test_a_present_section_decodes(self) -> None:
        """The path a cartridge run takes off the queue."""
        payload: JSONObject = {"cartridge": encode_cartridge_config(self._config())}
        assert _decode_optional_cartridge(payload) == self._config()

    def test_a_non_object_section_is_refused(self) -> None:
        """A number where a config belongs is a malformed payload."""
        with pytest.raises(JSONTypeError):
            _decode_optional_cartridge({"cartridge": 512})
