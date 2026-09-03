"""Cartridge configuration at the HTTP edge, and the pairs it refuses.

The cross-field rules here are stated in both directions on purpose -- the
strategy requires its config, and the config requires its strategy -- so both
directions are tested. A one-directional rule accepts a request carrying
settings nothing will read, and the caller never learns their slot count was
ignored.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, ErrorCode
from platform_core.json_utils import JSONValue

from model_trainer.api.validators.runs import _decode_train_request


def base_payload() -> dict[str, JSONValue]:
    """Return a minimal hf_lm payload asking for the cartridge strategy.

    Returns:
        The payload.
    """
    return {
        "model_family": "hf_lm",
        "model_size": "base",
        "max_seq_len": 128,
        "num_epochs": 1,
        "batch_size": 4,
        "learning_rate": 1e-4,
        "corpus_file_id": "cid",
        "corpus_format": "lines",
        "hub_model_id": "gpt2",
        "finetuning_strategy": "cartridge",
        "user_id": 0,
        "cartridge": {"num_slots": 512, "init_seed": 7},
    }


class TestDecoding:
    """Reading a cartridge section out of a request body."""

    def test_a_complete_config_decodes(self) -> None:
        """The ordinary case."""
        decoded = _decode_train_request(base_payload())["cartridge"]
        assert decoded == {"enabled": True, "num_slots": 512, "init_seed": 7}

    def test_enabled_defaults_to_true(self) -> None:
        """Supplying the section is asking for it, matching how lora reads."""
        payload = base_payload()
        payload["cartridge"] = {"num_slots": 8, "init_seed": 1}
        decoded = _decode_train_request(payload)["cartridge"]
        assert decoded is not None and decoded["enabled"] is True

    def test_enabled_can_be_stated_false(self) -> None:
        """Accepted here and refused deeper in, where the strategy runs.

        The edge decodes what was sent; the contradiction between a cartridge
        strategy and a disabled cartridge is the strategy's to report, and it
        does, with its own code.
        """
        payload = base_payload()
        payload["cartridge"] = {"enabled": False, "num_slots": 8, "init_seed": 1}
        decoded = _decode_train_request(payload)["cartridge"]
        assert decoded is not None and decoded["enabled"] is False

    def test_a_non_boolean_enabled_is_refused(self) -> None:
        """A string "false" would otherwise read as true."""
        payload = base_payload()
        payload["cartridge"] = {"enabled": "no", "num_slots": 8, "init_seed": 1}
        with pytest.raises(AppError) as excinfo:
            _decode_train_request(payload)
        assert excinfo.value.code is ErrorCode.INVALID_INPUT

    def test_a_missing_slot_count_is_refused(self) -> None:
        """There is no defensible default capacity."""
        payload = base_payload()
        payload["cartridge"] = {"init_seed": 1}
        with pytest.raises(AppError) as excinfo:
            _decode_train_request(payload)
        assert excinfo.value.code is ErrorCode.INVALID_INPUT

    def test_a_zero_slot_count_is_refused(self) -> None:
        """Zero slots holds no parameters and would train having learned nothing."""
        payload = base_payload()
        payload["cartridge"] = {"num_slots": 0, "init_seed": 1}
        with pytest.raises(AppError) as excinfo:
            _decode_train_request(payload)
        assert excinfo.value.code is ErrorCode.INVALID_INPUT

    def test_a_slot_count_above_the_ceiling_is_refused(self) -> None:
        """Above the largest size the method has published measurements for."""
        payload = base_payload()
        payload["cartridge"] = {"num_slots": 8193, "init_seed": 1}
        with pytest.raises(AppError) as excinfo:
            _decode_train_request(payload)
        assert excinfo.value.code is ErrorCode.INVALID_INPUT

    def test_the_largest_published_size_is_accepted(self) -> None:
        """The ceiling is inclusive, so the boundary itself must pass."""
        payload = base_payload()
        payload["cartridge"] = {"num_slots": 8192, "init_seed": 1}
        decoded = _decode_train_request(payload)["cartridge"]
        assert decoded is not None and decoded["num_slots"] == 8192

    def test_a_missing_seed_is_refused(self) -> None:
        """A run that cannot say what it started from cannot be repeated."""
        payload = base_payload()
        payload["cartridge"] = {"num_slots": 8}
        with pytest.raises(AppError) as excinfo:
            _decode_train_request(payload)
        assert excinfo.value.code is ErrorCode.INVALID_INPUT

    def test_a_negative_seed_is_refused(self) -> None:
        """Torch generators take non-negative seeds."""
        payload = base_payload()
        payload["cartridge"] = {"num_slots": 8, "init_seed": -1}
        with pytest.raises(AppError) as excinfo:
            _decode_train_request(payload)
        assert excinfo.value.code is ErrorCode.INVALID_INPUT

    def test_a_non_object_cartridge_is_refused(self) -> None:
        """A list or a string is not a config."""
        payload = base_payload()
        payload["cartridge"] = "512"
        with pytest.raises(AppError) as excinfo:
            _decode_train_request(payload)
        assert excinfo.value.code is ErrorCode.INVALID_INPUT


class TestTheCrossFieldRules:
    """A cartridge config and the cartridge strategy require each other."""

    def test_the_strategy_without_its_config_is_refused(self) -> None:
        """Selecting the strategy and omitting the section says nothing about size."""
        payload = base_payload()
        del payload["cartridge"]
        with pytest.raises(AppError) as excinfo:
            _decode_train_request(payload)
        assert excinfo.value.code is ErrorCode.INVALID_INPUT
        assert "cartridge config is required" in str(excinfo.value)

    def test_the_config_without_its_strategy_is_refused(self) -> None:
        """The other direction: no other strategy reads it.

        Without this, a request naming ``full`` and carrying a slot count is
        accepted, trains every parameter, and never tells the caller their
        cartridge settings were ignored.
        """
        payload = base_payload()
        payload["finetuning_strategy"] = "full"
        with pytest.raises(AppError) as excinfo:
            _decode_train_request(payload)
        assert excinfo.value.code is ErrorCode.INVALID_INPUT
        assert "requires finetuning_strategy 'cartridge'" in str(excinfo.value)

    def test_a_lora_config_cannot_be_combined_with_the_cartridge_strategy(self) -> None:
        """A cartridge touches no weight, so there is no adapter to describe."""
        payload = base_payload()
        payload["lora"] = {"r": 8}
        with pytest.raises(AppError) as excinfo:
            _decode_train_request(payload)
        assert excinfo.value.code is ErrorCode.INVALID_INPUT
        assert "cannot be combined" in str(excinfo.value)

    def test_a_request_without_any_cartridge_fields_is_unaffected(self) -> None:
        """The rules must not fire on the three strategies that predate them."""
        payload = base_payload()
        payload["finetuning_strategy"] = "full"
        del payload["cartridge"]
        assert _decode_train_request(payload)["cartridge"] is None
