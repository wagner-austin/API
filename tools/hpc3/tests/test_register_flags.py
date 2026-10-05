"""Each registration flag reads one answer, and refuses a malformed one by name."""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.cli import _register_flags as flags
from tests.conftest import cluster


class TestEveryFlagIsRequired:
    def test_all_missing_flags_are_named_at_once(self) -> None:
        with pytest.raises(ValueError, match=r"missing 2: \['--a', '--c'\]"):
            flags.require_every_flag({"--b": "1"}, ("--a", "--b", "--c"))


class TestNumbers:
    def test_a_whole_number_is_read(self) -> None:
        assert flags.positive_int("--cpus", "8") == 8

    @pytest.mark.parametrize("value", ["", "0", "-1", "2.5", "eight"])
    def test_anything_but_a_whole_number_of_at_least_one_is_refused(self, value: str) -> None:
        """Args:
        value: The malformed value.
        """
        with pytest.raises(ValueError, match="--cpus must be a whole number of at least 1"):
            _ = flags.positive_int("--cpus", value)

    @pytest.mark.parametrize(("value", "expected"), [("0", 0.0), ("12.5", 12.5), ("3.", 3.0)])
    def test_a_decimal_is_read(self, value: str, expected: float) -> None:
        """Args:
        value: The flag's value.
        expected: The number it means.
        """
        assert flags.non_negative_number("--gpu-hours", value) == expected

    @pytest.mark.parametrize("value", ["", ".5", "-1", "1.2.3", "lots"])
    def test_a_malformed_decimal_is_refused(self, value: str) -> None:
        """Args:
        value: The malformed value.
        """
        with pytest.raises(ValueError, match="--gpu-hours must be a number"):
            _ = flags.non_negative_number("--gpu-hours", value)


class TestYesOrNo:
    def test_yes_and_no_are_read(self) -> None:
        assert (flags.yes_or_no("--requeue", "yes"), flags.yes_or_no("--requeue", "no")) == (
            True,
            False,
        )

    def test_another_spelling_is_refused(self) -> None:
        with pytest.raises(ValueError, match="--requeue must be 'yes' or 'no', got 'true'"):
            _ = flags.yes_or_no("--requeue", "true")


class TestTheGpu:
    def test_none_is_cpu_only(self) -> None:
        assert flags.gpu_request(cluster(), "--gpu", "none") is None

    def test_a_model_and_count_is_read(self) -> None:
        assert flags.gpu_request(cluster(), "--gpu", "A100:2") == {"model": "A100", "count": 2}

    @pytest.mark.parametrize("value", ["A100", ":1", "cpu"])
    def test_a_malformed_request_is_refused(self, value: str) -> None:
        """Args:
        value: The malformed value.
        """
        with pytest.raises(ValueError, match="--gpu must be 'none' or MODEL:COUNT"):
            _ = flags.gpu_request(cluster(), "--gpu", value)

    def test_a_model_the_cluster_lacks_is_refused(self) -> None:
        with pytest.raises(AppError) as refused:
            _ = flags.gpu_request(cluster(), "--gpu", "H200:1")

        assert refused.value.code is Hpc3ErrorCode.GPU_TYPE_UNPINNED


class TestThePins:
    def test_none_pins_nothing(self) -> None:
        assert flags.pinned_packages("--pins", "none") == {}

    def test_pins_are_read_and_normalised(self) -> None:
        assert flags.pinned_packages("--pins", "Typing_Extensions==4.16.0,torch==2.6.0") == {
            "typing-extensions": "4.16.0",
            "torch": "2.6.0",
        }

    def test_an_entry_without_a_version_is_refused(self) -> None:
        with pytest.raises(ValueError, match="--pins entries must be name==version, got 'torch'"):
            _ = flags.pinned_packages("--pins", "numpy==2.3.5,torch")


class TestTheBudget:
    def test_free_spends_nothing_and_bills_no_account(self) -> None:
        assert flags.budget(12.0, "--billing", "free") == {
            "self_imposed_gpu_hours": 12.0,
            "max_service_units": 0.0,
            "charge_account": "",
        }

    def test_an_account_and_a_cap_are_read(self) -> None:
        assert flags.budget(0.0, "--billing", "mylab:500") == {
            "self_imposed_gpu_hours": 0.0,
            "max_service_units": 500.0,
            "charge_account": "mylab",
        }

    @pytest.mark.parametrize("value", ["mylab", ":500", "paid"])
    def test_a_malformed_billing_answer_is_refused(self, value: str) -> None:
        """Args:
        value: The malformed value.
        """
        with pytest.raises(ValueError, match="--billing must be 'free' or ACCOUNT:UNITS"):
            _ = flags.budget(0.0, "--billing", value)
