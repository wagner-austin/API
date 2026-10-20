"""The four facts every runner install declares, held to one rule on both platforms."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue

from fleet.contracts.runner_registration import (
    LINUX_UNIT_SUFFIX,
    RunnerRegistration,
    decode_runner_registration,
    encode_runner_registration,
    runner_service_name,
)
from fleet.contracts.runners import decode_runner_install


def _raw(unit_suffix: str) -> JSONObject:
    """lavender's MCPs runner as the roster spells it, with a platform's suffix.

    Args:
        unit_suffix: The platform's service suffix.

    Returns:
        The raw registration.
    """
    labels: list[JSONValue] = ["lavender"]
    return {
        "repo": "wagner-austin/MCPs",
        "runner_name": "lavender-2",
        "service": f"actions.runner.wagner-austin-MCPs.lavender-2{unit_suffix}",
        "labels": labels,
    }


class TestTheServiceName:
    @pytest.mark.parametrize(
        ("unit_suffix", "service"),
        [
            (LINUX_UNIT_SUFFIX, "actions.runner.wagner-austin-MCPs.lavender-2.service"),
            ("", "actions.runner.wagner-austin-MCPs.lavender-2"),
        ],
    )
    def test_the_configuration_s_name_carries_the_platform_s_suffix(
        self, unit_suffix: str, service: str
    ) -> None:
        assert (
            runner_service_name("wagner-austin/MCPs", "lavender-2", unit_suffix=unit_suffix)
            == service
        )


class TestTheDecoder:
    @pytest.mark.parametrize("unit_suffix", [LINUX_UNIT_SUFFIX, ""])
    def test_a_registration_round_trips_on_either_platform(self, unit_suffix: str) -> None:
        decoded = decode_runner_registration(_raw(unit_suffix), unit_suffix=unit_suffix)
        assert decoded == RunnerRegistration(
            repo="wagner-austin/MCPs",
            runner_name="lavender-2",
            service=f"actions.runner.wagner-austin-MCPs.lavender-2{unit_suffix}",
            labels=["lavender"],
        )
        assert encode_runner_registration(decoded) == _raw(unit_suffix)

    def test_a_windows_service_name_is_refused_for_a_linux_unit(self) -> None:
        with pytest.raises(JSONTypeError) as refused:
            decode_runner_registration(_raw(""), unit_suffix=LINUX_UNIT_SUFFIX)
        assert str(refused.value) == (
            "service must be 'actions.runner.wagner-austin-MCPs.lavender-2.service', the name "
            "the runner's configuration gives wagner-austin/MCPs's runner lavender-2; "
            "got 'actions.runner.wagner-austin-MCPs.lavender-2'"
        )

    @pytest.mark.parametrize(
        ("key", "value", "message"),
        [
            ("repo", "MCPs", "repo must be 'owner/name', got 'MCPs'"),
            ("runner_name", "", "runner_name must be non-empty"),
        ],
    )
    def test_a_field_github_could_not_register_is_refused(
        self, key: str, value: str, message: str
    ) -> None:
        raw = _raw(LINUX_UNIT_SUFFIX)
        raw[key] = value
        with pytest.raises(JSONTypeError) as refused:
            decode_runner_registration(raw, unit_suffix=LINUX_UNIT_SUFFIX)
        assert str(refused.value) == message


class TestTheLinuxInstall:
    """The Linux decoder now holds an install to what the Windows one always did."""

    def test_a_label_config_sh_would_split_is_refused_on_linux_too(self) -> None:
        raw = _raw(LINUX_UNIT_SUFFIX)
        labels: list[JSONValue] = ["lavender,gpu"]
        raw["labels"] = labels
        raw["workdir"] = "/home/gharunner/actions-runner-mcps-2/_work"
        with pytest.raises(JSONTypeError) as refused:
            decode_runner_install(raw)
        assert str(refused.value) == (
            "a label must be non-empty and hold no comma, got 'lavender,gpu'"
        )
