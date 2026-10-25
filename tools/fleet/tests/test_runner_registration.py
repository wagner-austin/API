"""The four facts every runner install declares, held to one rule."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue

from fleet.contracts.runner_registration import (
    UNIT_SUFFIX,
    RunnerRegistration,
    decode_runner_registration,
    encode_runner_registration,
    runner_service_name,
)
from fleet.contracts.runners import decode_runner_install


def _raw(service: str = "actions.runner.wagner-austin-MCPs.lavender-2.service") -> JSONObject:
    """lavender's MCPs runner as the roster spells it.

    Args:
        service: Its unit's name.

    Returns:
        The raw registration.
    """
    labels: list[JSONValue] = ["lavender"]
    return {
        "repo": "wagner-austin/MCPs",
        "runner_name": "lavender-2",
        "service": service,
        "labels": labels,
    }


class TestTheServiceName:
    def test_the_configuration_s_name_is_a_systemd_service(self) -> None:
        assert UNIT_SUFFIX == ".service"
        assert runner_service_name("wagner-austin/MCPs", "lavender-2") == (
            "actions.runner.wagner-austin-MCPs.lavender-2.service"
        )


class TestTheDecoder:
    def test_a_registration_round_trips(self) -> None:
        decoded = decode_runner_registration(_raw())
        assert decoded == RunnerRegistration(
            repo="wagner-austin/MCPs",
            runner_name="lavender-2",
            service="actions.runner.wagner-austin-MCPs.lavender-2.service",
            labels=["lavender"],
        )
        assert encode_runner_registration(decoded) == _raw()

    def test_a_service_name_without_the_unit_suffix_is_refused(self) -> None:
        with pytest.raises(JSONTypeError) as refused:
            decode_runner_registration(_raw("actions.runner.wagner-austin-MCPs.lavender-2"))
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
        raw = _raw()
        raw[key] = value
        with pytest.raises(JSONTypeError) as refused:
            decode_runner_registration(raw)
        assert str(refused.value) == message


class TestTheInstall:
    def test_a_label_config_sh_would_split_is_refused(self) -> None:
        raw = _raw()
        labels: list[JSONValue] = ["lavender,gpu"]
        raw["labels"] = labels
        raw["workdir"] = "/home/gharunner/actions-runner-mcps-2/_work"
        with pytest.raises(JSONTypeError) as refused:
            decode_runner_install(raw)
        assert str(refused.value) == (
            "a label must be non-empty and hold no comma, got 'lavender,gpu'"
        )
