"""The roster's Windows runner hosts: what the decoder admits and why it refuses the rest."""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError, JSONValue

from fleet.contracts.runner_windows import (
    WindowsRunnerInstall,
    decode_windows_runner_host,
    decode_windows_runner_install,
    encode_windows_runner_host,
)
from tests._runner_fixtures import raw_windows_host, raw_windows_install


class TestTheInstall:
    def test_a_declared_install_decodes_field_for_field(self) -> None:
        assert decode_windows_runner_install(raw_windows_install()) == WindowsRunnerInstall(
            repo="wagner-austin/chat",
            runner_name="loki",
            service="actions.runner.wagner-austin-chat.loki",
            root="C:/actions-runner/chat",
            labels=["loki"],
        )

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError) as refused:
            decode_windows_runner_install("loki")
        assert str(refused.value) == "windows runner install must be a JSON object, got str"

    @pytest.mark.parametrize(
        ("key", "value", "message"),
        [
            ("repo", "chat", "repo must be 'owner/name', got 'chat'"),
            ("repo", "wagner-austin/", "repo must be 'owner/name', got 'wagner-austin/'"),
            ("runner_name", "", "runner_name must be non-empty"),
            (
                "service",
                "actions.runner.chat",
                "service must be 'actions.runner.wagner-austin-chat.loki', the name config.cmd "
                "gives wagner-austin/chat's runner loki; got 'actions.runner.chat'",
            ),
            (
                "root",
                "C:\\actions-runner\\chat",
                "root must be an absolute Windows path with forward slashes, like "
                "C:/actions-runner/chat; got 'C:\\\\actions-runner\\\\chat'",
            ),
            (
                "root",
                "/opt/actions-runner",
                "root must be an absolute Windows path with forward slashes, like "
                "C:/actions-runner/chat; got '/opt/actions-runner'",
            ),
        ],
    )
    def test_a_field_a_render_cannot_carry_is_refused(
        self, key: str, value: str, message: str
    ) -> None:
        raw = raw_windows_install()
        raw[key] = value
        with pytest.raises(JSONTypeError) as refused:
            decode_windows_runner_install(raw)
        assert str(refused.value) == message

    def test_an_install_with_no_label_is_refused(self) -> None:
        raw = raw_windows_install()
        raw["labels"] = []
        with pytest.raises(JSONTypeError, match="labels must be non-empty"):
            decode_windows_runner_install(raw)

    @pytest.mark.parametrize("label", ["loki,win", ""])
    def test_a_label_config_cmd_would_split_or_drop_is_refused(self, label: str) -> None:
        raw = raw_windows_install()
        labels: list[JSONValue] = [label]
        raw["labels"] = labels
        with pytest.raises(JSONTypeError) as refused:
            decode_windows_runner_install(raw)
        assert str(refused.value) == f"a label must be non-empty and hold no comma, got {label!r}"


class TestTheHost:
    def test_a_declared_host_round_trips(self) -> None:
        host = decode_windows_runner_host(raw_windows_host())
        assert host["payload_dir"] == "C:/fleet/runner-payloads"
        assert host["base"]["execution_policy"] == "RemoteSigned"
        assert decode_windows_runner_host(dict(encode_windows_runner_host(host))) == host

    def test_a_host_with_no_base_is_refused(self) -> None:
        raw = raw_windows_host()
        del raw["base"]
        with pytest.raises(JSONTypeError) as refused:
            decode_windows_runner_host(raw)
        assert str(refused.value) == "windows host base must be a JSON object, got NoneType"

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError) as refused:
            decode_windows_runner_host([])
        assert str(refused.value) == "windows runner host must be a JSON object, got list"

    def test_a_host_with_no_install_is_refused(self) -> None:
        raw = raw_windows_host()
        raw["installs"] = []
        with pytest.raises(JSONTypeError) as refused:
            decode_windows_runner_host(raw)
        assert str(refused.value) == "a windows runner host must declare at least one install"

    def test_two_installs_in_one_directory_are_refused(self) -> None:
        raw = raw_windows_host()
        other = raw_windows_install("wagner-austin/MCPs")
        other["root"] = "C:/actions-runner/chat"
        installs: list[JSONValue] = [raw_windows_install(), other]
        raw["installs"] = installs
        with pytest.raises(JSONTypeError) as refused:
            decode_windows_runner_host(raw)
        assert str(refused.value) == "two installs share a root: C:/actions-runner/chat"

    def test_one_runner_declared_twice_shares_its_service_and_is_refused(self) -> None:
        raw = raw_windows_host()
        installs: list[JSONValue] = [raw_windows_install(), raw_windows_install()]
        raw["installs"] = installs
        with pytest.raises(JSONTypeError) as refused:
            decode_windows_runner_host(raw)
        assert str(refused.value) == (
            "two installs share a service: actions.runner.wagner-austin-chat.loki"
        )

    def test_a_payload_directory_that_is_not_a_windows_path_is_refused(self) -> None:
        raw = raw_windows_host()
        raw["payload_dir"] = "/home/corvis/fleet"
        with pytest.raises(JSONTypeError, match="payload_dir must be an absolute Windows path"):
            decode_windows_runner_host(raw)
