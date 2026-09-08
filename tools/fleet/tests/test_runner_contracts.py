"""The runner roster's contracts: every decode refusal, and the round trip.

The shipped ``runners.json`` is decoded here through the same functions
production uses, so the document in the repo cannot drift from the contract
that reads it -- the same discipline as every other document this package
owns.
"""

from __future__ import annotations

import pathlib

import pytest
from platform_core.json_utils import JSONTypeError, JSONValue, load_json_str

from fleet.contracts.runners import (
    FileAsset,
    HostRunnerSpec,
    RunnerInstall,
    decode_file_asset,
    decode_host_runner_spec,
    decode_runner_install,
    decode_runner_spec,
    encode_runner_spec,
)

#: The shipped roster, resolved from this file so the test runs from any cwd.
SHIPPED_ROSTER = pathlib.Path(__file__).resolve().parent.parent / "runners.json"


def _install(**overrides: JSONValue) -> dict[str, JSONValue]:
    """A valid raw install, with fields overridden per test.

    Args:
        overrides: Field values replacing the valid defaults.

    Returns:
        The raw mapping.
    """
    raw: dict[str, JSONValue] = {
        "repo": "wagner-austin/API",
        "runner_name": "lavender-wsl",
        "service": "actions.runner.wagner-austin-API.lavender-wsl.service",
        "workdir": "/home/gharunner/actions-runner-api-1/_work",
        "labels": ["lavender-wsl"],
    }
    raw.update(overrides)
    return raw


def _asset(**overrides: JSONValue) -> dict[str, JSONValue]:
    """A valid raw asset, with fields overridden per test.

    Args:
        overrides: Field values replacing the valid defaults.

    Returns:
        The raw mapping.
    """
    raw: dict[str, JSONValue] = {
        "path": "/data",
        "sha256": None,
        "writable": True,
        "reason": "checkpoint tests write under it",
        "manual": False,
        "provision_command": None,
    }
    raw.update(overrides)
    return raw


def _host(**overrides: JSONValue) -> dict[str, JSONValue]:
    """A valid raw host, with fields overridden per test.

    Args:
        overrides: Field values replacing the valid defaults.

    Returns:
        The raw mapping.
    """
    raw: dict[str, JSONValue] = {
        "name": "lavender",
        "host": "lavender",
        "wsl_distro": "Ubuntu",
        "keepalive_task": "wsl-keepalive",
        "wslconfig_min_memory_gb": 26,
        "scratch_dir": "C:/fleet/stage",
        "gpu_required": True,
        "systemd_timers": ["ci-clean.timer"],
        "installs": [_install()],
        "assets": [_asset()],
    }
    raw.update(overrides)
    return raw


class TestRunnerInstall:
    """decode_runner_install's contract."""

    def test_a_valid_install_round_trips(self) -> None:
        decoded: RunnerInstall = decode_runner_install(_install())
        assert decoded["repo"] == "wagner-austin/API"
        assert decoded["labels"] == ["lavender-wsl"]

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            decode_runner_install("not an object")

    @pytest.mark.parametrize("repo", ["no-slash", "a/b/c", "/name", "owner/"])
    def test_a_repo_that_is_not_owner_name_is_refused(self, repo: str) -> None:
        with pytest.raises(JSONTypeError, match="owner/name"):
            decode_runner_install(_install(repo=repo))

    def test_empty_labels_are_refused_because_nothing_could_target_the_runner(self) -> None:
        with pytest.raises(JSONTypeError, match="unreachable"):
            decode_runner_install(_install(labels=[]))


class TestFileAsset:
    """decode_file_asset's contract."""

    def test_a_valid_writable_asset_round_trips(self) -> None:
        decoded: FileAsset = decode_file_asset(_asset())
        assert decoded["writable"] is True
        assert decoded["sha256"] is None

    def test_a_valid_pinned_asset_keeps_its_pin(self) -> None:
        pin = "8a550a37e2d8a5430866090d4e7d5892f9010b47f52a5a09350fc66c620deec9"
        decoded = decode_file_asset(
            _asset(sha256=pin, writable=False, manual=True, provision_command=None)
        )
        assert decoded["sha256"] == pin

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            decode_file_asset(17)

    def test_an_absent_sha256_key_is_refused(self) -> None:
        raw = _asset()
        del raw["sha256"]
        with pytest.raises(JSONTypeError, match="must declare 'sha256'"):
            decode_file_asset(raw)

    @pytest.mark.parametrize("pin", ["abc", "G" * 64, "A" * 64])
    def test_a_malformed_pin_is_refused(self, pin: str) -> None:
        with pytest.raises(JSONTypeError, match="64 lowercase hex"):
            decode_file_asset(_asset(sha256=pin))

    def test_a_non_string_pin_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="string or null"):
            decode_file_asset(_asset(sha256=64))

    def test_a_provision_command_on_a_manual_asset_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be null on a manual or writable"):
            decode_file_asset(
                _asset(manual=True, writable=False, provision_command="curl something")
            )

    def test_a_provision_command_on_a_writable_asset_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be null on a manual or writable"):
            decode_file_asset(_asset(provision_command="mkdir -p /data"))

    def test_a_fetchable_asset_without_a_command_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must carry provision_command"):
            decode_file_asset(_asset(writable=False, provision_command=None))

    def test_a_fetchable_asset_with_a_command_is_accepted(self) -> None:
        decoded = decode_file_asset(_asset(writable=False, provision_command="git clone x /opt/x"))
        assert decoded["provision_command"] == "git clone x /opt/x"

    def test_an_absent_provision_command_key_is_refused(self) -> None:
        raw = _asset()
        del raw["provision_command"]
        with pytest.raises(JSONTypeError, match="must declare 'provision_command'"):
            decode_file_asset(raw)

    def test_a_mistyped_provision_command_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="string or null"):
            decode_file_asset(_asset(provision_command=3))


class TestHostRunnerSpec:
    """decode_host_runner_spec's contract."""

    def test_a_valid_host_round_trips(self) -> None:
        decoded: HostRunnerSpec = decode_host_runner_spec(_host())
        assert decoded["name"] == "lavender"
        assert decoded["wslconfig_min_memory_gb"] == 26

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            decode_host_runner_spec([])

    def test_a_null_keepalive_task_is_a_state_not_an_omission(self) -> None:
        decoded = decode_host_runner_spec(_host(keepalive_task=None))
        assert decoded["keepalive_task"] is None

    def test_an_absent_keepalive_key_is_refused(self) -> None:
        raw = _host()
        del raw["keepalive_task"]
        with pytest.raises(JSONTypeError, match="must declare 'keepalive_task'"):
            decode_host_runner_spec(raw)

    def test_a_mistyped_keepalive_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="string or null"):
            decode_host_runner_spec(_host(keepalive_task=7))

    def test_an_absent_memory_floor_key_is_refused(self) -> None:
        raw = _host()
        del raw["wslconfig_min_memory_gb"]
        with pytest.raises(JSONTypeError, match="must declare 'wslconfig_min_memory_gb'"):
            decode_host_runner_spec(raw)

    def test_a_null_memory_floor_is_the_host_default(self) -> None:
        decoded = decode_host_runner_spec(_host(wslconfig_min_memory_gb=None))
        assert decoded["wslconfig_min_memory_gb"] is None

    @pytest.mark.parametrize("floor", [0, -4])
    def test_a_non_positive_memory_floor_is_refused(self, floor: int) -> None:
        with pytest.raises(JSONTypeError, match="must be positive"):
            decode_host_runner_spec(_host(wslconfig_min_memory_gb=floor))

    @pytest.mark.parametrize("floor", [True, "26", 2.5])
    def test_a_mistyped_memory_floor_is_refused(self, floor: JSONValue) -> None:
        with pytest.raises(JSONTypeError, match="integer or null"):
            decode_host_runner_spec(_host(wslconfig_min_memory_gb=floor))

    def test_a_host_with_no_installs_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="not a CI host"):
            decode_host_runner_spec(_host(installs=[]))


class TestRunnerSpec:
    """decode_runner_spec's contract, and the shipped document."""

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            decode_runner_spec(None)

    def test_an_empty_roster_is_refused_because_it_reads_as_healthy(self) -> None:
        with pytest.raises(JSONTypeError, match="reads exactly like a healthy fleet"):
            decode_runner_spec({"hosts": []})

    def test_duplicate_host_names_are_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be unique"):
            decode_runner_spec({"hosts": [_host(), _host()]})

    def test_encode_then_decode_is_the_identity(self) -> None:
        spec = decode_runner_spec({"hosts": [_host()]})
        assert decode_runner_spec(dict(encode_runner_spec(spec))) == spec

    def test_the_shipped_roster_decodes_and_round_trips(self) -> None:
        """The runners.json in the repo satisfies its own contract.

        This is the integration claim of the module: the document operators
        edit is validated by the exact decoder the CLI runs, so a bad edit
        fails here before it fails in the field.
        """
        spec = decode_runner_spec(load_json_str(SHIPPED_ROSTER.read_text(encoding="utf-8")))
        assert decode_runner_spec(dict(encode_runner_spec(spec))) == spec
        lavender = spec["hosts"][0]
        assert lavender["name"] == "lavender"
        assert len(lavender["installs"]) == 6
        pinned = [asset for asset in lavender["assets"] if asset["sha256"] is not None]
        assert len(pinned) == 1
        assert pinned[0]["manual"] is True
