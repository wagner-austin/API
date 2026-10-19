"""One machine of the roster, and the dispatch capacity it may declare (MCPs board task e447de72).

Every decoder is exercised on its own refusals rather than only its happy
path, because each refusal encodes a specific way a fleet goes wrong and a
decoder whose refusals are untested is a decoder that will be relaxed by
somebody who does not know why the check is there.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    dump_json_str,
    load_json_str,
)

from fleet.contracts.machine import MachineRole, decode_machine, encode_machine
from fleet.contracts.node import (
    NodeConfig,
    NodePlatform,
    decode_node_config,
    decode_node_gpu,
    encode_node_config,
    encode_node_gpu,
)
from tests._contract_fixtures import GPU, machine, node


def _decode(value: JSONValue) -> NodeConfig:
    """Decode a ``dispatch`` object as lavender's, a Windows machine of its own that answers.

    Args:
        value: The object.

    Returns:
        The node.
    """
    return decode_node_config(
        value, host="lavender", platform=NodePlatform.WINDOWS, enabled=True, wsl_host=None
    )


def _file(**changes: JSONValue) -> JSONObject:
    """lavender-wsl's machine file as this package reads it, with fields changed.

    Args:
        **changes: Fields to replace.

    Returns:
        The object.
    """
    base: JSONObject = {
        "role": "worker",
        "user": "corvis",
        "platform": "linux",
        "enabled": True,
        "guest": {"host": "lavender", "sshPort": 2222},
        "dispatch": {**encode_node_config(node()), "held": None},
        "tailnetIp": "100.65.161.27",
        "notes": "read by MCPs' decoder, carried here untouched",
    }
    return {**base, **changes}


def _guest_node(*, enabled: bool) -> NodeConfig:
    """The node :func:`_file` declares, with its machine's facts.

    Args:
        enabled: Whether it dispatches.

    Returns:
        lavender-wsl, a linux node inside lavender.
    """
    declared = node(host="lavender-wsl")
    declared["platform"] = NodePlatform.LINUX
    declared["enabled"] = enabled
    declared["wsl_host"] = "lavender"
    return declared


class TestNodeGpu:
    def test_a_device_survives_encoding(self) -> None:
        assert decode_node_gpu(load_json_str(dump_json_str(encode_node_gpu(GPU)))) == GPU

    def test_compute_capability_orders_as_a_version_not_a_float(self) -> None:
        """THE REASON IT IS A STRING, asserted on real architectures.

        Read as floats, 8.10 would sort BELOW 8.9 and a fleet comparing two
        cards would call the newer one older. Held as versions, the two are
        simply distinct values and nothing is tempted to order them
        arithmetically.
        """
        ampere = decode_node_gpu({**encode_node_gpu(GPU), "compute_capability": "8.9"})
        hopper = decode_node_gpu({**encode_node_gpu(GPU), "compute_capability": "8.10"})

        assert ampere["compute_capability"] != hopper["compute_capability"]
        assert float(hopper["compute_capability"]) < float(ampere["compute_capability"])

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            decode_node_gpu("GTX 1630")

    def test_a_card_reporting_no_memory_is_a_failed_probe(self) -> None:
        with pytest.raises(JSONTypeError, match="vram_mib must be positive"):
            decode_node_gpu({**encode_node_gpu(GPU), "vram_mib": 0})


class TestNodeConfig:
    def test_a_node_survives_encoding(self) -> None:
        original = node()

        assert _decode(load_json_str(dump_json_str(encode_node_config(original)))) == original

    def test_a_cpu_only_node_survives_encoding(self) -> None:
        original = node(gpu=None)

        assert _decode(load_json_str(dump_json_str(encode_node_config(original)))) == original

    def test_the_machines_own_facts_are_the_ones_handed_in(self) -> None:
        """``host``, ``platform``, ``enabled`` and ``wsl_host`` are said once,
        in the machine's file, so the capacity carries none of them and the
        decoder takes them as given."""
        encoded = encode_node_config(node())
        assert not {"host", "platform", "enabled", "wsl_host"} & set(encoded)

        decoded = decode_node_config(
            encoded,
            host="lavender-wsl",
            platform=NodePlatform.LINUX,
            enabled=False,
            wsl_host="lavender",
        )

        assert (
            decoded["host"],
            decoded["platform"],
            decoded["enabled"],
            decoded["wsl_host"],
        ) == ("lavender-wsl", NodePlatform.LINUX, False, "lavender")

    def test_a_windows_node_inside_another_machine_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="only a linux node runs inside another"):
            decode_node_config(
                encode_node_config(node()),
                host="lavender",
                platform=NodePlatform.WINDOWS,
                enabled=True,
                wsl_host="sedona",
            )

    def test_an_absent_gpu_key_is_refused(self) -> None:
        """Absence is indistinguishable from unfilled.

        The difference matters the moment a measurement is pinned to a card,
        so a CPU-only node says so with an explicit null.
        """
        encoded = encode_node_config(node())
        del encoded["gpu"]

        with pytest.raises(JSONTypeError, match="must declare 'gpu'"):
            _decode(encoded)

    def test_an_absent_test_database_key_is_refused(self) -> None:
        """Neither default is safe here either (MCPs board task 6bbfd171).

        False would hide a provisioned node from every Postgres-backed
        package; true would hand those packages to a node whose global test
        setup cannot reach a database.
        """
        encoded = encode_node_config(node())
        del encoded["test_database"]

        with pytest.raises(JSONTypeError, match="must declare 'test_database'"):
            _decode(encoded)

    def test_test_database_is_carried_both_ways(self) -> None:
        encoded = encode_node_config(node())
        assert encoded["test_database"] is False

        assert _decode({**encoded, "test_database": True})["test_database"] is True

    def test_a_non_object_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="must be a JSON object"):
            _decode(["lavender"])

    def test_a_node_with_no_cores_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="logical_cores must be at least 1"):
            _decode({**encode_node_config(node()), "logical_cores": 0})

    def test_an_empty_stage_root_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="stage_root must not be empty"):
            _decode({**encode_node_config(node()), "stage_root": ""})

    def test_a_node_with_no_memory_is_refused(self) -> None:
        with pytest.raises(JSONTypeError, match="ram_gb must be positive"):
            _decode({**encode_node_config(node()), "ram_gb": 0.0})


class TestMachine:
    def test_a_guest_node_takes_its_name_platform_and_host_from_the_machine(self) -> None:
        decoded = decode_machine("lavender-wsl", _file())

        assert (decoded["role"], decoded["user"], decoded["guest_host"], decoded["held"]) == (
            MachineRole.WORKER,
            "corvis",
            "lavender",
            None,
        )
        assert decoded["dispatch"] == _guest_node(enabled=True)

    def test_a_machine_that_declares_no_capacity_carries_its_reason(self) -> None:
        decoded = decode_machine(
            "phone", _file(role="client", user=None, guest=None, dispatch="never dispatched to")
        )

        assert (decoded["dispatch"], decoded["held"], decoded["user"]) == (
            None,
            "never dispatched to",
            None,
        )

    def test_a_held_node_dispatches_nothing_and_says_why(self) -> None:
        dispatch = {**encode_node_config(node()), "held": "its WSL VM holds the memory"}

        held = _file(platform="windows", guest=None, dispatch=dispatch)

        decoded = decode_machine("lavender", held)

        assert decoded["held"] == "its WSL VM holds the memory"
        assert decoded["dispatch"] == {**node(), "enabled": False}

    def test_a_machine_that_is_off_does_not_dispatch(self) -> None:
        decoded = decode_machine("lavender-wsl", _file(enabled=False))

        assert (decoded["dispatch"], decoded["held"]) == (_guest_node(enabled=False), None)

    def test_a_machine_survives_encoding_in_what_this_package_reads(self) -> None:
        original = decode_machine("lavender-wsl", _file())

        assert decode_machine("lavender-wsl", encode_machine(original)) == original
        held = machine("austinpc", None, held="the hub")
        assert decode_machine("austinpc", encode_machine(held)) == held

    @pytest.mark.parametrize(
        ("value", "message"),
        [
            ("lavender", "machine 'm' must be a JSON object, got str"),
            (_file(role="desktop"), "machine 'm' role must be one of hub, vpn-jump, worker"),
            (_file(platform="solaris"), "platform must be one of windows, linux"),
            (_file(guest="lavender"), "machine 'm' guest must be an object or null"),
            (_file(guest={"sshPort": 2222}), "machine 'm' guest.host must be a non-empty string"),
            (_file(dispatch=7), "machine 'm' must declare 'dispatch': the reason it takes"),
            (_file(dispatch="  "), "machine 'm' dispatch must be a non-empty string"),
            (
                _file(dispatch=encode_node_config(node())),
                "machine 'm' dispatch must declare 'held'",
            ),
            (
                _file(dispatch={**encode_node_config(node()), "held": ""}),
                "machine 'm' dispatch.held must be a non-empty string",
            ),
        ],
    )
    def test_a_malformed_machine_is_refused_by_field(self, value: JSONValue, message: str) -> None:
        with pytest.raises(JSONTypeError, match=message):
            decode_machine("m", value)

    def test_an_absent_guest_is_refused(self) -> None:
        """Null says the machine is on the tailnet in its own right; absent
        says nobody wrote it down, which a WSL guest's file once did."""
        value = _file()
        del value["guest"]

        with pytest.raises(JSONTypeError, match="machine 'm' must declare 'guest'"):
            decode_machine("m", value)

    def test_a_reason_on_a_machine_that_is_off_is_refused(self) -> None:
        """A machine that is off claims nothing already; a held reason on it
        would be a second statement of one fact."""
        dispatch = {**encode_node_config(node()), "held": "travels"}

        with pytest.raises(JSONTypeError, match="is not enabled and its dispatch is held"):
            decode_machine("m", _file(enabled=False, dispatch=dispatch))
