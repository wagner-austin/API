"""A project's ``node``, and what the workspace refuses of it (MCPs board tasks bf7a9460, f19d66a5).

MCPs' served lanes each name the search host whose own container they read,
so the key must survive encoding, be absent for every other project, and
name a node the workspace declares, whose host is that name. The hub suite
names the roster's one hub and nothing else.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
)

from fleet.contracts.pins import HUB_SUITE
from fleet.contracts.project import decode_project_config, encode_project_config
from fleet.contracts.workspace import decode_fleet_workspace, encode_fleet_workspace
from tests.test_contracts import _project, _workspace


def test_a_pinned_node_survives_encoding() -> None:
    encoded = {**encode_project_config(_project()), "node": "colossus"}

    decoded = decode_project_config(load_json_str(dump_json_str(encoded)))

    assert decoded["node"] == "colossus"
    assert encode_project_config(decoded)["node"] == "colossus"


def test_a_project_naming_no_node_runs_anywhere() -> None:
    """Absent is the ordinary case: every project but the served lanes."""
    encoded = encode_project_config(_project())
    del encoded["node"]

    assert decode_project_config(encoded)["node"] is None


@pytest.mark.parametrize("pin", ["", 7])
def test_a_pin_that_names_no_node_is_refused(pin: str | int) -> None:
    with pytest.raises(JSONTypeError, match=r"project\.node must be a node's registry name"):
        decode_project_config({**encode_project_config(_project()), "node": pin})


def test_a_project_pinned_to_a_declared_node_decodes() -> None:
    encoded = encode_fleet_workspace(_workspace())
    pinned = {**encode_project_config(_project()), "node": "lavender"}

    decoded = decode_fleet_workspace({**encoded, "projects": {"MCPs/served": pinned}})

    assert decoded["projects"]["MCPs/served"]["node"] == "lavender"


def test_a_project_pinned_to_an_undeclared_node_is_refused() -> None:
    """No runner would ever claim it: every node would refuse it."""
    encoded = encode_fleet_workspace(_workspace())
    pinned = {**encode_project_config(_project()), "node": "colossus"}

    with pytest.raises(JSONTypeError) as refused:
        decode_fleet_workspace({**encoded, "projects": {"MCPs/served": pinned}})

    assert str(refused.value) == (
        "project MCPs/served is pinned to node 'colossus', which the workspace does not "
        "declare (it declares lavender, loki); no runner would ever claim it"
    )


def test_a_pinned_node_s_host_is_the_name_it_is_pinned_by() -> None:
    """capacity.assess matches the pin against the weighing node's host,
    which is the machine file's name, so every declared pin matches."""
    encoded = encode_fleet_workspace(_workspace())
    pinned = {**encode_project_config(_project()), "node": "loki"}

    decoded = decode_fleet_workspace({**encoded, "projects": {"MCPs/served": pinned}})

    assert decoded["nodes"]["loki"]["host"] == "loki"


def _with_hub_suite(pin: str | None, hubs: tuple[str, ...]) -> JSONObject:
    """The fixture workspace with these machines as hubs and the hub suite pinned so.

    Args:
        pin: The hub suite's node, or None for an unpinned suite.
        hubs: The fixture machines (lavender, loki) whose role is hub.

    Returns:
        The encoded workspace.
    """
    encoded = encode_fleet_workspace(_workspace())
    machines = narrow_json_to_dict(encoded["machines"])
    roled: JSONObject = {
        name: {**narrow_json_to_dict(machine), "role": "hub" if name in hubs else "worker"}
        for name, machine in machines.items()
    }
    suite = {**encode_project_config(_project()), "node": pin}
    return {**encoded, "machines": roled, "projects": {HUB_SUITE: suite}}


def test_the_hub_suite_pinned_to_the_roster_s_hub_decodes() -> None:
    decoded = decode_fleet_workspace(_with_hub_suite("loki", ("loki",)))

    assert decoded["projects"][HUB_SUITE]["node"] == "loki"


@pytest.mark.parametrize(
    ("pin", "hubs", "named"),
    [
        ("lavender", ("loki",), "pinned to node 'lavender', and the roster's hub is loki"),
        (None, ("loki",), "pinned to node None, and the roster's hub is loki"),
        ("loki", (), "pinned to node 'loki', and the roster's hub is no machine"),
        ("loki", ("lavender", "loki"), "the roster's hub is lavender, loki;"),
    ],
)
def test_the_hub_suite_anywhere_but_the_one_hub_is_refused(
    pin: str | None, hubs: tuple[str, ...], named: str
) -> None:
    """On 2026-11-04 the suite still named diphtheria after the hub role had
    moved to lacrymal, and its cases ran where the hub's ssh is not."""
    with pytest.raises(JSONTypeError) as refused:
        decode_fleet_workspace(_with_hub_suite(pin, hubs))

    assert str(refused.value).startswith(f"project {HUB_SUITE} is ")
    assert named in str(refused.value)
    assert str(refused.value).endswith("so it is pinned to the one machine whose role is hub")
