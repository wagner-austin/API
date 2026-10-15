"""A project's ``node``, and what the workspace refuses of it (MCPs board task bf7a9460).

MCPs' served lanes each name the search host whose own container they read,
so the key must survive encoding, be absent for every other project, and
name a node the workspace declares under its own host.
"""

from __future__ import annotations

import pytest
from platform_core.json_utils import JSONTypeError, dump_json_str, load_json_str

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


def test_a_pin_to_a_node_whose_host_is_another_name_is_refused() -> None:
    """capacity.assess matches the pin against the weighing node's host."""
    encoded = encode_fleet_workspace(_workspace())
    pinned = {**encode_project_config(_project()), "node": "loki"}

    with pytest.raises(JSONTypeError, match="pinned to node 'loki', whose host is 'lavender'"):
        decode_fleet_workspace({**encoded, "projects": {"MCPs/served": pinned}})
