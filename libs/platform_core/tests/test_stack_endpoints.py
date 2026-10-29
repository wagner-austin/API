"""Forming a service's url from the MCPs stack's declaration, its placement
and this repository's roster (MCPs board tasks c6fc4882 and 50381dd9)."""

from __future__ import annotations

import pathlib

import pytest

from platform_core.error_codes_tooling import StackEndpointErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import InvalidJsonError, JSONTypeError, JSONValue, dump_json_str
from platform_core.mcp_testing import (
    DECLARED_FLEET_URL,
    DECLARED_STACK_HOST,
    DECLARED_TASKBOARD_URL,
    stack_endpoint_files,
)
from platform_core.stack_endpoints import (
    FLEET_SERVICE,
    ROSTER_MACHINES_PATH,
    STACK_ENDPOINTS_PATH,
    STACK_PLACEMENT_PATH,
    TASKBOARD_SERVICE,
    declared_url,
)

#: The roster file :func:`stack_endpoint_files` writes for the stack's host.
MACHINE = ROSTER_MACHINES_PATH / f"{DECLARED_STACK_HOST}.json"


def url(files: dict[pathlib.Path, str], service: str) -> str:
    """:func:`declared_url` over files held in memory."""
    return declared_url(lambda path: files[path], service)


def refusal(files: dict[pathlib.Path, str]) -> AppError[StackEndpointErrorCode]:
    """The error :func:`declared_url` raises for the taskboard."""
    with pytest.raises(AppError) as caught:
        url(files, TASKBOARD_SERVICE)
    return caught.value


def test_each_url_is_the_placed_hosts_address_at_the_declared_port() -> None:
    files = stack_endpoint_files()
    assert url(files, TASKBOARD_SERVICE) == DECLARED_TASKBOARD_URL
    assert url(files, FLEET_SERVICE) == DECLARED_FLEET_URL


def test_moving_a_service_moves_its_url_to_the_new_hosts_address() -> None:
    files = stack_endpoint_files()
    files[STACK_PLACEMENT_PATH] = dump_json_str(
        {"services": {TASKBOARD_SERVICE: [DECLARED_STACK_HOST], FLEET_SERVICE: ["elsewhere"]}}
    )
    files[ROSTER_MACHINES_PATH / "elsewhere.json"] = dump_json_str({"tailnetIp": "100.64.0.8"})
    assert url(files, FLEET_SERVICE) == "http://100.64.0.8:8035/mcp"
    assert url(files, TASKBOARD_SERVICE) == DECLARED_TASKBOARD_URL


@pytest.mark.parametrize(
    "declaration",
    [
        {},
        {"services": []},
        {"services": {}},
        {"services": {TASKBOARD_SERVICE: "http://x"}},
        {"services": {TASKBOARD_SERVICE: {"url": "http://100.64.0.3:8033/mcp"}}},
        {"services": {TASKBOARD_SERVICE: {"port": "8033"}}},
        {"services": {TASKBOARD_SERVICE: {"port": True}}},
    ],
)
def test_a_declaration_without_the_services_port_is_refused_naming_it_and_the_file(
    declaration: JSONValue,
) -> None:
    files = stack_endpoint_files()
    files[STACK_ENDPOINTS_PATH] = dump_json_str(declaration)
    caught = refusal(files)
    assert caught.code is StackEndpointErrorCode.UNDECLARED
    assert caught.message == f"{STACK_ENDPOINTS_PATH} declares no port for {TASKBOARD_SERVICE}"


@pytest.mark.parametrize(
    "hosts",
    [None, DECLARED_STACK_HOST, [], [DECLARED_STACK_HOST, "elsewhere"], [7]],
)
def test_a_service_on_no_one_host_is_refused_naming_the_placement(hosts: JSONValue) -> None:
    files = stack_endpoint_files()
    services: JSONValue = {} if hosts is None else {TASKBOARD_SERVICE: hosts}
    files[STACK_PLACEMENT_PATH] = dump_json_str({"services": services})
    caught = refusal(files)
    assert caught.code is StackEndpointErrorCode.UNPLACED
    assert caught.message == f"{STACK_PLACEMENT_PATH} places {TASKBOARD_SERVICE} on no one host"


@pytest.mark.parametrize("machine", [{}, {"tailnetIp": ""}, {"tailnetIp": 7}])
def test_a_host_the_roster_gives_no_address_is_refused_naming_its_file(
    machine: JSONValue,
) -> None:
    files = stack_endpoint_files()
    files[MACHINE] = dump_json_str(machine)
    caught = refusal(files)
    assert caught.code is StackEndpointErrorCode.UNADDRESSED
    assert caught.message == (
        f"{MACHINE} names no tailnetIp for {DECLARED_STACK_HOST}, where {TASKBOARD_SERVICE} runs"
    )


def test_a_declaration_that_is_not_json_raises_the_parse_error() -> None:
    files = stack_endpoint_files()
    files[STACK_ENDPOINTS_PATH] = "not json"
    with pytest.raises(InvalidJsonError):
        url(files, TASKBOARD_SERVICE)


def test_a_declaration_that_is_not_an_object_raises_the_type_error() -> None:
    files = stack_endpoint_files()
    files[STACK_ENDPOINTS_PATH] = "[]"
    with pytest.raises(JSONTypeError):
        url(files, TASKBOARD_SERVICE)
