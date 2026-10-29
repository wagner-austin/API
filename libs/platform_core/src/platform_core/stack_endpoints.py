"""Where the MCPs stack's services are reached from this machine.

ONE DECLARATION, READ, NOT COPIED. The MCPs repository declares each
service a caller dials past its proxy, and the port that service is
published at, once, in ``scripts/fleet/stack-endpoints.json``. Tools in
this monorepo that call the stack used to carry their own literal instead,
``http://127.0.0.1:8033/mcp``, which was right while the hub forwarded each
moved service's port to diphtheria. The hub stopped on 2026-09-24 (MCPs
board task c6fc4882), and the literal failed every ``pre-push`` enrolment
and every wake-bridge cycle at once; a second copy of an address is a
second place for it to go stale, so this module reads the first one.

THE ADDRESS IS FORMED, NEVER WRITTEN (MCPs board task 50381dd9). Every
stack host publishes its services on its own roster ``tailnetIp``, so a
service's url is the declared port at the address of the one host the MCPs
``scripts/fleet/placement.json`` puts it on, read from this repository's
fleet roster, ``tools/fleet/registry/machines/<host>.json``. The MCPs
readers form it the same way (MCPs ``packages/maketools`` ``stack_endpoints``).

The caller passes its own file seam, which this module reads all three
files through, the way :mod:`platform_core.session_label` already reads the
stack's ``.env``: the seams stay in the packages that own them.
"""

from __future__ import annotations

import pathlib
from collections.abc import Callable
from typing import Final

from platform_core.error_codes_tooling import StackEndpointErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import JSONValue, load_json_str, narrow_json_to_dict

#: The MCPs checkout beside this one, where the stack's declarations live.
_MCPS_FLEET: Final = pathlib.Path.home() / "PROJECTS" / "MCPs" / "scripts" / "fleet"

#: The declaration of each service's published port.
STACK_ENDPOINTS_PATH: Final = _MCPS_FLEET / "stack-endpoints.json"

#: Where each MCPs service runs.
STACK_PLACEMENT_PATH: Final = _MCPS_FLEET / "placement.json"

#: This repository's fleet roster, one file per machine.
ROSTER_MACHINES_PATH: Final = (
    pathlib.Path.home() / "PROJECTS" / "API" / "tools" / "fleet" / "registry" / "machines"
)

#: The declaration's name for the taskboard.
TASKBOARD_SERVICE: Final = "taskboard-mcp"

#: The declaration's name for fleet-mcp, whose dispatch queue the fleet
#: agent's ticks claim from.
FLEET_SERVICE: Final = "fleet-mcp"


def _field(text: str, *path: str) -> JSONValue | None:
    """One value nested in a JSON object, or None where any step is absent.

    Raises:
        InvalidJsonError: When the text is not JSON.
        JSONTypeError: When the document is not an object.
    """
    value: JSONValue | None = narrow_json_to_dict(load_json_str(text))
    for key in path:
        value = value.get(key) if isinstance(value, dict) else None
    return value


def declared_url(read_text: Callable[[pathlib.Path], str], service: str) -> str:
    """The url a service is reached at: its port at its placed host's address.

    Args:
        read_text: Reads a UTF-8 file whole.
        service: The service's name, e.g. :data:`TASKBOARD_SERVICE`.

    Returns:
        ``http://<tailnetIp>:<port>/mcp``.

    Raises:
        InvalidJsonError: When a file is not JSON.
        JSONTypeError: When a file is not an object.
        AppError: ``STACK_ENDPOINT_UNDECLARED`` when the declaration gives
            the service no integer port, ``STACK_ENDPOINT_UNPLACED`` when the
            placement puts it on no one host, and
            ``STACK_ENDPOINT_UNADDRESSED`` when that host's roster file names
            no tailnetIp, each naming the service and the file.
    """
    port = _field(read_text(STACK_ENDPOINTS_PATH), "services", service, "port")
    if not isinstance(port, int) or isinstance(port, bool):
        raise AppError(
            code=StackEndpointErrorCode.UNDECLARED,
            message=f"{STACK_ENDPOINTS_PATH} declares no port for {service}",
        )
    hosts = _field(read_text(STACK_PLACEMENT_PATH), "services", service)
    if not isinstance(hosts, list) or len(hosts) != 1 or not isinstance(hosts[0], str):
        raise AppError(
            code=StackEndpointErrorCode.UNPLACED,
            message=f"{STACK_PLACEMENT_PATH} places {service} on no one host",
        )
    machine = ROSTER_MACHINES_PATH / f"{hosts[0]}.json"
    address = _field(read_text(machine), "tailnetIp")
    if not isinstance(address, str) or address == "":
        raise AppError(
            code=StackEndpointErrorCode.UNADDRESSED,
            message=f"{machine} names no tailnetIp for {hosts[0]}, where {service} runs",
        )
    return f"http://{address}:{port}/mcp"


__all__ = [
    "FLEET_SERVICE",
    "ROSTER_MACHINES_PATH",
    "STACK_ENDPOINTS_PATH",
    "STACK_PLACEMENT_PATH",
    "TASKBOARD_SERVICE",
    "declared_url",
]
