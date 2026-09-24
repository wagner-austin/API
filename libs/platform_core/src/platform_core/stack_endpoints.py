"""Where the MCPs stack's services are reached from this machine.

ONE DECLARATION, READ, NOT COPIED. The MCPs repository writes every
service's address down once, in ``scripts/fleet/stack-endpoints.json``, and
its session hooks and upgrade gate read it there. Tools in this monorepo
that call the stack used to carry their own literal instead,
``http://127.0.0.1:8033/mcp``, which was right while the hub forwarded each
moved service's port to diphtheria. The hub stopped on 2026-09-24 (MCPs
board task c6fc4882), and the literal failed every ``pre-push`` enrolment
and every wake-bridge cycle at once; a second copy of an address is a
second place for it to go stale, so this module reads the first one.

The caller passes the file's TEXT, read through its own file seam, the
way :mod:`platform_core.session_label` already takes the stack's ``.env``:
the seams stay in the packages that own them.
"""

from __future__ import annotations

import pathlib
from typing import Final

from platform_core.error_codes_tooling import StackEndpointErrorCode
from platform_core.errors import AppError
from platform_core.json_utils import load_json_str, narrow_json_to_dict

#: The declaration, in the MCPs checkout beside this one.
STACK_ENDPOINTS_PATH: Final = (
    pathlib.Path.home() / "PROJECTS" / "MCPs" / "scripts" / "fleet" / "stack-endpoints.json"
)

#: The declaration's name for the taskboard.
TASKBOARD_SERVICE: Final = "taskboard-mcp"


def declared_url(endpoints_text: str, service: str) -> str:
    """The url the stack's endpoint declaration gives a service.

    Args:
        endpoints_text: The whole of :data:`STACK_ENDPOINTS_PATH`.
        service: The service's name there, e.g. :data:`TASKBOARD_SERVICE`.

    Returns:
        Its ``services.<service>.url``.

    Raises:
        InvalidJsonError: When the text is not JSON.
        JSONTypeError: When the document is not an object.
        AppError: ``STACK_ENDPOINT_UNDECLARED`` when the declaration carries
            no non-empty url for the service, naming the service and the
            file.
    """
    document = narrow_json_to_dict(load_json_str(endpoints_text))
    services = document.get("services")
    entry = services.get(service) if isinstance(services, dict) else None
    url = entry.get("url") if isinstance(entry, dict) else None
    if not isinstance(url, str) or url == "":
        raise AppError(
            code=StackEndpointErrorCode.UNDECLARED,
            message=f"{STACK_ENDPOINTS_PATH} declares no url for {service}",
        )
    return url


__all__ = [
    "STACK_ENDPOINTS_PATH",
    "TASKBOARD_SERVICE",
    "declared_url",
]
