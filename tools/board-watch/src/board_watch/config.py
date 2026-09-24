"""What the shell must be given before it can call the board.

Two secrets from the process environment and a URL from the stack's endpoint
declaration, overridable from the environment, all through the
:mod:`board_watch._test_hooks` seams so a test never touches ``os.environ``
or the real file.

The credentials are NOT discovered. An earlier prototype read them by
shelling out to ``docker inspect`` and ``psql``, which made the watcher
depend on the container runtime being present, on the caller having
permission to inspect containers, and on two more processes existing before
a single poll could happen. Requiring them in the environment moves that
work to the operator's shell once, where it is visible, instead of into
every poll.
"""

from __future__ import annotations

from typing import Final

from platform_core.error_codes_tooling import BoardWatchErrorCode
from platform_core.errors import AppError
from platform_core.mcp_client import McpCredentials
from platform_core.stack_endpoints import STACK_ENDPOINTS_PATH, TASKBOARD_SERVICE, declared_url

from board_watch import _test_hooks

#: Environment variable holding taskboard-mcp's ``x-api-key`` value.
API_KEY_VARIABLE: Final = "TASKBOARD_MCP_API_KEY"

#: Environment variable holding the tenant whose board is being read.
TENANT_ID_VARIABLE: Final = "CORVIS_TENANT_ID"

#: Environment variable overriding the endpoint, for a non-default deployment.
URL_VARIABLE: Final = "BOARD_WATCH_URL"


def load_credentials() -> McpCredentials:
    """Read both secrets from the environment and the endpoint from the
    stack's declaration, unless the environment overrides it.

    The default endpoint is where the MCPs repository declares the
    taskboard (:mod:`platform_core.stack_endpoints`), read after both
    secrets so a shell missing one hears about that first. It was a literal
    ``http://127.0.0.1:8033/mcp`` until 2026-09-24, when the hub stopped
    forwarding that port to diphtheria (MCPs board task c6fc4882) and every
    bridge on the hub lost the board at once.

    Returns:
        The credentials.

    Raises:
        AppError: ``API_KEY_MISSING`` or ``TENANT_ID_MISSING`` when the
            matching variable is unset or empty. The two are separate codes
            because the operator fixes them in different places: the key is
            taskboard-mcp's own container environment, the tenant id is a row
            in the ``tenants`` table. ``STACK_ENDPOINT_UNDECLARED`` when no
            override is set and the declaration names no taskboard url.
        OSError: When no override is set and the declaration cannot be read.
    """
    api_key = _test_hooks.env(API_KEY_VARIABLE)
    if api_key is None:
        raise AppError(
            code=BoardWatchErrorCode.API_KEY_MISSING,
            message=(
                f"{API_KEY_VARIABLE} is unset; read it from the taskboard-mcp "
                "container environment and export it before watching"
            ),
        )
    tenant_id = _test_hooks.env(TENANT_ID_VARIABLE)
    if tenant_id is None:
        raise AppError(
            code=BoardWatchErrorCode.TENANT_ID_MISSING,
            message=(
                f"{TENANT_ID_VARIABLE} is unset; it is the tenants row whose "
                "board is being read, and the board has no default tenant"
            ),
        )
    url = _test_hooks.env(URL_VARIABLE)
    if url is None:
        url = declared_url(_test_hooks.read_text(STACK_ENDPOINTS_PATH), TASKBOARD_SERVICE)
    return McpCredentials(url=url, api_key=api_key, tenant_id=tenant_id)
