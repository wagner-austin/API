"""The Linux recipe for a project whose check reads only loopback: in a network of its own.

MCPs board task 649fcc7a. A Linux check runs in the host's network
namespace, and Chromium fails every request it has in flight, to localhost
too, with ERR_NETWORK_CHANGED whenever the host's addresses change. A fleet
node's addresses change each time a GitHub Actions job's service container
starts or stops: on lavender on 2026-10-25 that failed idle's browser test
files, imported from vitest's Vite server on localhost:63315, in fleet jobs
dee131c3 and 1a8e13bc, each beside a container's bridge going down, and the
same check passed in a namespace holding only lo beside a container started
and removed every 4 s.

So a project that declares ``loopback_only`` runs its install steps in the
host's network, as every project does, and its ``make check`` through
``fleet-check-unit loopback`` (:mod:`fleet.contracts.check_unit`): a unit
bound to the check's own, with ``PrivateNetwork=yes``, in the directory the
build is in. Unprivileged user namespaces are refused on the nodes, so no
repository can make that namespace itself.
"""

from __future__ import annotations

from fleet.contracts.check_unit import CHECK_UNIT
from fleet.contracts.project import MAKE_TARGET
from fleet.core.agent_label import AGENT_LABEL_VARIABLE
from fleet.core.names import (
    CACHE_VARIABLE,
    POETRY_IN_PROJECT,
    POETRY_KEYRING_OFF,
    REGISTRY_VARIABLE,
    WORKSPACE_VARIABLE,
)

#: Every variable a build exports for its install steps and recipe, in the
#: order its export line names them. A loopback-only check's ``make check``
#: is handed each one again, with ``PATH``, through env(1): it runs as a unit
#: of its own that ``fleet-check-unit loopback`` starts through sudo, and
#: sudo resets the environment.
BUILD_EXPORTS: tuple[str, ...] = (
    "npm_config_cache",
    "POETRY_CACHE_DIR",
    POETRY_IN_PROJECT[0],
    POETRY_KEYRING_OFF[0],
    "PLAYWRIGHT_BROWSERS_PATH",
    "PYTEST_XDIST_AUTO_NUM_WORKERS",
    "CORVIS_TEST_MAX_WORKERS",
    AGENT_LABEL_VARIABLE,
    CACHE_VARIABLE,
    WORKSPACE_VARIABLE,
    REGISTRY_VARIABLE,
)


def recipe_command(loopback_unit: str | None) -> str:
    """The command line a build runs its recipe with.

    Args:
        loopback_unit: The check's own unit, ``fleet-<run id>``, for a
            project that declares ``loopback_only``, else None.

    Returns:
        ``make check``, or for a loopback-only project the same through
        ``fleet-check-unit loopback`` with ``PATH`` and every one of
        :data:`BUILD_EXPORTS` passed by value, so it runs in a network
        namespace holding only lo, in the directory the build is in, as the
        check's own account, and ends when the check's unit does.
    """
    if loopback_unit is None:
        return f"make {MAKE_TARGET}"
    passed = " ".join(f'{name}="${name}"' for name in ("PATH", *BUILD_EXPORTS))
    return f"{CHECK_UNIT} loopback '{loopback_unit}' /usr/bin/env {passed} make {MAKE_TARGET}"


__all__ = ["BUILD_EXPORTS", "recipe_command"]
