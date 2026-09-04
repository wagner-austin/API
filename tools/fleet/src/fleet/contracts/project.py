"""What one project costs to build, declared rather than discovered.

THE FIELD THAT MATTERS IS ``worker_ram_gb`` ON THE BUDGET, NOT HERE. A node
declares what a worker costs it; a project declares what it needs. The two
meet in :func:`~fleet.core.capacity.plan_dispatch`, and keeping them apart is
what lets one project be dispatchable to a small node and another not.

WHY ``command`` IS NOT A FIELD. Every project here is built by ``make check``,
and that is the point of the monorepo's 48 Makefiles: the recipe is the
project's own business and the dispatcher's business is only where to run it.
A per-project command field would let a project be dispatched by a recipe that
is not the one a human runs locally, and the two would drift silently. If a
project needs a different recipe, it changes its Makefile.
"""

from __future__ import annotations

import math

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    require_float,
    require_int,
)
from typing_extensions import TypedDict

from fleet.contracts.resources import decode_resources, encode_resources

#: The recipe every dispatch runs. One string, in one place, so a dispatch and
#: a local build cannot diverge.
MAKE_TARGET = "check"


class ProjectConfig(TypedDict):
    """A body of work the fleet may build.

    Attributes:
        worker_ram_gb: Memory one test worker of THIS project holds. Declared
            per project because it is a property of what the suite imports:
            a torch suite reserves about 1.1 GB per worker on import while a
            pure-Python one holds a fraction of that, and using one number
            for both either wastes a large node or wedges a small one.
        minimum_workers: Fewest workers this suite is worth dispatching with.
            A node that cannot afford this many is refused rather than given
            the work at one worker, because a suite that takes four minutes
            at sixteen workers takes an hour at one and will be preempted by
            its own lease expiring.
        expected_minutes: How long the suite takes on a node of the fleet's
            usual size. Used to size the lease, so a dispatch cannot hold a
            project's environment for longer than the work plausibly needs.
        exclusive_resources: Fleet-wide things this suite needs exclusively,
            by name. Empty for a self-contained suite, which is every project
            whose cost is only CPU and memory.

            DECLARED PER PROJECT BECAUSE IT IS A FACT ABOUT THE RECIPE.
            ``packages/db``'s ``test`` target runs ``migrate-test`` before
            vitest, and that applies migrations to a single shared
            ``corvis_test``; the suite touches that database every time it
            runs, on any node. A workspace-level or node-level declaration
            could not say which SUITES contend, only which machines do, and
            the machines are not the thing there is one of.
    """

    worker_ram_gb: float
    minimum_workers: int
    expected_minutes: int
    exclusive_resources: tuple[str, ...]


def lease_seconds(project: ProjectConfig, *, slack: float) -> int:
    """How long a dispatch of this project may hold its lease.

    Args:
        project: The project being dispatched.
        slack: Multiplier over ``expected_minutes``, covering a slower node
            and a cold environment sync. Passed rather than defaulted so the
            caller that knows whether this is a cold node decides.

    Returns:
        Whole seconds. Rounded UP, because a lease that expires one second
        into a running suite is the failure the expiry exists to prevent,
        inverted.

    Raises:
        ValueError: If ``slack`` is not greater than 1.0. A slack of exactly
            one sizes the lease to the expected duration, so any node slower
            than the one the estimate came from loses its lease mid-suite;
            below one it is guaranteed to. Refused rather than clamped,
            because a caller passing 0.5 has made a mistake that clamping
            would hide.
    """
    if slack <= 1.0:
        raise ValueError(
            f"slack must be greater than 1.0, got {slack}; a lease sized to the expected "
            "duration expires mid-suite on any node slower than the one that produced the "
            "estimate"
        )
    return math.ceil(project["expected_minutes"] * 60 * slack)


def encode_project_config(project: ProjectConfig) -> JSONObject:
    """Encode one project's declaration.

    Args:
        project: The project to encode.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "worker_ram_gb": project["worker_ram_gb"],
        "minimum_workers": project["minimum_workers"],
        "expected_minutes": project["expected_minutes"],
        "exclusive_resources": encode_resources(project["exclusive_resources"]),
    }


def decode_project_config(value: JSONValue) -> ProjectConfig:
    """Decode and validate one project's declaration.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated project.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, or a field holds a number that cannot describe work.
            ``worker_ram_gb`` must be positive because it divides a node's
            free memory; ``minimum_workers`` must be at least one because a
            project needing zero workers is not a project; and
            ``expected_minutes`` must be positive because it sizes the lease
            and a zero would produce one that has already expired.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"project must be a JSON object, got {type(value).__name__}")
    worker_ram_gb = require_float(value, "worker_ram_gb")
    if worker_ram_gb <= 0.0:
        raise JSONTypeError(
            f"worker_ram_gb must be positive, got {worker_ram_gb}; it divides a node's free "
            "memory to give a worker count"
        )
    minimum_workers = require_int(value, "minimum_workers")
    if minimum_workers < 1:
        raise JSONTypeError(
            f"minimum_workers must be at least 1, got {minimum_workers}; a project that needs "
            "no workers has no suite to dispatch"
        )
    expected_minutes = require_int(value, "expected_minutes")
    if expected_minutes < 1:
        raise JSONTypeError(
            f"expected_minutes must be at least 1, got {expected_minutes}; it sizes the lease, "
            "and a zero produces one that has already expired when it is taken"
        )
    return ProjectConfig(
        worker_ram_gb=worker_ram_gb,
        minimum_workers=minimum_workers,
        expected_minutes=expected_minutes,
        exclusive_resources=decode_resources(
            value.get("exclusive_resources"), field="project.exclusive_resources"
        ),
    )


__all__ = [
    "MAKE_TARGET",
    "ProjectConfig",
    "decode_project_config",
    "encode_project_config",
    "lease_seconds",
]
