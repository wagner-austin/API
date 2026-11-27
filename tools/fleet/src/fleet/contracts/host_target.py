"""The make target a project runs on one node's own account, declared in the registry.

MCPs board task 57791f39. osrs-lab's manual-desktop has a ``make update``
that builds the recorder and the login plugin at its tree's commit and
installs them into every OSRS desktop home on the machine running it. Those
homes are under the account of loki, the machine holding the desktops, so a
fleet check cannot run it: a docker project's check runs as execdocker in an
isolated copy that cannot reach the runner's home
(:mod:`fleet.core.linux_isolated_build`), which is what keeps a check from
touching production. Nor can loki fetch the commit itself: a node holds no
credential (:mod:`fleet.core.export`).

So a project declares one host target: the make target, the node whose
account runs it, the workers its build may take and the deadline it is
stopped at. :mod:`fleet.core.host_target` stages the remote's default
branch on that node from the hub's mirror and runs the target there. The
node is named here and nowhere else, decoded by the one node-name decoder a
project's own pin is (:func:`fleet.contracts.node.decode_node_name`), so
where the work runs has one source.

The target's grammar is a make target the node's ``sh`` can be given bare:
a lowercase letter, then lowercase letters, digits and hyphens.
"""

from __future__ import annotations

import re
from typing import Final

from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue, require_int, require_str
from typing_extensions import ReadOnly, TypedDict

from fleet.contracts.node import decode_node_name


class HostTarget(TypedDict):
    """A make target a project runs on one node's own account.

    Attributes:
        target: The make target, run in the project's directory of the
            staged tree.
        node: The registry name of the node whose account runs it.
        workers: The CPUs its build may take, given to it as the fleet gives
            a check its grant.
        deadline_seconds: How long it may run before the node stops it.
    """

    target: ReadOnly[str]
    node: ReadOnly[str]
    workers: ReadOnly[int]
    deadline_seconds: ReadOnly[int]


#: A make target the node's ``sh`` is given bare.
TARGET_NAME: Final[re.Pattern[str]] = re.compile(r"[a-z][a-z0-9-]*")

#: The shortest deadline a host target may declare: a minute, so a deadline
#: written in minutes by mistake is refused rather than stopping the target
#: at once.
MIN_DEADLINE_SECONDS: Final[int] = 60

#: The longest deadline a host target may declare: half an hour. The hub runs
#: the target through the rolled launcher (:mod:`fleet.cli.rolled`), which
#: ends its agent at :data:`fleet.cli.rolled.AGENT_WALL_SECONDS`, 39 minutes;
#: inside it go this deadline, ``timeout``'s kill after it, the transport's
#: own bound and the fetch, stage and retire around the run, so the node's
#: deadline, never the launcher's, is what stops a run that overruns.
MAX_DEADLINE_SECONDS: Final[int] = 1800

#: The keys a host target declares, all of them required.
KEYS: Final[frozenset[str]] = frozenset({"target", "node", "workers", "deadline_seconds"})


def decode_host_target(value: JSONValue) -> HostTarget | None:
    """Decode a project's optional ``host_target``.

    Args:
        value: The key's value, or None when the key is absent.

    Returns:
        The host target, or None for a project that declares none.

    Raises:
        JSONTypeError: When the value is not an object of exactly
            :data:`KEYS`; when the target is outside :data:`TARGET_NAME`;
            when the node is not a non-empty string
            (:func:`fleet.contracts.node.decode_node_name`); when workers is below
            one; or when the deadline is below :data:`MIN_DEADLINE_SECONDS`
            or above :data:`MAX_DEADLINE_SECONDS`. A misspelt declaration
            would run some other target, nowhere, stop it before it began,
            or outlast the launcher that runs it.
    """
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != KEYS:
        raise JSONTypeError(
            f"project.host_target must be an object of exactly {sorted(KEYS)}, got {value!r}"
        )
    target = require_str(value, "target")
    if TARGET_NAME.fullmatch(target) is None:
        raise JSONTypeError(
            f"project.host_target.target must be a make target of a lowercase letter then "
            f"lowercase letters, digits and hyphens, got {target!r}"
        )
    node = decode_node_name(value["node"], field="project.host_target.node")
    workers = require_int(value, "workers")
    if workers < 1:
        raise JSONTypeError(
            f"project.host_target.workers must be at least 1, got {workers}; a build granted "
            "no CPU refuses to start"
        )
    deadline_seconds = require_int(value, "deadline_seconds")
    if deadline_seconds < MIN_DEADLINE_SECONDS:
        raise JSONTypeError(
            f"project.host_target.deadline_seconds must be at least {MIN_DEADLINE_SECONDS}, "
            f"got {deadline_seconds}; a shorter deadline stops the target before its build "
            "can finish"
        )
    if deadline_seconds > MAX_DEADLINE_SECONDS:
        raise JSONTypeError(
            f"project.host_target.deadline_seconds must be at most {MAX_DEADLINE_SECONDS}, "
            f"got {deadline_seconds}; the rolled launcher would end the run before the node's "
            "deadline stopped it, and no end would be recorded"
        )
    return HostTarget(target=target, node=node, workers=workers, deadline_seconds=deadline_seconds)


def encode_host_target(host_target: HostTarget | None) -> JSONObject | None:
    """Encode a project's ``host_target``.

    Args:
        host_target: The host target, or None.

    Returns:
        Its JSON object, or None for a project that declares none.
    """
    if host_target is None:
        return None
    return {
        "target": host_target["target"],
        "node": host_target["node"],
        "workers": host_target["workers"],
        "deadline_seconds": host_target["deadline_seconds"],
    }


__all__ = [
    "KEYS",
    "MAX_DEADLINE_SECONDS",
    "MIN_DEADLINE_SECONDS",
    "TARGET_NAME",
    "HostTarget",
    "decode_host_target",
    "encode_host_target",
]
