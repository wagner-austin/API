"""One machine of the fleet's roster, ``registry/machines/<name>.json``.

THE FLEET IS WRITTEN DOWN ONCE (MCPs board task e447de72). Until 2026-10-19
it was written down twice: MCPs ``fleet-mcp/fleet-nodes.json`` held every
machine on the tailnet with its identity and reach (role, account, tailnet
address, tunnel, boot units, ...), and this registry's ``nodes.json`` held
the machines dispatch may use with what each can take. Both declared
``enabled`` and ``platform``, each declared a ``gpu`` and a
``test_database``, and nothing held them together but a comparison a
session ran by hand: on 2026-10-19 lacrymal's ``test_database`` read true
here and false there until a session noticed. The operator, that day: "One
roster. Fold fleet-nodes.json into the API registry and have everything
read it." The roster lives here because every dispatch must work with no
MCPs checkout present, and MCPs already reads this registry.

ONE FILE PER MACHINE, NAMED BY THE MACHINE, as ``projects/`` holds one file
per project: a name is a path, so no two files declare one machine, and
enrolling a machine writes one new file. The file carries the machine's
identity and, under ``dispatch``, what dispatch may take from it:

* a string, the reason the machine takes no dispatched work and declares no
  capacity (the hub, the session host, the phone), since silence cannot be
  told from an oversight;
* or an object, the capacity :func:`fleet.contracts.node.decode_node_config`
  reads, with ``held``: null when its runners claim work, or the reason they
  do not (lavender's Windows lane, retired on 2026-09-29 while lavender-wsl
  still names lavender as the machine it runs inside).

So ``enabled`` is said once, whether the machine is expected to answer, and
a node dispatches exactly when it is enabled and not held. ``host`` is the
machine's name, which is its ssh alias everywhere, and a node runs inside
another's WSL exactly when its ``guest`` names a host.

THIS PACKAGE DECODES WHAT IT READS. The identity fields only MCPs reads
(``tailnetIp``, ``reachKey``, ``bootMac``, ``disabledUntil``, ``tunnel``,
``guest.sshPort``, ``duties``, ``ledger_observer``, ``ciWorkersPerRunner``,
``boot``, ``notes``) are decoded by MCPs ``fleet-mcp``'s registry decoder,
which every fleet audit runs over this directory; deciding their shapes
twice, in two languages, would be the duplicate this file replaced.
"""

from __future__ import annotations

from enum import StrEnum

from platform_core.json_utils import (
    JSONObject,
    JSONTypeError,
    JSONValue,
    narrow_json_to_str,
    require_bool,
    require_str,
)
from platform_core.members import find_member
from typing_extensions import TypedDict

from fleet.contracts.node import (
    NodeConfig,
    NodePlatform,
    decode_node_config,
    decode_node_platform,
    encode_node_config,
)


class MachineRole(StrEnum):
    """What a machine is for, the identity registry's closed vocabulary."""

    HUB = "hub"
    VPN_JUMP = "vpn-jump"
    WORKER = "worker"
    SESSION_HOST = "session-host"
    CLIENT = "client"
    #: A machine the operator can reach but does not own (a relative's PC,
    #: helped remotely with consent); MCPs fleet-mcp's FleetRole names the same
    #: six, and fleet-enroll takes it as --role (MCPs board task e447de72).
    UNTRUSTED_MANAGED = "untrusted-managed"


class Machine(TypedDict):
    """One machine as the roster declares it, as far as this package reads it.

    Attributes:
        name: The machine's name, its file's stem and its ssh alias.
        role: What it is for.
        user: The account the fleet provisioned on it, or None for a client,
            which is never provisioned.
        platform: Its operating-system family, the dialect every script to
            it is rendered in.
        enabled: Whether it is expected to answer at all.
        guest_host: The machine it runs inside and is reached through, or
            None for one on the tailnet in its own right.
        dispatch: What dispatch may take from it, or None when it declares
            no capacity.
        held: Why its runners claim no work, or None when they do or when
            it is simply off; set for every machine whose ``dispatch`` is a
            reason and for a declared node held out of dispatch.
    """

    name: str
    role: MachineRole
    user: str | None
    platform: NodePlatform
    enabled: bool
    guest_host: str | None
    dispatch: NodeConfig | None
    held: str | None


def decode_machine(name: str, value: JSONValue) -> Machine:
    """Decode and validate one machine's file.

    Args:
        name: The machine's name, its file's stem.
        value: The file's object.

    Returns:
        The validated machine.

    Raises:
        JSONTypeError: If the value is not an object, a field this package
            reads is missing or mistyped, ``role`` or ``platform`` is outside
            its closed set, ``guest`` is neither null nor an object naming a
            host, ``dispatch`` is neither a non-empty reason nor an object,
            ``held`` is neither null nor a non-empty reason, or a node held
            out of dispatch is not enabled (a machine that is off is held by
            being off, and a reason on it would say a second thing).
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"machine {name!r} must be a JSON object, got {type(value).__name__}")
    role = _decode_role(name, require_str(value, "role"))
    platform = decode_node_platform(require_str(value, "platform"))
    enabled = require_bool(value, "enabled")
    raw_user = value.get("user")
    guest_host = _decode_guest_host(name, value)
    dispatch = value.get("dispatch")
    if isinstance(dispatch, str):
        return Machine(
            name=name,
            role=role,
            user=None if raw_user is None else narrow_json_to_str(raw_user),
            platform=platform,
            enabled=enabled,
            guest_host=guest_host,
            dispatch=None,
            held=_reason(name, dispatch, "dispatch"),
        )
    if not isinstance(dispatch, dict):
        raise JSONTypeError(
            f"machine {name!r} must declare 'dispatch': the reason it takes no dispatched work, "
            "or an object of what dispatch may take from it"
        )
    held = _decode_held(name, dispatch, enabled=enabled)
    return Machine(
        name=name,
        role=role,
        user=None if raw_user is None else narrow_json_to_str(raw_user),
        platform=platform,
        enabled=enabled,
        guest_host=guest_host,
        dispatch=decode_node_config(
            dispatch,
            host=name,
            platform=platform,
            enabled=enabled and held is None,
            wsl_host=guest_host,
        ),
        held=held,
    )


def encode_machine(machine: Machine) -> JSONObject:
    """Encode the fields of a machine's file this package reads.

    Args:
        machine: The machine.

    Returns:
        Its ``role``, ``user``, ``platform``, ``enabled``, ``guest`` (as
        ``{host}`` or null) and ``dispatch``: its held reason when it
        declares no capacity, else its capacity with ``held``.
    """
    node = machine["dispatch"]
    dispatch: JSONValue = (
        machine["held"] if node is None else {**encode_node_config(node), "held": machine["held"]}
    )
    guest_host = machine["guest_host"]
    return {
        "role": machine["role"].value,
        "user": machine["user"],
        "platform": machine["platform"].value,
        "enabled": machine["enabled"],
        "guest": None if guest_host is None else {"host": guest_host},
        "dispatch": dispatch,
    }


def _decode_role(name: str, value: str) -> MachineRole:
    """Read a machine's role into the closed set.

    Args:
        name: The machine, for the message.
        value: The declared role.

    Returns:
        The role.

    Raises:
        JSONTypeError: If it is not one of :class:`MachineRole`'s words.
    """
    role = find_member(value, MachineRole)
    if role is not None:
        return role
    raise JSONTypeError(
        f"machine {name!r} role must be one of {', '.join(MachineRole)}, got {value!r}"
    )


def _decode_guest_host(name: str, value: JSONObject) -> str | None:
    """Read the machine another one runs inside, if any.

    Args:
        name: The machine, for the message.
        value: Its object.

    Returns:
        ``guest.host``, or None when ``guest`` is null.

    Raises:
        JSONTypeError: If ``guest`` is absent, or neither null nor an object
            carrying a non-empty ``host``.
    """
    if "guest" not in value:
        raise JSONTypeError(
            f"machine {name!r} must declare 'guest': null for a machine on the tailnet in its "
            "own right, or {host, sshPort} for one reached through the machine it runs inside"
        )
    guest = value["guest"]
    if guest is None:
        return None
    if not isinstance(guest, dict):
        raise JSONTypeError(f"machine {name!r} guest must be an object or null")
    return _reason(name, guest.get("host"), "guest.host")


def _decode_held(name: str, dispatch: JSONObject, *, enabled: bool) -> str | None:
    """Read why a declared node's runners claim no work, if they do not.

    Args:
        name: The machine, for the message.
        dispatch: Its ``dispatch`` object.
        enabled: Whether the machine is expected to answer.

    Returns:
        The reason, or None.

    Raises:
        JSONTypeError: If ``held`` is absent, is neither null nor a
            non-empty string, or is set on a machine that is not enabled.
    """
    if "held" not in dispatch:
        raise JSONTypeError(
            f"machine {name!r} dispatch must declare 'held': null when its runners claim work, "
            "or the reason they do not"
        )
    value = dispatch["held"]
    if value is None:
        return None
    reason = _reason(name, value, "dispatch.held")
    if not enabled:
        raise JSONTypeError(
            f"machine {name!r} is not enabled and its dispatch is held ({reason!r}); a machine "
            "that is off claims nothing already, so hold it by its enabled alone"
        )
    return reason


def _reason(name: str, value: JSONValue, field: str) -> str:
    """Read a required non-empty string.

    Args:
        name: The machine, for the message.
        value: The value.
        field: Its path in the machine's object.

    Returns:
        The string.

    Raises:
        JSONTypeError: If it is not a non-empty string.
    """
    if not isinstance(value, str) or not value.strip():
        raise JSONTypeError(f"machine {name!r} {field} must be a non-empty string, got {value!r}")
    return value


__all__ = ["Machine", "MachineRole", "decode_machine", "encode_machine"]
