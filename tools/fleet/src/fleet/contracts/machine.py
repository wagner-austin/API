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
  do not (emerald, a NixOS storage machine with no Docker, compiler or test
  database for a check to run on).

A SERVER TAKES NO FLEET WORK (MCPs board task 4ec65b8f; the operator,
2026-10-24: "so demure*will be dedicated as the server pc. not a worker."). A
``server`` runs the corvis stack, so the fleet's pool is never offered to
it: its ``dispatch`` is a reason, or a capacity whose budget declares
``pinned_checks_only``, which leaves it only the checks pinned to it by
name, the ones that read the server itself. Dispatch reads that budget flag
and never the role, so the role is held to the flag here rather than
spelled a second time in :mod:`fleet.core.capacity`.

So ``enabled`` is said once, whether the machine is expected to answer, and
a node dispatches exactly when it is enabled and not held. ``host`` is the
machine's name, which is its ssh alias everywhere.

EVERY MACHINE IS LINUX (board task e705f08c). The fleet renders only ``sh``
(:mod:`fleet.core.dialect`), so a ``platform`` other than ``linux`` is
refused rather than carried: loki, the last Windows machine, left the roster
on 2026-10-25 and returns through enrolment once it runs NixOS.

THIS PACKAGE DECODES WHAT IT READS. The identity fields only MCPs reads
(``tailnetIp``, ``reachKey``, ``bootMac``, ``netbootWifi``, ``installDisk``,
``disabledUntil``, ``tunnel``, ``guest``, ``duties``, ``standby``,
``ciWorkersPerRunner``, ``boot``, ``notes``) are decoded by MCPs
``fleet-mcp``'s registry decoder, which every fleet audit runs over this
directory; deciding their shapes twice, in two languages, would be the
duplicate this file replaced.

``ledger_observer`` IS READ HERE TOO, because the hub tick's observe pass
acts on it (MCPs board task 911b0303). It names the arm that stamps the
machine's watermark in ``agent_session_observer_passes``, and the fleet
audit holds each machine to the arm it names. The pass used to choose its
machines by role instead, on the premise that a hub is observed by
``pcsession-mcp``; on 2026-10-18 MCPs d48341e6f made diphtheria a hub while
its file still named ``fleet-arm``, and the arm last passed over it at
07:12:06Z. Reading the one declaration both sides grade by keeps the arm
and the audit from disagreeing again.
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

from fleet.contracts.node import NodeConfig, decode_node_config, encode_node_config

#: The one ``platform`` a machine file may declare.
PLATFORM = "linux"


class MachineRole(StrEnum):
    """What a machine is for, the identity registry's closed vocabulary."""

    HUB = "hub"
    VPN_JUMP = "vpn-jump"
    WORKER = "worker"
    SESSION_HOST = "session-host"
    #: Runs the corvis stack and takes no fleet work but the checks pinned to
    #: it by name (MCPs board task 4ec65b8f).
    SERVER = "server"
    CLIENT = "client"
    #: A machine the operator can reach but does not own (a relative's PC,
    #: helped remotely with consent); MCPs fleet-mcp's FleetRole names the same
    #: seven, and fleet-enroll takes it as --role (MCPs board task e447de72).
    UNTRUSTED_MANAGED = "untrusted-managed"


class LedgerObserver(StrEnum):
    """Which arm records a machine's Claude Code sessions in the board's ledger.

    MCPs fleet-mcp's SessionObserver names the same three, the words the
    ledger's observer column admits.
    """

    #: The hub tick's observe pass, over ssh (:mod:`fleet.core.observe`).
    FLEET_ARM = "fleet-arm"
    #: ``pcsession-mcp``, over its read-only mount of one machine's sessions.
    PCSESSION = "pcsession"
    #: Nothing: a client, or a machine the operator does not own.
    NONE = "none"


class Machine(TypedDict):
    """One machine as the roster declares it, as far as this package reads it.

    Attributes:
        name: The machine's name, its file's stem and its ssh alias.
        role: What it is for.
        user: The account the fleet provisioned on it, or None for a client,
            which is never provisioned.
        enabled: Whether it is expected to answer at all.
        ledger_observer: The arm that records its sessions in the ledger.
        dispatch: What dispatch may take from it, or None when it declares
            no capacity.
        held: Why its runners claim no work, or None when they do or when
            it is simply off; set for every machine whose ``dispatch`` is a
            reason and for a declared node held out of dispatch.
    """

    name: str
    role: MachineRole
    user: str | None
    enabled: bool
    ledger_observer: LedgerObserver
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
            reads is missing or mistyped, ``role`` or ``ledger_observer`` is
            outside its closed set, ``platform`` is not :data:`PLATFORM`,
            ``dispatch`` is neither a non-empty reason nor an object,
            ``held`` is neither null nor a non-empty reason, a node held
            out of dispatch is not enabled (a machine that is off is held by
            being off, and a reason on it would say a second thing), or a
            server declares a capacity that would take checks not pinned to
            it.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"machine {name!r} must be a JSON object, got {type(value).__name__}")
    role = _decode_role(name, require_str(value, "role"))
    platform = require_str(value, "platform")
    if platform != PLATFORM:
        raise JSONTypeError(
            f"machine {name!r} platform must be {PLATFORM!r}, got {platform!r}; the fleet "
            "renders every script in sh and runs no machine of another platform"
        )
    enabled = require_bool(value, "enabled")
    ledger_observer = _decode_ledger_observer(name, require_str(value, "ledger_observer"))
    raw_user = value.get("user")
    dispatch = value.get("dispatch")
    if isinstance(dispatch, str):
        return Machine(
            name=name,
            role=role,
            user=None if raw_user is None else narrow_json_to_str(raw_user),
            enabled=enabled,
            ledger_observer=ledger_observer,
            dispatch=None,
            held=_reason(name, dispatch, "dispatch"),
        )
    if not isinstance(dispatch, dict):
        raise JSONTypeError(
            f"machine {name!r} must declare 'dispatch': the reason it takes no dispatched work, "
            "or an object of what dispatch may take from it"
        )
    held = _decode_held(name, dispatch, enabled=enabled)
    node = decode_node_config(dispatch, host=name, enabled=enabled and held is None)
    if role is MachineRole.SERVER and not node["budget"]["pinned_checks_only"]:
        raise JSONTypeError(
            f"machine {name!r} is a server, which takes no fleet work, and its dispatch would "
            "take any check its tags fit: declare budget.pinned_checks_only true, so it takes "
            "only the checks pinned to it, or give dispatch as the reason it takes none"
        )
    return Machine(
        name=name,
        role=role,
        user=None if raw_user is None else narrow_json_to_str(raw_user),
        enabled=enabled,
        ledger_observer=ledger_observer,
        dispatch=node,
        held=held,
    )


def encode_machine(machine: Machine) -> JSONObject:
    """Encode the fields of a machine's file this package reads.

    Args:
        machine: The machine.

    Returns:
        Its ``role``, ``user``, ``platform`` (:data:`PLATFORM`), ``enabled``,
        ``ledger_observer`` and ``dispatch``: its held reason when it
        declares no capacity, else its capacity with ``held``.
    """
    node = machine["dispatch"]
    dispatch: JSONValue = (
        machine["held"] if node is None else {**encode_node_config(node), "held": machine["held"]}
    )
    return {
        "role": machine["role"].value,
        "user": machine["user"],
        "platform": PLATFORM,
        "enabled": machine["enabled"],
        "ledger_observer": machine["ledger_observer"].value,
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


def _decode_ledger_observer(name: str, value: str) -> LedgerObserver:
    """Read a machine's ledger observer into the closed set.

    Args:
        name: The machine, for the message.
        value: The declared observer.

    Returns:
        The observer.

    Raises:
        JSONTypeError: If it is not one of :class:`LedgerObserver`'s words.
    """
    observer = find_member(value, LedgerObserver)
    if observer is not None:
        return observer
    raise JSONTypeError(
        f"machine {name!r} ledger_observer must be one of {', '.join(LedgerObserver)}, "
        f"got {value!r}"
    )


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


__all__ = [
    "PLATFORM",
    "LedgerObserver",
    "Machine",
    "MachineRole",
    "decode_machine",
    "encode_machine",
]
