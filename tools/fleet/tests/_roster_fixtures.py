"""Machine files of the roster for tests, and the accessors that change them in place.

The roster is ``registry/machines/<name>.json`` (:mod:`fleet.contracts.machine`,
MCPs board task e447de72), so a test that changes a node changes its machine's
file, through these, rather than a ``nodes`` object no registry declares.
"""

from __future__ import annotations

from platform_core.json_utils import JSONObject, JSONValue


def dispatch_document() -> JSONObject:
    """The ``dispatch`` object of a Windows worker that claims work.

    Returns:
        Its capacity, with ``held`` null.
    """
    return {
        "held": None,
        "stage_root": "C:/fleet/stage",
        "logical_cores": 16,
        "ram_gb": 32.0,
        "gpu": None,
        "test_database": False,
        "rust": None,
        "cxx": None,
        "docker": None,
        "stack": None,
        "elevated": False,
        "budget": {
            "reserved_cores": 2,
            "reserved_ram_gb": 4.0,
            "worker_ram_gb": 1.1,
            "max_disk_gb": 20.0,
            "checks_at_once": None,
            "pinned_checks_only": False,
        },
    }


def machine_document(
    *,
    role: str = "hub",
    platform: str = "windows",
    enabled: bool = True,
    ledger_observer: str = "none",
    guest: str | None = None,
    dispatch: JSONValue = None,
) -> JSONObject:
    """One machine file of the roster, as this package reads it.

    Args:
        role: Its role, a hub by default, as diphtheria is both a hub and a
            dispatch node.
        platform: Its platform.
        enabled: Whether it is expected to answer.
        ledger_observer: The arm that records its sessions. None observes
            it by default: a hub tick's observe pass visits only a machine
            declaring ``fleet-arm``, so a test of anything else is asked no
            ssh it did not script.
        guest: The machine it runs inside, or None.
        dispatch: Its ``dispatch``; None gives :func:`dispatch_document`.

    Returns:
        The machine's object.
    """
    return {
        "role": role,
        "user": "austi",
        "platform": platform,
        "enabled": enabled,
        "ledger_observer": ledger_observer,
        "guest": None if guest is None else {"host": guest, "sshPort": 2222},
        "dispatch": dispatch_document() if dispatch is None else dispatch,
    }


def machines_of(document: JSONObject) -> JSONObject:
    """A workspace document's machines, to read or change in place.

    Args:
        document: A workspace document.

    Returns:
        Its ``machines`` object.

    Raises:
        AssertionError: When it is not an object.
    """
    machines = document["machines"]
    if not isinstance(machines, dict):
        raise AssertionError("the document must declare its machines as an object")
    return machines


def machine_of(document: JSONObject, name: str) -> JSONObject:
    """One machine of a workspace document, to read or change in place.

    Args:
        document: A workspace document.
        name: The machine.

    Returns:
        Its object.

    Raises:
        AssertionError: When it is not an object.
    """
    machine = machines_of(document)[name]
    if not isinstance(machine, dict):
        raise AssertionError(f"machine {name} must be an object")
    return machine


def dispatch_of(document: JSONObject, name: str) -> JSONObject:
    """One machine's ``dispatch`` capacity, to read or change in place.

    Args:
        document: A workspace document.
        name: The machine.

    Returns:
        Its ``dispatch`` object.

    Raises:
        AssertionError: When the machine declares no capacity.
    """
    dispatch = machine_of(document, name)["dispatch"]
    if not isinstance(dispatch, dict):
        raise AssertionError(f"machine {name} must declare capacity")
    return dispatch


__all__ = [
    "dispatch_document",
    "dispatch_of",
    "machine_document",
    "machine_of",
    "machines_of",
]
