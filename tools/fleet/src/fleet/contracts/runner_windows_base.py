"""The machine settings a Windows runner host holds for every runner on it.

MCPs board task 1e9ab738. loki's first jobs as a runner host, on
2026-10-20, all failed at actions/setup-python: "setup.ps1 cannot be loaded
because running scripts is disabled on this system", since the machine's
LocalMachine execution policy was unset and its runners run as SYSTEM, which
has no CurrentUser policy of its own. The Windows base this package laid on
lavender before its reimage (deleted with the Windows runner side in
0531d9be2) held that policy, Win32 long paths, git's system core.longpaths
and the machine variables the SYSTEM account needs, each for a failure
measured then; this is that base, lifted for the one Windows runner host.

Long paths and core.longpaths are not declared: every Windows runner host
holds both, since MCPs CI extracts trees whose paths pass 260 characters.
The execution policy and the machine variables are the roster's.
"""

from __future__ import annotations

from platform_core.json_utils import JSONObject, JSONTypeError, JSONValue, require_list, require_str
from typing_extensions import TypedDict

#: Execution policies a runner host may be held to. Restricted, the Windows
#: client default, refuses every script a PowerShell step writes to
#: ``_temp``; RemoteSigned runs local scripts and is what lavender carried.
EXECUTION_POLICIES: tuple[str, ...] = ("RemoteSigned", "AllSigned")


class MachineVariable(TypedDict):
    """One machine-scope environment variable every runner service inherits.

    Attributes:
        name: The variable's name, a plain identifier. ``Path`` is refused:
            the machine PATH is the machine's own, which no runner needs
            changed (loki's carries git, Python and chocolatey's make).
        value: The exact value the variable must hold.
        reason: Why the host needs it, printed beside a drifted audit row.
    """

    name: str
    value: str
    reason: str


class WindowsHostBase(TypedDict):
    """What a Windows runner host holds before any runner runs on it.

    Attributes:
        execution_policy: The LocalMachine execution policy, one of
            :data:`EXECUTION_POLICIES`.
        machine_environment: Machine-scope variables every runner service
            inherits, distinct by name.
    """

    execution_policy: str
    machine_environment: list[MachineVariable]


def decode_machine_variable(value: JSONValue) -> MachineVariable:
    """Decode and validate one machine variable.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated variable.

    Raises:
        JSONTypeError: If the value is not an object, a field is missing or
            mistyped, the name is not a plain identifier or is ``Path``, or
            the value or reason is empty. An empty value would delete the
            variable the row exists to hold.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"machine variable must be a JSON object, got {type(value).__name__}")
    name = require_str(value, "name")
    if not (name[:1].isalpha() or name[:1] == "_") or not all(
        c.isalnum() or c == "_" for c in name
    ):
        raise JSONTypeError(f"a machine variable's name must be a plain identifier, got {name!r}")
    if name.lower() == "path":
        raise JSONTypeError("the machine PATH is the machine's own, not a declared variable")
    held = require_str(value, "value")
    reason = require_str(value, "reason")
    if not held or not reason:
        raise JSONTypeError(f"machine variable {name} needs a non-empty value and reason")
    return MachineVariable(name=name, value=held, reason=reason)


def decode_windows_host_base(value: JSONValue) -> WindowsHostBase:
    """Decode and validate a Windows runner host's base.

    Args:
        value: Value produced by the JSON loader.

    Returns:
        The validated base.

    Raises:
        JSONTypeError: If the value is not an object, the policy is not one
            of :data:`EXECUTION_POLICIES`, a variable is invalid, or two
            share a name, compared case-insensitively as Windows compares
            them: two values for one variable make the install rewrite it on
            every run and the audit fail one of the rows forever.
    """
    if not isinstance(value, dict):
        raise JSONTypeError(f"windows host base must be a JSON object, got {type(value).__name__}")
    policy = require_str(value, "execution_policy")
    if policy not in EXECUTION_POLICIES:
        raise JSONTypeError(
            f"execution_policy must be one of {', '.join(EXECUTION_POLICIES)}, got {policy!r}"
        )
    variables = [
        decode_machine_variable(entry) for entry in require_list(value, "machine_environment")
    ]
    names = [variable["name"].lower() for variable in variables]
    if len(set(names)) != len(names):
        raise JSONTypeError(f"machine_environment declares a name twice: {names}")
    return WindowsHostBase(execution_policy=policy, machine_environment=variables)


def encode_windows_host_base(base: WindowsHostBase) -> JSONObject:
    """Encode a Windows runner host's base.

    Args:
        base: The base.

    Returns:
        JSON-serialisable mapping carrying every field.
    """
    return {
        "execution_policy": base["execution_policy"],
        "machine_environment": [
            {"name": each["name"], "value": each["value"], "reason": each["reason"]}
            for each in base["machine_environment"]
        ],
    }


__all__ = [
    "EXECUTION_POLICIES",
    "MachineVariable",
    "WindowsHostBase",
    "decode_machine_variable",
    "decode_windows_host_base",
    "encode_windows_host_base",
]
