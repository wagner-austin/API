"""Machine-scope environment variables a runner host declares.

The roster's ``base.machine_environment`` names variables every Windows
runner service must inherit. The first one exists because of the account
the services run as (:mod:`fleet.core.runner_account`): SYSTEM's profile is
``C:\\Windows\\System32\\config\\systemprofile``, so a poetry virtualenv
created under it sits inside System32, and a 32-bit program -- GnuWin32's
``make`` and ezwinports' ``make`` 4.4.1 are both i386 -- is redirected to
SysWOW64 when it names a System32 path. MCPs' session-audit recipes launch
that venv's python from ``make``, and on the rebuilt lavender every such
launch failed with "The system cannot find the file specified" (MCPs run
36264009927). Moving poetry's cache outside System32 made the same tests
pass under SYSTEM with either make, measured 2026-09-26.

WRITTEN THROUGH THE REGISTRY, APPLIED BY A REBOOT. The value is the machine
environment key's own, written with the same cmdlets as the base's other
registry writes, so the executed tests can stand in for them. A service
inherits the service manager's environment, which Windows builds at boot,
so a changed variable asks the rebuild for the reboot it already knows how
to perform and wait out; restarting the services would not reach them.
"""

from __future__ import annotations

from fleet.contracts.runner_base import MachineVariable
from fleet.contracts.runners import HostRunnerSpec
from fleet.core.script_values import scriptable

#: Where machine-scope environment variables live.
MACHINE_ENVIRONMENT_KEY = "HKLM:\\SYSTEM\\CurrentControlSet\\Control\\Session Manager\\Environment"


def machine_variable_check_id(variable: MachineVariable) -> str:
    """The audit row holding one machine variable to its value.

    Args:
        variable: The declared variable.

    Returns:
        ``machine-env:<name>``; the value is the drift detail, not the id,
        so a value change in the roster keeps the row's identity.
    """
    return f"machine-env:{variable['name']}"


def render_machine_environment_parameters(spec: HostRunnerSpec) -> list[str]:
    """The Windows base's param-block lines for the declared variables.

    The key is a parameter so the Pester suite over the committed render
    passes a scratch HKCU key; the variables are one ``name=value`` string
    each, split at the first ``=``, which no variable name contains.

    Args:
        spec: The host's roster entry.

    Returns:
        ``$EnvironmentKey`` and ``$MachineVariables``, each ending in a
        comma; the array is empty when the roster declares none.

    Raises:
        ValueError: When a name or value cannot be embedded verbatim; see
            :func:`fleet.core.script_values.scriptable`.
    """
    pairs = []
    for variable in spec["base"]["machine_environment"]:
        name = scriptable(variable["name"], label="machine variable name")
        value = scriptable(variable["value"], label=f"machine variable {name}")
        pairs.append(f"'{name}={value}'")
    return [
        f"    [string]$EnvironmentKey = '{MACHINE_ENVIRONMENT_KEY}',",
        f"    [string[]]$MachineVariables = @({', '.join(pairs)}),",
    ]


def render_machine_environment_lines() -> list[str]:
    """The Windows base's lines that set each declared machine variable.

    Returns:
        One loop over ``$MachineVariables``. A variable already holding its
        value is left alone; a changed one is written, printed, and sets the
        base's ``$Reboot``, so the services start into it. The held value is
        read with ``GetValue``, which answers ``$null`` for an absent one
        where a property read fails under strict mode.
    """
    return [
        "foreach ($pair in $MachineVariables) {",
        "    $name, $value = $pair -split '=', 2",
        "    if ([string](Get-Item -LiteralPath $EnvironmentKey).GetValue($name) -cne $value) {",
        "        Set-ItemProperty -LiteralPath $EnvironmentKey -Name $name -Value $value",
        "        $Reboot = $true",
        "        Write-Output ('set the machine variable ' + $name + ' to ' + $value)",
        "    }",
        "}",
    ]


def render_machine_environment_check_lines(spec: HostRunnerSpec) -> list[str]:
    """The audit driver's lines for every declared variable's row.

    The value is JOINED from the pipeline, never cast, so an absent
    variable still emits its drifted row rather than a short transcript.

    Args:
        spec: The host's roster entry.

    Returns:
        One ``Emit`` per variable, with the value the registry holds as the
        drift detail.

    Raises:
        ValueError: When a name or value cannot be embedded verbatim.
    """
    lines: list[str] = []
    for variable in spec["base"]["machine_environment"]:
        name = scriptable(variable["name"], label="machine variable name")
        value = scriptable(variable["value"], label=f"machine variable {name}")
        lines += [
            f"$Held = (@((Get-ItemProperty -LiteralPath '{MACHINE_ENVIRONMENT_KEY}').'{name}') "
            "-join '')",
            f"Emit '{machine_variable_check_id(variable)}' ($Held -ceq '{value}') "
            f"('the machine environment holds {name}=' + $Held + '; the roster says {value}')",
        ]
    return lines


__all__ = [
    "MACHINE_ENVIRONMENT_KEY",
    "machine_variable_check_id",
    "render_machine_environment_check_lines",
    "render_machine_environment_lines",
    "render_machine_environment_parameters",
]
