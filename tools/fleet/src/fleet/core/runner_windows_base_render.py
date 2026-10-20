"""The Windows runner host's base, as lines of its install and its audit.

MCPs board task 1e9ab738 (:mod:`fleet.contracts.runner_windows_base` says
why each setting exists). :mod:`fleet.core.runner_windows_render` places
these lines: the parameters in each script's param block, the install's
body before its first runner, the audit's rows before its first runner's.

EVERY EFFECT IS A PARAMETER, as in the runners' own lines: the two registry
keys, git, and the scope the machine variables are read and written in, so
the Pester suite runs the committed install against scratch keys under
HKCU, a git stand-in and the Process scope, and nothing on the test machine
changes. The execution policy is written to the registry value
``Set-ExecutionPolicy -Scope LocalMachine`` writes, directly: under the
install's own Process-scope Bypass the cmdlet reports the LocalMachine
change as overridden and, with ErrorActionPreference Stop, throws after
succeeding (measured on lavender, before 0531d9be2 deleted that base).

A MACHINE VARIABLE REACHES A RUNNER ONLY WHEN ITS SERVICE STARTS: the
listener's environment is read then, and each job's worker inherits it. So
the install sets ``$restart`` when it writes one, and
:func:`fleet.core.runner_windows_render.render_install_script` restarts each
running runner service after it.
"""

from __future__ import annotations

from platform_core.json_utils import dump_json_str

from fleet.contracts.runner_windows_base import WindowsHostBase
from fleet.core.runner_audit_types import ExpectedCheck
from fleet.core.script_values import scriptable

#: The registry key whose ``ExecutionPolicy`` value is the LocalMachine
#: execution policy.
EXECUTION_POLICY_KEY = "HKLM:\\SOFTWARE\\Microsoft\\PowerShell\\1\\ShellIds\\Microsoft.PowerShell"

#: The registry key whose ``LongPathsEnabled`` value lifts Win32's 260
#: character path limit.
LONG_PATHS_KEY = "HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem"

#: The long paths row's id.
LONG_PATHS_CHECK_ID = "long-paths"

#: git's system core.longpaths row's id.
GIT_LONG_PATHS_CHECK_ID = "git-longpaths"

#: Why the policy row matters.
POLICY_REASON = (
    "a runner's PowerShell steps run scripts from _temp, which the Restricted default refuses: "
    "every job loki took on 2026-10-20 failed at actions/setup-python's setup.ps1"
)

#: Why the long paths rows matter.
LONG_PATHS_REASON = (
    "MCPs CI's supervisor Pester case extracts a tree whose paths pass 260 characters, and "
    "CreateDirectory and git's checkout fail there without them"
)


def policy_check_id(base: WindowsHostBase) -> str:
    """The execution policy row's id, naming the policy it holds the host to.

    Args:
        base: The host's base.

    Returns:
        ``execution-policy:LocalMachine:<policy>``.
    """
    return f"execution-policy:LocalMachine:{base['execution_policy']}"


def _variables_json(base: WindowsHostBase) -> str:
    """The machine variables as the JSON string both scripts decode.

    Args:
        base: The host's base.

    Returns:
        A JSON array of ``{"Name", "Value"}`` objects.

    Raises:
        ValueError: When a name or value could not be embedded verbatim
            (:func:`fleet.core.script_values.scriptable`).
    """
    return dump_json_str(
        [
            {
                "Name": scriptable(each["name"], label="machine variable"),
                "Value": scriptable(each["value"], label=f"machine variable {each['name']}"),
            }
            for each in base["machine_environment"]
        ]
    )


def parameter_lines(base: WindowsHostBase) -> list[str]:
    """The base's param-block lines, each ending in a comma.

    The install and the audit take the same ones: one writes each setting
    and the other reads it.

    Args:
        base: The host's base.

    Returns:
        The lines, before the script's own parameters.

    Raises:
        ValueError: As :func:`_variables_json` says. The policy is one of
            the decoder's closed set, which every member of embeds.
    """
    return [
        f"    [string]$Policy = '{base['execution_policy']}',",
        f"    [string]$PolicyKey = '{EXECUTION_POLICY_KEY}',",
        f"    [string]$FileSystemKey = '{LONG_PATHS_KEY}',",
        "    [string]$Git = 'git',",
        "    [string]$VariableScope = 'Machine',",
        f"    [string]$MachineEnvironment = '{_variables_json(base)}',",
    ]


#: The install's base lines, run before the first runner; ``$restart`` is
#: true afterwards when a machine variable was written.
INSTALL_LINES: tuple[str, ...] = (
    "if ([string](Get-Item -LiteralPath $PolicyKey).GetValue('ExecutionPolicy') -ne $Policy) {",
    "    Set-ItemProperty -LiteralPath $PolicyKey -Name ExecutionPolicy -Value $Policy",
    '    "base: set the LocalMachine execution policy to $Policy"',
    "}",
    "if ((Get-Item -LiteralPath $FileSystemKey).GetValue('LongPathsEnabled') -ne 1) {",
    "    Set-ItemProperty -LiteralPath $FileSystemKey -Name LongPathsEnabled -Value 1 -Type DWord",
    "    'base: enabled Win32 long paths'",
    "}",
    # git exits 1 for a key that is not set, which is the stock answer.
    "$held = & $Git config --system --get core.longpaths",
    "if ($LASTEXITCODE -gt 1) {",
    '    throw "RUNNER_BASE_FAILED: git config --system --get core.longpaths exited $LASTEXITCODE"',
    "}",
    "if ((@($held) -join '') -ne 'true') {",
    "    & $Git config --system core.longpaths true",
    "    if ($LASTEXITCODE -ne 0) {",
    '        throw "RUNNER_BASE_FAILED: git config --system core.longpaths exited $LASTEXITCODE"',
    "    }",
    "    'base: set git core.longpaths for the system'",
    "}",
    "$restart = $false",
    "foreach ($variable in ($MachineEnvironment | ConvertFrom-Json)) {",
    "    if ([Environment]::GetEnvironmentVariable($variable.Name, $VariableScope) "
    "-cne $variable.Value) {",
    "        [Environment]::SetEnvironmentVariable($variable.Name, $variable.Value, "
    "$VariableScope)",
    '        "base: set the machine variable $($variable.Name)"',
    "        $restart = $true",
    "    }",
    "}",
)

#: The audit's base rows, in :func:`expected_checks` order; ``Get-Answer``
#: is the audit's own reader of a native's exit code and words.
AUDIT_LINES: tuple[str, ...] = (
    "$heldPolicy = [string](Get-Item -LiteralPath $PolicyKey).GetValue('ExecutionPolicy')",
    "if ($heldPolicy -eq $Policy) {",
    '    "CHECK execution-policy:LocalMachine:$Policy OK"',
    "} else {",
    '    "CHECK execution-policy:LocalMachine:$Policy DRIFT the LocalMachine ExecutionPolicy '
    'value is: $heldPolicy"',
    "}",
    "$heldLongPaths = (Get-Item -LiteralPath $FileSystemKey).GetValue('LongPathsEnabled')",
    "if ($heldLongPaths -eq 1) {",
    f"    'CHECK {LONG_PATHS_CHECK_ID} OK'",
    "} else {",
    f'    "CHECK {LONG_PATHS_CHECK_ID} DRIFT LongPathsEnabled is: $heldLongPaths"',
    "}",
    "$gitLongPaths = Get-Answer $Git @('config', '--system', '--get', 'core.longpaths')",
    "if ($gitLongPaths.Code -eq 0 -and $gitLongPaths.Text -eq 'true') {",
    f"    'CHECK {GIT_LONG_PATHS_CHECK_ID} OK'",
    "} else {",
    f'    "CHECK {GIT_LONG_PATHS_CHECK_ID} DRIFT git config --system --get core.longpaths exited '
    '$($gitLongPaths.Code): $($gitLongPaths.Text)"',
    "}",
    "foreach ($variable in ($MachineEnvironment | ConvertFrom-Json)) {",
    "    $heldValue = [string][Environment]::GetEnvironmentVariable($variable.Name, "
    "$VariableScope)",
    "    if ($heldValue -ceq $variable.Value) {",
    '        "CHECK machine-env:$($variable.Name) OK"',
    "    } else {",
    '        "CHECK machine-env:$($variable.Name) DRIFT the machine variable holds: $heldValue"',
    "    }",
    "}",
)


def expected_checks(base: WindowsHostBase) -> list[ExpectedCheck]:
    """Every base row the audit prints, in order.

    Args:
        base: The host's base.

    Returns:
        The policy, long paths and git rows, then one per machine variable.
    """
    variables = [
        ExpectedCheck(check_id=f"machine-env:{each['name']}", reason=each["reason"])
        for each in base["machine_environment"]
    ]
    return [
        ExpectedCheck(check_id=policy_check_id(base), reason=POLICY_REASON),
        ExpectedCheck(check_id=LONG_PATHS_CHECK_ID, reason=LONG_PATHS_REASON),
        ExpectedCheck(check_id=GIT_LONG_PATHS_CHECK_ID, reason=LONG_PATHS_REASON),
        *variables,
    ]


__all__ = [
    "AUDIT_LINES",
    "EXECUTION_POLICY_KEY",
    "GIT_LONG_PATHS_CHECK_ID",
    "INSTALL_LINES",
    "LONG_PATHS_CHECK_ID",
    "LONG_PATHS_KEY",
    "LONG_PATHS_REASON",
    "POLICY_REASON",
    "expected_checks",
    "parameter_lines",
    "policy_check_id",
]
