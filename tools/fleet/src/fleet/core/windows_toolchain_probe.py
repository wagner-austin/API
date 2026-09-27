"""The PowerShell toolchain probe: what a Windows node has installed.

Split out of :mod:`fleet.core.dialect_windows` when the ``cxx`` line took
that module past the 600-line ceiling (MCPs board task 3f19c136). It is one
constant with its own history, and the dialect hands it out unchanged
through :meth:`fleet.core.dialect_windows.WindowsDialect.toolchain_probe_script`.
It is committed as a render under ``rendered/`` and executed by
``tests/pester/rendered-dialect-toolchain.Tests.ps1`` with stand-in tools on
a PATH the suite lays out (MCPs board task d69786fa).
"""

from __future__ import annotations

#: The toolchain probe, verbatim.
#:
#: UNDER THE STRICT HEADER, SO NOTHING IS ASKED WITH ``SilentlyContinue``. A
#: tool is found by walking ``PATH`` and ``PATHEXT`` as cmd.exe resolves a
#: name, which answers '' for an absent tool where ``Get-Command`` would
#: throw, and every tool is run through cmd.exe with cmd's own ``2>&1``: a
#: native's stderr redirected by PowerShell 5.1 is an error record, which
#: ``Stop`` would end the probe on. ``cmd.exe`` and ``vswhere.exe`` are
#: parameters whose defaults are plain strings, so the suite passes a
#: stand-in vswhere.
#:
#: ``--version`` is asked of each tool and the first line kept, because git
#: and poetry both print several. A tool that is present but declines to
#: answer yields an empty version rather than a failure: absence and silence
#: are different states, and only the first stops a dispatch.
#:
#: ``pip`` is the one manager that is not an executable on PATH but a module
#: of the interpreter, so it is asked as ``python -m pip --version`` of the
#: same interpreter the tool loop reported, and only when there is one. Its
#: exit decides it: a python whose pip is missing exits non-zero and reads
#: absent. The output is collected whole and the first line taken afterwards,
#: never by stopping the pipeline early: under PowerShell 5.1 that reported
#: the native process as exit -1, and a present pip read as absent (measured
#: on the hub 2026-09-20).
#:
#: A ``python`` that resolves under ``Microsoft\\WindowsApps`` is reported
#: ABSENT. On a node whose PATH has no real interpreter ahead of it, that is
#: the App Execution Alias stub, which answers ``--version`` with "Python was
#: not found; run without arguments to install from the Microsoft Store" and
#: exit 9009. Read as present, that sentence became the version, the node was
#: offered no install, and every slime check lavender took died at its
#: Makefile's first python call (fleet board task e62c8120, 2026-09-22/23).
#: The real Store build lives under the same directory and is refused too,
#: deliberately: it sandboxes ``%LOCALAPPDATA%`` writes and broke poetry's
#: venv on sedona (:mod:`fleet.contracts.toolchain`).
#:
#: ``cargo`` and ``cxx`` are asked though no build requires them: each answer
#: is compared with the node's declaration (:mod:`fleet.contracts.capability`).
#: ``cxx`` is what node-gyp's find-visualstudio needs, the VC tools component,
#: read from vswhere as its installationVersion and collected whole, for the
#: same PowerShell 5.1 reason as pip. A node without vswhere, or whose vswhere
#: finds no VC tools (sedona on 2026-09-27: the installer present, the
#: component absent), reports ``cxx=no=``. ``docker`` is always ``no`` here:
#: the capability is the execution suite's rootless daemon under its own
#: Linux user (MCPs board task 6c4516af), which no Windows node carries,
#: and Docker Desktop is exactly the kind of shared daemon it excludes.
TOOLCHAIN_PROBE_SCRIPT = r"""param(
    [string]$Cmd = "$env:SystemRoot\System32\cmd.exe",
    [string]$VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
function Find-Tool {
    param([string]$Name)
    foreach ($directory in @($env:PATH -split ';' | Where-Object { $_ -ne '' })) {
        foreach ($extension in @($env:PATHEXT -split ';' | Where-Object { $_ -ne '' })) {
            $candidate = [System.IO.Path]::Combine($directory.Trim('"'), "$Name$extension")
            if ([System.IO.File]::Exists($candidate)) {
                return $candidate
            }
        }
    }
    return ''
}
function Invoke-Answer {
    param([string]$Shell, [string]$Path, [string]$Arguments)
    $lines = & $Shell /d /s /c "`"$Path`" $Arguments 2>&1"
    $succeeded = $LASTEXITCODE -eq 0
    $first = [string](@(@($lines) + '') | Select-Object -First 1)
    return [pscustomobject]@{ Succeeded = $succeeded; First = $first.Trim() }
}
$python = Find-Tool 'python'
if ($python -like '*\Microsoft\WindowsApps\*') {
    $python = ''
}
foreach ($tool in @('python', 'poetry', 'git', 'make', 'node', 'tar', 'cargo', 'winget', 'choco')) {
    $found = $python
    if ($tool -ne 'python') {
        $found = Find-Tool $tool
    }
    if ($found -ne '') {
        "$tool=yes=" + (Invoke-Answer $Cmd $found '--version').First
    } else {
        "$tool=no="
    }
}
$pip = 'pip=no='
if ($python -ne '') {
    $answer = Invoke-Answer $Cmd $python '-m pip --version'
    if ($answer.Succeeded) {
        $pip = "pip=yes=$($answer.First)"
    }
}
$pip
$query = '-products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 ' +
    '-property installationVersion'
$vc = ''
if ([System.IO.File]::Exists($VsWhere)) {
    $vc = (Invoke-Answer $Cmd $VsWhere $query).First
}
if ($vc -ne '') {
    "cxx=yes=$vc"
} else {
    'cxx=no='
}
'docker=no='
"""


__all__ = ["TOOLCHAIN_PROBE_SCRIPT"]
