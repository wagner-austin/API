"""The PowerShell toolchain probe: what a Windows node has installed.

Split out of :mod:`fleet.core.dialect_windows` when the ``cxx`` line took
that module past the 600-line ceiling (MCPs board task 3f19c136). It is one
constant with its own history, and the dialect hands it out unchanged
through :meth:`fleet.core.dialect_windows.WindowsDialect.toolchain_probe_script`.
"""

from __future__ import annotations

#: The toolchain probe, verbatim.
#:
#: ``--version`` is asked of each tool and the first line kept, because git
#: and poetry both print several. A tool that is present but declines to
#: answer yields an empty version rather than a failure: absence and silence
#: are different states, and only the first stops a dispatch.
#:
#: ``pip`` is the one manager that is not an executable on PATH but a module
#: of the interpreter, so it is asked as ``python -m pip --version`` behind the
#: same ``$python`` the tool loop reported: without the guard a node with no Python
#: would print a CommandNotFound error where a line was expected. Its output
#: is collected whole and the first line taken afterwards, NOT piped through
#: ``Select-Object -First 1`` like the others: that stops the pipeline early,
#: PowerShell 5.1 then reports the native process as exit -1, and a present
#: pip read as absent (measured on the hub 2026-09-20).
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
#: component absent), reports ``cxx=no=``.
TOOLCHAIN_PROBE_SCRIPT = """\
$python = Get-Command python -ErrorAction SilentlyContinue
if ($python -and $python.Source -like '*\\Microsoft\\WindowsApps\\*') { $python = $null }
foreach ($tool in @('python','poetry','git','make','node','tar','cargo','winget','choco')) {
  $found = $python
  if ($tool -ne 'python') { $found = Get-Command $tool -ErrorAction SilentlyContinue }
  if ($found) {
    $raw = (& $tool --version 2>&1 | Select-Object -First 1)
    $text = ($raw | Out-String).Trim() -replace '[\\r\\n]', ' '
    "$tool=yes=$text"
  } else {
    "$tool=no="
  }
}
if ($python) {
  $lines = @(& python -m pip --version 2>&1)
  if ($LASTEXITCODE -eq 0) {
    "pip=yes=" + (($lines[0] | Out-String).Trim() -replace '[\\r\\n]', ' ')
  } else {
    "pip=no="
  }
} else {
  "pip=no="
}
$installer = Join-Path ([Environment]::GetFolderPath('ProgramFilesX86')) 'Microsoft Visual Studio'
$vswhere = Join-Path $installer 'Installer\\vswhere.exe'
$vcTools = 'Microsoft.VisualStudio.Component.VC.Tools.x86.x64'
$vc = @()
if (Test-Path -LiteralPath $vswhere) {
  $vc = @(& $vswhere -products * -requires $vcTools -property installationVersion)
}
if ($vc.Count -gt 0 -and $vc[0]) {
  "cxx=yes=" + (($vc[0] | Out-String).Trim())
} else {
  "cxx=no="
}
"""


__all__ = ["TOOLCHAIN_PROBE_SCRIPT"]
