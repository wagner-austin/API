<#
.SYNOPSIS
    Save the PowerShell modules the pinned MCPs harness imports (API's powershell CI job).
.DESCRIPTION
    Reads the MCPs commit API pins in .githooks/mcps-harness.sha, reads that
    commit's scripts/powershell-requirements.psd1 out of the MCPs clone with
    git, and saves each module at the version it pins under ModuleRoot. The
    harness imports exactly those versions from the pinned commit, so the
    modules are read from that commit too, never from MCPs' main (MCPs board
    task d69786fa, A3).
.PARAMETER RepoRoot
    API's checkout, holding .githooks/mcps-harness.sha.
.PARAMETER McpsRoot
    The MCPs clone beside it, with the pinned commit in its history.
.PARAMETER ModuleRoot
    The directory the modules are saved under, created when absent.
.PARAMETER Git
    git.
.PARAMETER Repository
    The registered repository each module is saved from.
.PARAMETER SaveModule
    Saves one module: its name, its exact version, the directory and the
    repository. The default is Save-Module.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$RepoRoot,
    [Parameter(Mandatory)][string]$McpsRoot,
    [Parameter(Mandatory)][string]$ModuleRoot,
    [string]$Git = 'git',
    [string]$Repository = 'PSGallery',
    [scriptblock]$SaveModule = {
        param([string]$Name, [string]$Version, [string]$Path, [string]$From)
        Save-Module -Name $Name -RequiredVersion $Version -Repository $From -Path $Path
    }
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$pin = [System.IO.File]::ReadAllText((Join-Path $RepoRoot '.githooks\mcps-harness.sha')).Trim()
if ($pin -notmatch '^[0-9a-f]{40}$') {
    throw "HARNESS_PIN_INVALID: .githooks/mcps-harness.sha holds '$pin', not a 40-character lowercase commit id"
}
$text = & $Git -C $McpsRoot show "${pin}:scripts/powershell-requirements.psd1"
if ($LASTEXITCODE -ne 0) {
    throw "HARNESS_PIN_UNREADABLE: git show of scripts/powershell-requirements.psd1 at $pin in $McpsRoot exited $LASTEXITCODE"
}
$requirementsPath = Join-Path ([System.IO.Path]::GetTempPath()) ('mcps-requirements-' + [guid]::NewGuid().ToString('N') + '.psd1')
[System.IO.File]::WriteAllLines($requirementsPath, [string[]]$text)
$requirements = Import-PowerShellDataFile -LiteralPath $requirementsPath
[System.IO.File]::Delete($requirementsPath)
[void][System.IO.Directory]::CreateDirectory($ModuleRoot)
foreach ($requirement in $requirements.GetEnumerator()) {
    & $SaveModule $requirement.Key $requirement.Value $ModuleRoot $Repository
}
Write-Information "Saved $($requirements.Count) module(s) pinned by MCPs $pin under $ModuleRoot." -InformationAction Continue
