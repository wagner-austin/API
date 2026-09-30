<#
.SYNOPSIS
    Save the PowerShell modules the pinned MCPs harness imports (API's powershell CI job).
.DESCRIPTION
    Reads the MCPs commit API pins in .githooks/mcps-harness.sha, reads that
    commit's scripts/powershell-requirements.psd1 AND its
    scripts/ps-harness/Install-HarnessModule.ps1 out of the MCPs clone with
    git, and runs that installer over those pins into ModuleRoot. The harness
    imports exactly those versions from the pinned commit, so the modules are
    read from that commit too, never from MCPs' main (MCPs board task
    d69786fa, A3).

    THE SAVING IS MCPs', NOT A COPY (MCPs board task da8b4f11). The installer
    is the one MCPs CI and `make install-ps-modules` run (MCPs board task
    e5a50aa8); this script used to restate its Save-Module loop, without the
    -ErrorAction Stop that makes an unknown repository a failure under
    PowerShellGet 1.0.0.1. Read at the pinned commit, it moves with the pin
    like the harness does, so the pin must name a commit that carries it.
.PARAMETER RepoRoot
    API's checkout, holding .githooks/mcps-harness.sha.
.PARAMETER McpsRoot
    The MCPs clone beside it, with the pinned commit in its history.
.PARAMETER ModuleRoot
    The directory the modules are saved under, created when absent.
.PARAMETER Git
    git.
.PARAMETER Repository
    The registered repository each module is saved from, passed to the
    installer.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory)][string]$RepoRoot,
    [Parameter(Mandatory)][string]$McpsRoot,
    [Parameter(Mandatory)][string]$ModuleRoot,
    [string]$Git = 'git',
    [string]$Repository = 'PSGallery'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Read-PinnedFile {
    <#
    .SYNOPSIS
        Write one file of the pinned MCPs commit to a directory, and return its path.
    .PARAMETER GitCommand
        git.
    .PARAMETER Repo
        The MCPs clone.
    .PARAMETER Pin
        The pinned commit.
    .PARAMETER RelativePath
        The file's path in MCPs.
    .PARAMETER Directory
        Where the copy is written, under the file's own name.
    #>
    param(
        [Parameter(Mandatory)][string]$GitCommand,
        [Parameter(Mandatory)][string]$Repo,
        [Parameter(Mandatory)][string]$Pin,
        [Parameter(Mandatory)][string]$RelativePath,
        [Parameter(Mandatory)][string]$Directory
    )
    $text = & $GitCommand -C $Repo show "${Pin}:$RelativePath"
    if ($LASTEXITCODE -ne 0) {
        throw "HARNESS_PIN_UNREADABLE: git show of $RelativePath at $Pin in $Repo exited $LASTEXITCODE"
    }
    $path = Join-Path $Directory (Split-Path -Leaf $RelativePath)
    [System.IO.File]::WriteAllLines($path, [string[]]$text)
    return $path
}

$pin = [System.IO.File]::ReadAllText((Join-Path $RepoRoot '.githooks\mcps-harness.sha')).Trim()
if ($pin -notmatch '^[0-9a-f]{40}$') {
    throw "HARNESS_PIN_INVALID: .githooks/mcps-harness.sha holds '$pin', not a 40-character lowercase commit id"
}
$staging = Join-Path ([System.IO.Path]::GetTempPath()) ('mcps-harness-modules-' + [guid]::NewGuid().ToString('N'))
[void][System.IO.Directory]::CreateDirectory($staging)
try {
    $requirements = Read-PinnedFile $Git $McpsRoot $pin 'scripts/powershell-requirements.psd1' $staging
    $installer = Read-PinnedFile $Git $McpsRoot $pin 'scripts/ps-harness/Install-HarnessModule.ps1' $staging
    & $installer -RequirementsPath $requirements -ModuleRoot $ModuleRoot -Repository $Repository
    if ($LASTEXITCODE -ne 0) {
        throw "HARNESS_INSTALLER_EXITED: the installer at MCPs $pin exited $LASTEXITCODE"
    }
}
finally {
    [System.IO.Directory]::Delete($staging, $true)
}
