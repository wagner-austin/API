Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# scripts/ci/Save-HarnessModules.ps1 (MCPs board tasks d69786fa, A3, and
# da8b4f11). The fixture cases run the real script against a real git
# repository shaped like MCPs, whose requirements file AND installer change
# after the commit API pins, so what is proved is that the PINNED commit's
# installer runs over the PINNED commit's pins, not main's. That installer is
# a recording stand-in, because MCPs' real one downloads from the PowerShell
# Gallery. The last case runs the real one: API's own pin, read out of the
# MCPs clone beside this checkout, refusing a repository that is not
# registered, which Save-Module does before any download.

BeforeAll {
    $script:entry = Join-Path (Split-Path -Parent $PSScriptRoot) 'Save-HarnessModules.ps1'
    $script:apiRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
    # The pinned requirements carry a byte outside ASCII, an e-acute in a
    # comment, so a copy decoded through the console's code page shows.
    $script:pinnedText = "@{ Pester = '5.7.1' } # caf$([char]0xE9)"

    function Invoke-TestGit {
        param([string]$Repo, [string[]]$Arguments)
        $output = & git -C $Repo -c user.name=test -c user.email=test@example.invalid @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "TEST_GIT_EXITED: git $($Arguments -join ' ') exited $LASTEXITCODE"
        }
        return $output
    }

    function Get-RecordingInstaller {
        # Writes what it was given, and the pins' text, beside the modules.
        param([string]$Word, [int]$ExitCode)
        return @"
param([string]`$RequirementsPath, [string]`$ModuleRoot, [string]`$Repository)
[void][System.IO.Directory]::CreateDirectory(`$ModuleRoot)
[System.IO.File]::WriteAllText((Join-Path `$ModuleRoot 'record.txt'), "$Word|`$Repository|`$RequirementsPath|" + [System.IO.File]::ReadAllText(`$RequirementsPath))
exit $ExitCode
"@
    }

    function Initialize-PinnedWorld {
        param([string]$Pin, [int]$ExitCode = 0)
        $root = Join-Path $TestDrive ('world-' + [guid]::NewGuid().ToString('N'))
        $mcps = Join-Path $root 'MCPs'
        $api = Join-Path $root 'API'
        $requirements = Join-Path $mcps 'scripts\powershell-requirements.psd1'
        $installer = Join-Path $mcps 'scripts\ps-harness\Install-HarnessModule.ps1'
        [void][System.IO.Directory]::CreateDirectory((Split-Path -Parent $installer))
        [void][System.IO.Directory]::CreateDirectory((Join-Path $api '.githooks'))
        [System.IO.File]::WriteAllText($requirements, $script:pinnedText, [System.Text.UTF8Encoding]::new($false))
        [System.IO.File]::WriteAllText($installer, (Get-RecordingInstaller 'pinned' $ExitCode))
        [void](Invoke-TestGit $mcps @('init', '--quiet'))
        [void](Invoke-TestGit $mcps @('add', '.'))
        [void](Invoke-TestGit $mcps @('commit', '--quiet', '-m', 'pinned'))
        $pinned = [string](Invoke-TestGit $mcps @('rev-parse', 'HEAD'))
        # main moves on after the pin: later modules, and a later installer.
        [System.IO.File]::WriteAllText($requirements, "@{ Pester = '9.9.9' }")
        [System.IO.File]::WriteAllText($installer, (Get-RecordingInstaller 'later' 0))
        [void](Invoke-TestGit $mcps @('commit', '--quiet', '-am', 'later'))
        $written = $pinned
        if ($PSBoundParameters.ContainsKey('Pin')) {
            $written = $Pin
        }
        [System.IO.File]::WriteAllText((Join-Path $api '.githooks\mcps-harness.sha'), "$written`n")
        $modules = Join-Path $root 'modules'
        return [pscustomobject]@{
            Parameters = @{ RepoRoot = $api; McpsRoot = $mcps; ModuleRoot = $modules }
            Record = Join-Path $modules 'record.txt'
        }
    }

    function Invoke-TestEntry {
        param([hashtable]$Parameters)
        $global:LASTEXITCODE = 0
        & $script:entry @Parameters 6>$null
        if ($LASTEXITCODE -ne 0) {
            throw "TEST_ENTRY_EXITED: $script:entry exited $LASTEXITCODE"
        }
    }
}

Describe 'Saving the modules the pinned harness imports' {
    It 'runs the pinned commit''s installer over the pinned commit''s pins byte for byte, not main''s, under a fresh console''s code page, and leaves nothing staged' {
        $world = Initialize-PinnedWorld
        # Code page 437 is what a fresh Windows PowerShell console decodes a
        # native command's output with; a piped copy mangled the e-acute.
        $encoding = [Console]::OutputEncoding
        [Console]::OutputEncoding = [System.Text.Encoding]::GetEncoding(437)
        try {
            Invoke-TestEntry $world.Parameters
        }
        finally {
            [Console]::OutputEncoding = $encoding
        }
        $said = [System.IO.File]::ReadAllText($world.Record) -split '\|'
        $said[0] | Should -BeExactly 'pinned'
        $said[1] | Should -BeExactly 'PSGallery'
        $said[3] | Should -BeExactly $script:pinnedText
        # The staged copies are gone once the installer has run.
        [System.IO.File]::Exists($said[2]) | Should -BeFalse
    }
    It 'passes the repository it is given to the installer' {
        $world = Initialize-PinnedWorld
        $parameters = $world.Parameters
        $parameters.Repository = 'Mirror'
        Invoke-TestEntry $parameters
        ([System.IO.File]::ReadAllText($world.Record) -split '\|')[1] | Should -BeExactly 'Mirror'
    }
    It 'refuses a pin that is not a commit id, before asking git or installing anything' {
        $world = Initialize-PinnedWorld -Pin 'main'
        { Invoke-TestEntry $world.Parameters } |
            Should -Throw "HARNESS_PIN_INVALID: .githooks/mcps-harness.sha holds 'main', not a 40-character lowercase commit id"
        [System.IO.File]::Exists($world.Record) | Should -BeFalse
    }
    It 'refuses a pinned commit the MCPs clone does not hold, with git''s exit code' {
        $absent = '0' * 40
        $world = Initialize-PinnedWorld -Pin $absent
        { Invoke-TestEntry $world.Parameters } |
            Should -Throw "HARNESS_PIN_UNREADABLE: git show of scripts/powershell-requirements.psd1 at $absent in $($world.Parameters.McpsRoot) exited 128"
        [System.IO.File]::Exists($world.Record) | Should -BeFalse
    }
    It 'refuses an installer that exits non-zero, naming the pinned commit' {
        $world = Initialize-PinnedWorld -ExitCode 3
        $pin = ([System.IO.File]::ReadAllText((Join-Path $world.Parameters.RepoRoot '.githooks\mcps-harness.sha'))).Trim()
        { Invoke-TestEntry $world.Parameters } |
            Should -Throw "HARNESS_INSTALLER_EXITED: the installer at MCPs $pin exited 3"
    }
    It 'runs MCPs'' real installer at API''s own pin, which refuses a repository that is not registered' {
        $modules = Join-Path $TestDrive 'real-modules'
        $parameters = @{
            RepoRoot = $script:apiRoot
            McpsRoot = Join-Path (Split-Path -Parent $script:apiRoot) 'MCPs'
            ModuleRoot = $modules
            Repository = 'NoSuchRepository-test'
        }
        { Invoke-TestEntry $parameters } | Should -Throw `
            -ExpectedMessage "Unable to find repository 'NoSuchRepository-test'. Use Get-PSRepository to see all available repositories." `
            -ErrorId 'SourceNotFound,Microsoft.PowerShell.PackageManagement.Cmdlets.GetPackageSource'
        # The installer created the root before it asked the gallery.
        [System.IO.Directory]::Exists($modules) | Should -BeTrue
    }
}
