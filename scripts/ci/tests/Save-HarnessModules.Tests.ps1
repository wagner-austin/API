Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# scripts/ci/Save-HarnessModules.ps1 (MCPs board task d69786fa, A3). Every
# case runs the real script against a real git repository shaped like MCPs,
# whose requirements file changes after the commit API pins, so what is
# proved is that the PINNED commit's versions are saved, not main's. Only
# Save-Module is a stand-in, a scriptblock that records what it was asked,
# because the real one downloads from the PowerShell Gallery.

BeforeAll {
    $script:entry = Join-Path (Split-Path -Parent $PSScriptRoot) 'Save-HarnessModules.ps1'

    function Invoke-TestGit {
        param([string]$Repo, [string[]]$Arguments)
        $output = & git -C $Repo -c user.name=test -c user.email=test@example.invalid @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "TEST_GIT_EXITED: git $($Arguments -join ' ') exited $LASTEXITCODE"
        }
        return $output
    }

    function Initialize-PinnedWorld {
        param([string]$Pin)
        $root = Join-Path $TestDrive ('world-' + [guid]::NewGuid().ToString('N'))
        $mcps = Join-Path $root 'MCPs'
        $api = Join-Path $root 'API'
        $requirements = Join-Path $mcps 'scripts\powershell-requirements.psd1'
        [void][System.IO.Directory]::CreateDirectory((Split-Path -Parent $requirements))
        [void][System.IO.Directory]::CreateDirectory((Join-Path $api '.githooks'))
        [System.IO.File]::WriteAllText($requirements, "@{`r`n    Pester = '5.7.1'`r`n    PSScriptAnalyzer = '1.25.0'`r`n}`r`n")
        [void](Invoke-TestGit $mcps @('init', '--quiet'))
        [void](Invoke-TestGit $mcps @('add', '.'))
        [void](Invoke-TestGit $mcps @('commit', '--quiet', '-m', 'pinned'))
        $pinned = [string](Invoke-TestGit $mcps @('rev-parse', 'HEAD'))
        # main moves on after the pin: a later harness wanting later modules.
        [System.IO.File]::WriteAllText($requirements, "@{`r`n    Pester = '9.9.9'`r`n}`r`n")
        [void](Invoke-TestGit $mcps @('commit', '--quiet', '-am', 'later'))
        $written = $pinned
        if ($PSBoundParameters.ContainsKey('Pin')) {
            $written = $Pin
        }
        [System.IO.File]::WriteAllText((Join-Path $api '.githooks\mcps-harness.sha'), "$written`n")
        return [pscustomobject]@{ Api = $api; Mcps = $mcps; Modules = Join-Path $root 'modules'; Pinned = $pinned }
    }

    function Initialize-SaveRecorder {
        $calls = [System.Collections.Generic.List[string]]::new()
        $record = { param([string]$Name, [string]$Version, [string]$Path, [string]$From) $calls.Add("$Name $Version $From $Path") }.GetNewClosure()
        return [pscustomobject]@{ Calls = $calls; Save = $record }
    }

    function Invoke-TestEntry {
        param([hashtable]$Parameters)
        $global:LASTEXITCODE = 0
        & $script:entry @Parameters 6>$null
        if ($LASTEXITCODE -ne 0) {
            throw "TEST_ENTRY_EXITED: $script:entry exited $LASTEXITCODE"
        }
    }

    function Get-EntryParameter {
        param($World, $Recorder)
        return @{ RepoRoot = $World.Api; McpsRoot = $World.Mcps; ModuleRoot = $World.Modules; SaveModule = $Recorder.Save }
    }
}

Describe 'Saving the modules the pinned harness imports' {
    It 'saves each module at the version the pinned commit requires, not the one main has moved to' {
        $world = Initialize-PinnedWorld
        $recorder = Initialize-SaveRecorder
        Invoke-TestEntry (Get-EntryParameter $world $recorder)
        @($recorder.Calls | Sort-Object) | Should -Be @(
            "Pester 5.7.1 PSGallery $($world.Modules)",
            "PSScriptAnalyzer 1.25.0 PSGallery $($world.Modules)"
        )
        [System.IO.Directory]::Exists($world.Modules) | Should -BeTrue
    }
    It 'saves through the real Save-Module by default, which refuses a repository that is not registered by name' {
        # The default's only effect is Save-Module's own: an unregistered
        # repository is refused locally, so no case reaches the gallery.
        $world = Initialize-PinnedWorld
        $parameters = @{ RepoRoot = $world.Api; McpsRoot = $world.Mcps; ModuleRoot = $world.Modules; Repository = 'NoSuchRepository-test' }
        { Invoke-TestEntry $parameters } | Should -Throw `
            -ExpectedMessage "Unable to find repository 'NoSuchRepository-test'. Use Get-PSRepository to see all available repositories." `
            -ErrorId 'SourceNotFound,Microsoft.PowerShell.PackageManagement.Cmdlets.GetPackageSource'
    }
    It 'refuses a pin that is not a commit id, before asking git or saving anything' {
        $world = Initialize-PinnedWorld -Pin 'main'
        $recorder = Initialize-SaveRecorder
        { Invoke-TestEntry (Get-EntryParameter $world $recorder) } |
            Should -Throw "HARNESS_PIN_INVALID: .githooks/mcps-harness.sha holds 'main', not a 40-character lowercase commit id"
        $recorder.Calls.Count | Should -Be 0
    }
    It 'refuses a pinned commit the MCPs clone does not hold, with git''s exit code' {
        $absent = '0' * 40
        $world = Initialize-PinnedWorld -Pin $absent
        $recorder = Initialize-SaveRecorder
        { Invoke-TestEntry (Get-EntryParameter $world $recorder) } |
            Should -Throw "HARNESS_PIN_UNREADABLE: git show of scripts/powershell-requirements.psd1 at $absent in $($world.Mcps) exited 128"
        $recorder.Calls.Count | Should -Be 0
    }
}
