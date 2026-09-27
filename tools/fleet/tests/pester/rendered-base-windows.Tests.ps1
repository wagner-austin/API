Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# The rebuild's Windows base stage, from each roster host's committed render
# under rendered/ (fleet.core.runner_base_render, MCPs board task d69786fa,
# A2). Every effect is a parameter, so each case runs the render as written
# with stand-in dism, msiexec, wsl and git that record their calls, scratch
# HKCU keys in place of the three HKLM ones, a user variable of its own in
# place of the machine PATH, .wslconfig under TestDrive, and the MSI served
# from a file:// URL whose digest the case computed. The roster's own values
# (features, WSL version, PATH entries, machine variables) are the render's
# defaults, read from its param block rather than restated here. The cases
# expect a host to declare a memory floor, a PATH entry and a machine
# variable, as lavender does; a host that declares none of one needs a case
# of its own, and fails here until it has one.

BeforeAll {
    . (Join-Path $PSScriptRoot 'rendered-fixtures.ps1')

    # Everything one case's run touches, laid out as a stock or a laid host.
    function Initialize-World {
        param(
            [string]$Name,
            [string]$FeatureState = 'Disabled', [int]$QueryExit = 0, [int]$EnableExit = 3010,
            [string]$WslAnswer = '', [int]$WslExit = 1, [int]$MsiExit = 3010,
            [string]$Policy = 'Restricted', [int]$LongPaths = 0,
            [string]$GitHeld = '', [int]$GitGetExit = 1, [int]$GitSetExit = 0,
            [string]$PathValue = '', [string]$EnvValue = '', [string]$Pin = ''
        )
        $root = Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        [void][System.IO.Directory]::CreateDirectory($root)
        $payload = Join-Path $root 'served.msi'
        [System.IO.File]::WriteAllText($payload, 'fleet rebuild test payload')
        $digest = (Get-FileHash -Algorithm SHA256 -LiteralPath $payload).Hash.ToLower()
        $registry = 'HKCU:\Software\fleet-test-' + [guid]::NewGuid().ToString('N')
        foreach ($key in 'Policy', 'FileSystem', 'Environment') {
            [void](New-Item -Path "$registry\$key" -Force)
        }
        Set-ItemProperty -LiteralPath "$registry\Policy" -Name ExecutionPolicy -Value $Policy
        Set-ItemProperty -LiteralPath "$registry\FileSystem" -Name LongPathsEnabled -Value $LongPaths -Type DWord
        if ($EnvValue -ne '') {
            $variable = @(Get-RenderedDefault $Name 'MachineVariables')[0] -split '=', 2
            Set-ItemProperty -LiteralPath "$registry\Environment" -Name $variable[0] -Value $EnvValue
        }
        $pathVariable = 'FLEET_TEST_PATH_' + [guid]::NewGuid().ToString('N')
        if ($PathValue -ne '') {
            [Environment]::SetEnvironmentVariable($pathVariable, $PathValue, 'User')
        }
        $dism = Initialize-Batch 'dism' @(
            'echo %* | findstr /c:"/get-featureinfo" >nul', 'if not errorlevel 1 goto query', "exit /b $EnableExit",
            ':query', "echo State : $FeatureState", "exit /b $QueryExit")
        $wslBody = @("exit /b $WslExit")
        if ($WslAnswer -ne '') {
            $wslBody = @("echo $WslAnswer") + $wslBody
        }
        $wsl = Initialize-Batch 'wsl' $wslBody
        $gitGet = @("exit /b $GitGetExit")
        if ($GitHeld -ne '') {
            $gitGet = @("echo $GitHeld") + $gitGet
        }
        $git = Initialize-Batch 'git' (@('if "%3"=="--get" goto get', "exit /b $GitSetExit", ':get') + $gitGet)
        $msiexec = Initialize-Batch 'msiexec' @("exit /b $MsiExit")
        $pin = @{ $true = $digest; $false = $Pin }[$Pin -eq '']
        $script:cleanup.Add([pscustomobject]@{ Registry = $registry; PathVariable = $pathVariable })
        return [pscustomobject]@{
            Registry = $registry; PathVariable = $pathVariable; Root = $root; Digest = $digest
            Dism = $dism; Wsl = $wsl; Git = $git; Msiexec = $msiexec
            Parameters = @{
                Scratch = (Join-Path $root 'scratch'); MsiUrl = ([Uri]$payload).AbsoluteUri; MsiSha256 = $pin
                PolicyKey = "$registry\Policy"; FileSystemKey = "$registry\FileSystem"; EnvironmentKey = "$registry\Environment"
                PathVariable = $pathVariable; PathScope = 'User'; WslConfigPath = (Join-Path $root '.wslconfig')
                Git = $git.Path; Dism = $dism.Path; Msiexec = $msiexec.Path; Wsl = $wsl.Path
            }
        }
    }

    function Invoke-Base {
        param([string]$Name, [object]$World)
        return [string[]]@(Invoke-Rendered $Name $World.Parameters)
    }

    $script:cleanup = [System.Collections.Generic.List[object]]::new()
}

AfterAll {
    foreach ($world in $script:cleanup) {
        Remove-Item -LiteralPath $world.Registry -Recurse -Force
        [Environment]::SetEnvironmentVariable($world.PathVariable, $null, 'User')
    }
}

Describe 'The Windows base for <_>' -ForEach @(Get-ChildItem -LiteralPath (Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'rendered') -Filter 'base-windows-*.ps1' | ForEach-Object { $_.BaseName }) {
    BeforeAll {
        $script:name = $_
        $script:features = [string[]]@(Get-RenderedDefault $script:name 'Features')
        $script:version = [string](Get-RenderedDefault $script:name 'WslVersion')
        $script:policy = [string](Get-RenderedDefault $script:name 'Policy')
        $script:entries = [string[]]@(Get-RenderedDefault $script:name 'PathEntries')
        $script:variable = @(Get-RenderedDefault $script:name 'MachineVariables')[0] -split '=', 2
        $script:msiFile = [string](Get-RenderedDefault $script:name 'MsiFile')
    }

    It 'lays a stock host: every feature, the WSL release, the policy, long paths, PATH, the machine variable and .wslconfig, then asks for a restart' {
        $world = Initialize-World $script:name
        $said = Invoke-Base $script:name $world
        $expected = @($script:features | ForEach-Object { "enabled Windows feature $_" }) + @(
            "installed WSL $($script:version)", "set the LocalMachine execution policy to $($script:policy)",
            'enabled Win32 long paths', 'set git core.longpaths for the system') +
            @($script:entries | ForEach-Object { "added to the machine PATH: $_" }) + @(
            "set the machine variable $($script:variable[0]) to $($script:variable[1])",
            'wrote .wslconfig; the ceiling applies when the VM next starts', 'FLEET-REBOOT-REQUIRED')
        $said | Should -Be $expected
        (Read-CallRecord $world.Dism).Count | Should -Be (2 * $script:features.Count)
        @(Read-CallRecord $world.Msiexec)[0] | Should -BeLike "/i `"*\scratch\$($script:msiFile)`" /qn /norestart"
        Test-Path -LiteralPath (Join-Path $world.Parameters.Scratch $script:msiFile) | Should -BeFalse
        Read-CallRecord $world.Git | Should -Be @('config --system --get core.longpaths', 'config --system core.longpaths true')
        (Get-Item -LiteralPath "$($world.Registry)\Policy").GetValue('ExecutionPolicy') | Should -BeExactly $script:policy
        (Get-Item -LiteralPath "$($world.Registry)\FileSystem").GetValue('LongPathsEnabled') | Should -Be 1
        (Get-Item -LiteralPath "$($world.Registry)\Environment").GetValue($script:variable[0]) | Should -BeExactly $script:variable[1]
        [Environment]::GetEnvironmentVariable($world.PathVariable, 'User') | Should -BeExactly ($script:entries -join ';')
        [System.IO.File]::ReadAllText($world.Parameters.WslConfigPath) | Should -BeLike '`[wsl2`]*memory=*GB*swap=8GB*'
    }
    It 'changes nothing on a laid host and asks for nothing' {
        $world = Initialize-World $script:name -FeatureState 'Enabled' -WslAnswer "WSL version: $($script:version).0" -WslExit 0 -Policy $script:policy `
            -LongPaths 1 -GitHeld 'true' -GitGetExit 0 -PathValue ('C:\already;' + ($script:entries -join ';')) -EnvValue $script:variable[1]
        Invoke-Base $script:name $world | Should -Be @('wrote .wslconfig; the ceiling applies when the VM next starts')
        Read-CallRecord $world.Msiexec | Should -Be @()
        Read-CallRecord $world.Git | Should -Be @('config --system --get core.longpaths')
    }
    It 'asks for no restart when the features and the MSI need none, and appends to a PATH that holds others' {
        $world = Initialize-World $script:name -EnableExit 0 -MsiExit 0 -Policy $script:policy -LongPaths 1 -GitHeld 'true' -GitGetExit 0 `
            -PathValue 'C:\already;' -EnvValue $script:variable[1]
        $said = Invoke-Base $script:name $world
        $said | Should -Not -Contain 'FLEET-REBOOT-REQUIRED'
        $said | Should -Contain "installed WSL $($script:version)"
        [Environment]::GetEnvironmentVariable($world.PathVariable, 'User') | Should -BeExactly ((@('C:\already') + $script:entries) -join ';')
    }
    It 'asks for the restart services need when only the machine variable changed' {
        $world = Initialize-World $script:name -FeatureState 'Enabled' -WslAnswer "WSL version: $($script:version).0" -WslExit 0 -Policy $script:policy `
            -LongPaths 1 -GitHeld 'true' -GitGetExit 0 -PathValue ($script:entries -join ';') -EnvValue 'C:\Users\x\AppData\Local\pypoetry'
        Invoke-Base $script:name $world | Should -Be @(
            "set the machine variable $($script:variable[0]) to $($script:variable[1])",
            'wrote .wslconfig; the ceiling applies when the VM next starts', 'FLEET-REBOOT-REQUIRED')
    }
    It 'resumes from an MSI an interrupted run already downloaded, verifying it rather than fetching it again' {
        $world = Initialize-World $script:name -FeatureState 'Enabled' -MsiExit 0
        [void][System.IO.Directory]::CreateDirectory($world.Parameters.Scratch)
        $left = Join-Path $world.Parameters.Scratch $script:msiFile
        [System.IO.File]::WriteAllText($left, 'fleet rebuild test payload')
        $world.Parameters.MsiUrl = ([Uri](Join-Path $world.Root 'never-served.msi')).AbsoluteUri
        Invoke-Base $script:name $world | Should -Contain "installed WSL $($script:version)"
        Test-Path -LiteralPath $left | Should -BeFalse
    }
    It 'refuses and removes a download that fails its digest' {
        $world = Initialize-World $script:name -FeatureState 'Enabled' -Pin ('0' * 64)
        { Invoke-Base $script:name $world } | Should -Throw "WSL MSI sha256 $($world.Digest) does not match the pin $('0' * 64)"
        Test-Path -LiteralPath (Join-Path $world.Parameters.Scratch $script:msiFile) | Should -BeFalse
    }
    It 'refuses by name <Case>' -ForEach @(
        @{ Case = 'a failed msiexec'; World = @{ FeatureState = 'Enabled'; MsiExit = 1603 }; Message = 'msiexec for WSL * exited 1603' }
        @{ Case = 'a feature dism cannot read'; World = @{ QueryExit = 5 }; Message = 'dism could not read the Windows feature *, exit 5' }
        @{ Case = 'a feature dism cannot enable'; World = @{ EnableExit = 87 }; Message = 'dism could not enable the Windows feature *, exit 87' }
        @{ Case = 'a git that cannot read its system config'; World = @{ FeatureState = 'Enabled'; MsiExit = 0; GitGetExit = 128 }
            Message = 'git config --system --get core.longpaths exited 128' }
        @{ Case = 'a git that cannot write its system config'; World = @{ FeatureState = 'Enabled'; MsiExit = 0; GitSetExit = 5 }
            Message = 'git config --system core.longpaths exited 5' }
    ) {
        $world = Initialize-World $script:name @World
        { Invoke-Base $script:name $world } | Should -Throw $Message
    }
}
