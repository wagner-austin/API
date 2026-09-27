Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# The runner audit, from each roster host's committed render under rendered/
# (fleet.core.runner_audit, MCPs board task d69786fa, A2). Each case runs the
# render with a stand-in wsl.exe that answers each probe by its command line,
# stand-in schtasks and git, scratch HKCU keys, and the Windows-side reads
# as script blocks, then holds the transcript to the render's own
# Write-Check lines: one CHECK line per row, in order, OK on a healthy host
# and DRIFT with the probe's words on a sick one, which is the contract
# parse_audit_transcript scores.

BeforeAll {
    . (Join-Path $PSScriptRoot 'rendered-fixtures.ps1')

    # The check ids a render writes, in order, read from its own text.
    function Get-CheckId {
        param([string]$Name)
        $text = [System.IO.File]::ReadAllText((Join-Path $script:rendered "$Name.ps1"))
        return [string[]]@([regex]::Matches($text, "(?m)^\s*Write-Check '([^']+)'") | ForEach-Object { $_.Groups[1].Value })
    }

    # A pinned asset's digest, as the render compares it.
    function Get-Pin {
        param([string]$Name)
        $text = [System.IO.File]::ReadAllText((Join-Path $script:rendered "$Name.ps1"))
        return [string[]]@([regex]::Matches($text, "StartsWith\('([0-9a-f]{64})'\)") | ForEach-Object { $_.Groups[1].Value })
    }

    # A healthy host or a sick one: every probe answers, or every probe fails.
    # A healthy host can differ in one answer: the disk df reports, the
    # account a service runs as, the machine variable's value, git's
    # core.longpaths.
    function Initialize-Host {
        param(
            [string]$Name, [switch]$Sick,
            [string]$Used = '40G', [string]$StartName = 'LocalSystem', [string]$Held = '', [string]$GitAnswer = 'true'
        )
        $registry = 'HKCU:\Software\fleet-test-' + [guid]::NewGuid().ToString('N')
        foreach ($key in 'Policy', 'FileSystem', 'Environment') {
            [void](New-Item -Path "$registry\$key" -Force)
        }
        $script:keys.Add($registry)
        $asked = [System.Collections.Generic.List[string]]::new()
        if ($Sick) {
            $wsl = Initialize-Batch 'wsl' @('echo There is no distribution with the supplied name.', 'exit /b 1')
            $schtasks = Initialize-Batch 'schtasks' @('echo ERROR: The system cannot find the file specified.', 'exit /b 1')
            $git = Initialize-Batch 'git' @('exit /b 1')
            Set-ItemProperty -LiteralPath "$registry\Policy" -Name ExecutionPolicy -Value 'Restricted'
            $service = { param([string]$Name) $asked.Add("service $Name"); @() }.GetNewClosure()
            $workdir = { param([string]$Path) $asked.Add("workdir $Path"); $false }.GetNewClosure()
        } else {
            $pins = @(Get-Pin $Name)
            $sum = @($pins + ('0' * 64))[0]
            $wsl = Initialize-Batch 'wsl' @(
                'echo %* | findstr /c:"free -m" >nul', 'if not errorlevel 1 goto free',
                'echo %* | findstr /c:"df -BG" >nul', 'if not errorlevel 1 goto df',
                'echo %* | findstr /c:"nvidia-smi" >nul', 'if not errorlevel 1 goto gpu',
                'echo %* | findstr /c:"is-enabled" >nul', 'if not errorlevel 1 goto enabled',
                'echo %* | findstr /c:"is-active" >nul', 'if not errorlevel 1 goto active',
                'echo %* | findstr /c:"sha256sum" >nul', 'if not errorlevel 1 goto sum',
                'exit /b 0',
                ':free', 'echo               total        used', 'echo Mem:          64000        2000', 'exit /b 0',
                ':df', 'echo  Used', "echo  $Used", 'exit /b 0',
                ':gpu', 'echo NVIDIA GeForce RTX 4090 Laptop GPU', 'exit /b 0',
                ':enabled', 'echo enabled', 'exit /b 0',
                ':active', 'echo active', 'exit /b 0',
                ':sum', "echo $sum  /asset", 'exit /b 0')
            $schtasks = Initialize-Batch 'schtasks' @('echo "TaskName","Next Run Time","Status"', 'echo "\wsl-keepalive","N/A","Running"', 'exit /b 0')
            $git = Initialize-Batch 'git' @("echo $GitAnswer", 'exit /b 0')
            # The policy the roster wants is the last part of its row's id.
            $policy = @(Get-CheckId $Name | Where-Object { $_ -like 'execution-policy:LocalMachine:*' })[0].Split(':')[-1]
            Set-ItemProperty -LiteralPath "$registry\Policy" -Name ExecutionPolicy -Value $policy
            Set-ItemProperty -LiteralPath "$registry\FileSystem" -Name LongPathsEnabled -Value 1 -Type DWord
            $text = [System.IO.File]::ReadAllText((Join-Path $script:rendered "$Name.ps1"))
            foreach ($variable in [regex]::Matches($text, 'GetValue\(''([A-Z_]+)''\)\s*Write-Check ''[^'']+'' \(\$Held -ceq ''([^'']*)''\)')) {
                $value = @{ $true = $variable.Groups[2].Value; $false = $Held }[$Held -eq '']
                Set-ItemProperty -LiteralPath "$registry\Environment" -Name $variable.Groups[1].Value -Value $value
            }
            $account = $StartName
            $service = { param([string]$Name) $asked.Add("service $Name"); [pscustomobject]@{ State = 'Running'; StartName = $account } }.GetNewClosure()
            $workdir = { param([string]$Path) $asked.Add("workdir $Path"); $true }.GetNewClosure()
        }
        return [pscustomobject]@{
            Wsl = $wsl; Asked = $asked
            Parameters = @{
                PolicyKey = "$registry\Policy"; FileSystemKey = "$registry\FileSystem"; EnvironmentKey = "$registry\Environment"
                GetService = $service; TestWorkdir = $workdir; Git = $git.Path; Schtasks = $schtasks.Path; Wsl = $wsl.Path
            }
        }
    }

    $script:keys = [System.Collections.Generic.List[string]]::new()
}

AfterAll {
    foreach ($key in $script:keys) {
        Remove-Item -LiteralPath $key -Recurse -Force
    }
}

Describe 'The runner audit for <_>' -ForEach @(Get-ChildItem -LiteralPath (Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'rendered') -Filter 'audit-*.ps1' | ForEach-Object { $_.BaseName }) {
    BeforeAll {
        $script:name = $_
        $script:ids = Get-CheckId $script:name
    }

    It 'reports every row OK, in the order the roster declares them, on a healthy host' {
        $healthy = Initialize-Host $script:name
        $said = [string[]]@(Invoke-Rendered $script:name $healthy.Parameters)
        $said | Should -Be @($script:ids | ForEach-Object { "CHECK $_ OK" })
        @(Read-CallRecord $healthy.Wsl)[0] | Should -BeLike '-d * -- free -m'
    }
    It 'reports every row DRIFT with the probe''s own words on a sick host, and never comes up short' {
        $sick = Initialize-Host $script:name -Sick
        $said = [string[]]@(Invoke-Rendered $script:name $sick.Parameters)
        $said.Count | Should -Be $script:ids.Count
        foreach ($index in 0..($script:ids.Count - 1)) {
            $said[$index] | Should -BeLike "CHECK $($script:ids[$index]) DRIFT *"
        }
        @($said | Where-Object { $_ -like 'CHECK asset:* DRIFT test -e exited 1' }).Count | Should -BeGreaterThan 0
        @($said | Where-Object { $_ -like 'CHECK service:windows:* DRIFT Win32_Service State: ' }).Count | Should -BeGreaterThan 0
        @($said | Where-Object { $_ -like 'CHECK execution-policy:* DRIFT the LocalMachine ExecutionPolicy value is: Restricted' }).Count | Should -Be 1
        @($said | Where-Object { $_ -like 'CHECK gpu:* DRIFT *(exit 1)' }).Count | Should -Be @($script:ids | Where-Object { $_ -like 'gpu:*' }).Count
        # Each Windows service and workdir is asked once, in roster order.
        $services = @($script:ids | Where-Object { $_ -like 'service:windows:*' } | ForEach-Object { 'service ' + $_.Substring(16) })
        @($sick.Asked | Where-Object { $_ -like 'service *' }) | Should -Be $services
        @($sick.Asked | Where-Object { $_ -like 'workdir *' }).Count | Should -Be $services.Count
        @($said | Where-Object { $_ -like 'CHECK disk:* DRIFT the distro root uses -1 GB*There is no distribution with the supplied name. (exit 1)' }).Count | Should -Be 1
    }
    It 'asks each WSL runner''s own PATH for the GPU' {
        $healthy = Initialize-Host $script:name
        [void](Invoke-Rendered $script:name $healthy.Parameters)
        $gpu = @(Read-CallRecord $healthy.Wsl | Where-Object { $_ -like '*nvidia-smi*' })
        $gpu.Count | Should -Be @($script:ids | Where-Object { $_ -like 'gpu:*' }).Count
        foreach ($call in $gpu) {
            $call | Should -Match "PATH=\$\(cat /home/gharunner/[^/]+/\.path\) nvidia-smi --query-gpu=name --format=csv,noheader"
        }
    }
    It 'drifts exactly the one row whose answer is <Case>' -ForEach @(
        @{ Case = 'a disk one GB past its ceiling'; Variant = @{ Used = '151G' }; Row = 'disk:*'; Detail = 'the distro root uses 151 GB against a ceiling of 150 GB*' }
        @{ Case = 'a service running as another account'; Variant = @{ StartName = 'NT AUTHORITY\NetworkService' }; Row = 'account:windows:*'
            Detail = 'Win32_Service StartName: NT AUTHORITY\NetworkService' }
        @{ Case = 'a machine variable differing only in case'; Variant = @{ Held = 'C:\FLEET\POETRY' }; Row = 'machine-env:*'
            Detail = 'the machine environment holds POETRY_CACHE_DIR=C:\FLEET\POETRY; the roster says C:\fleet\poetry' }
        @{ Case = 'git without core.longpaths'; Variant = @{ GitAnswer = 'false' }; Row = 'long-paths:*'; Detail = 'LongPathsEnabled=1 git core.longpaths=false (exit 0)' }
    ) {
        $variant = Initialize-Host $script:name @Variant
        $said = [string[]]@(Invoke-Rendered $script:name $variant.Parameters)
        $drifted = @($said | Where-Object { $_ -like 'CHECK * DRIFT *' })
        $drifted.Count | Should -Be @($script:ids | Where-Object { $_ -like $Row }).Count
        foreach ($line in $drifted) {
            $line | Should -BeLike "CHECK $Row DRIFT $Detail"
        }
    }
    It 'holds a disk at its ceiling to it' {
        $full = Initialize-Host $script:name -Used '150G'
        @(Invoke-Rendered $script:name $full.Parameters | Where-Object { $_ -like 'CHECK disk:* OK' }).Count | Should -Be 1
    }
    It 'reads Windows services and workdirs through its own defaults, still one row each' {
        $sick = Initialize-Host $script:name -Sick
        $sick.Parameters.Remove('GetService')
        $sick.Parameters.Remove('TestWorkdir')
        $said = [string[]]@(Invoke-Rendered $script:name $sick.Parameters)
        @($said | ForEach-Object { ($_ -split ' ')[1] }) | Should -Be $script:ids
    }
}
