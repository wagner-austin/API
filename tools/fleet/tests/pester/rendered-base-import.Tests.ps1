Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# The rebuild's distro import stage, from each roster host's committed render
# under rendered/ (fleet.core.runner_base_render, MCPs board task d69786fa,
# A2). Each case passes a scratch HKCU key in place of WSL's Lxss key, laid
# out with the registrations the case needs, a stand-in wsl.exe that records
# its import, and the image served from a file:// URL whose digest the case
# computed. The distro's name and version are the render's own defaults.

BeforeAll {
    . (Join-Path $PSScriptRoot 'rendered-fixtures.ps1')

    # A scratch Lxss key holding the named registrations, or no key at all.
    function Initialize-Import {
        param([string[]]$Registered, [switch]$NoKey, [int]$ImportExit = 0, [string]$Pin = '')
        $root = Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        [void][System.IO.Directory]::CreateDirectory($root)
        $payload = Join-Path $root 'served.tar.gz'
        [System.IO.File]::WriteAllText($payload, 'fleet rebuild test payload')
        $digest = (Get-FileHash -Algorithm SHA256 -LiteralPath $payload).Hash.ToLower()
        $key = 'HKCU:\Software\fleet-test-' + [guid]::NewGuid().ToString('N')
        $script:keys.Add($key)
        if (-not $NoKey) {
            [void](New-Item -Path $key -Force)
            foreach ($distro in $Registered) {
                $entry = New-Item -Path (Join-Path $key ('{' + [guid]::NewGuid().ToString() + '}')) -Force
                Set-ItemProperty -LiteralPath $entry.PSPath -Name DistributionName -Value $distro
            }
        }
        $wsl = Initialize-Batch 'wsl' @("exit /b $ImportExit")
        $pin = @{ $true = $digest; $false = $Pin }[$Pin -eq '']
        return [pscustomobject]@{
            Wsl = $wsl; Root = $root; Digest = $digest
            Parameters = @{
                Scratch = (Join-Path $root 'scratch'); DistroDir = (Join-Path $root 'distro')
                RootfsUrl = ([Uri]$payload).AbsoluteUri; RootfsSha256 = $pin; LxssKey = $key; Wsl = $wsl.Path
            }
        }
    }

    $script:keys = [System.Collections.Generic.List[string]]::new()
}

AfterAll {
    foreach ($key in $script:keys) {
        if (Test-Path -LiteralPath $key) {
            Remove-Item -LiteralPath $key -Recurse -Force
        }
    }
}

Describe 'The distro import for <_>' -ForEach @(Get-ChildItem -LiteralPath (Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'rendered') -Filter 'base-import-*.ps1' | ForEach-Object { $_.BaseName }) {
    BeforeAll {
        $script:name = $_
        $script:distro = [string](Get-RenderedDefault $script:name 'Distro')
        $script:version = [string](Get-RenderedDefault $script:name 'RootfsVersion')
        $script:file = [string](Get-RenderedDefault $script:name 'RootfsFile')
    }

    It 'leaves a registered distro alone and never runs wsl' {
        $import = Initialize-Import -Registered @('docker-desktop', $script:distro)
        @(Invoke-Rendered $script:name $import.Parameters) | Should -Be @()
        Read-CallRecord $import.Wsl | Should -Be @()
    }
    It 'imports the verified image as the distro <Case>, and removes the image' -ForEach @(
        @{ Case = 'on a host that has registered none'; NoKey = $true; Registered = @() }
        @{ Case = 'beside a registration of another name'; NoKey = $false; Registered = @('docker-desktop') }
    ) {
        $import = Initialize-Import -Registered $Registered -NoKey:$NoKey
        $image = Join-Path $import.Parameters.Scratch $script:file
        @(Invoke-Rendered $script:name $import.Parameters) | Should -Be @("imported $($script:distro) $($script:version)")
        Read-CallRecord $import.Wsl | Should -Be @("--import $($script:distro) $($import.Parameters.DistroDir) $image --version 2")
        Test-Path -LiteralPath $import.Parameters.DistroDir -PathType Container | Should -BeTrue
        Test-Path -LiteralPath $image | Should -BeFalse
    }
    It 'resumes from an image an interrupted run already downloaded, verifying it rather than fetching it again' {
        $import = Initialize-Import -NoKey
        [void][System.IO.Directory]::CreateDirectory($import.Parameters.Scratch)
        [System.IO.File]::WriteAllText((Join-Path $import.Parameters.Scratch $script:file), 'fleet rebuild test payload')
        $import.Parameters.RootfsUrl = ([Uri](Join-Path $import.Root 'never-served.tar.gz')).AbsoluteUri
        @(Invoke-Rendered $script:name $import.Parameters) | Should -Be @("imported $($script:distro) $($script:version)")
    }
    It 'refuses a failed import by its exit code and keeps the verified image for the re-run' {
        $import = Initialize-Import -NoKey -ImportExit 1
        { Invoke-Rendered $script:name $import.Parameters } | Should -Throw "wsl --import $($script:distro) exited 1"
        [System.IO.File]::ReadAllText((Join-Path $import.Parameters.Scratch $script:file)) | Should -BeExactly 'fleet rebuild test payload'
    }
    It 'refuses and removes an image that fails its digest, importing nothing' {
        $import = Initialize-Import -NoKey -Pin ('0' * 64)
        { Invoke-Rendered $script:name $import.Parameters } | Should -Throw "rootfs sha256 $($import.Digest) does not match the pin $('0' * 64)"
        Test-Path -LiteralPath (Join-Path $import.Parameters.Scratch $script:file) | Should -BeFalse
        Read-CallRecord $import.Wsl | Should -Be @()
    }
}
