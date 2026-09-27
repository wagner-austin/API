Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# The driver that runs a sent bash payload inside a host's distro as root,
# from each roster host's committed render under rendered/
# (fleet.core.runner_distro, MCPs board task d69786fa, A2). wsl.exe is a
# stand-in batch file that records its arguments and the WSL_UTF8 it was
# given, prints what a payload would, and exits with the case's code, so
# nothing here touches a distro.

BeforeAll {
    . (Join-Path $PSScriptRoot 'rendered-fixtures.ps1')
}

Describe 'The distro driver for <_>' -ForEach @(Get-ChildItem -LiteralPath (Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'rendered') -Filter 'distro-driver-*.ps1' | ForEach-Object { $_.BaseName }) {
    BeforeAll {
        $script:name = $_
        $script:distro = Get-RenderedDefault $script:name 'Distro'
        $script:payload = Get-RenderedDefault $script:name 'Payload'
    }

    It 'runs the payload by path in the roster''s distro as root, with UTF-8 output, and passes its output through' {
        $wsl = Initialize-Batch 'wsl' @('echo WSL_UTF8=%WSL_UTF8%', 'echo ran', 'exit /b 0')
        $said = [string[]]@(Invoke-Rendered $script:name @{ Wsl = $wsl.Path })
        $said | Should -Be @('WSL_UTF8=1', 'ran')
        Read-CallRecord $wsl | Should -Be @("-d $script:distro -u root -- bash $script:payload")
    }
    It 'refuses by name a payload that exits non-zero, naming the payload, the distro and the code' {
        $wsl = Initialize-Batch 'wsl' @('exit /b 3')
        { Invoke-Rendered $script:name @{ Wsl = $wsl.Path } } |
            Should -Throw "FLEET_DISTRO_PAYLOAD_FAILED: bash $script:payload in $script:distro exited 3"
    }
    It 'runs another payload in another distro when the caller names them' {
        $wsl = Initialize-Batch 'wsl' @('exit /b 0')
        [void](Invoke-Rendered $script:name @{ Wsl = $wsl.Path; Distro = 'Other'; Payload = '/mnt/c/x/y.sh' })
        Read-CallRecord $wsl | Should -Be @('-d Other -u root -- bash /mnt/c/x/y.sh')
    }
}
