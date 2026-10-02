Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# The toolchain probe a Windows node answers, from its committed render under
# rendered/ (fleet.core.windows_toolchain_probe, MCPs board task d69786fa,
# A2). Each case lays out a PATH of stand-in tools under $TestDrive, one
# entry of it empty and one quoted as a node's PATH can carry them, runs the
# render with a stand-in vswhere, and reads every line it prints. PATH and
# PATHEXT are the case's own and restored after it.

BeforeAll {
    . (Join-Path $PSScriptRoot 'rendered-fixtures.ps1')

    # A .cmd that records each call's arguments, prints the given lines on
    # the given stream, and exits with the given code.
    function Initialize-Answer {
        param([string]$Directory, [string]$Name, [string[]]$Lines, [int]$ExitCode, [switch]$Stderr)
        [void][System.IO.Directory]::CreateDirectory($Directory)
        $record = Join-Path $Directory "$Name.calls.txt"
        $stream = @{ $true = '1>&2 '; $false = '' }[$Stderr.IsPresent]
        $echoes = @($Lines | ForEach-Object { "${stream}echo $_" }) -join "`r`n"
        $body = "@echo off`r`necho %*>>`"$record`"`r`n$echoes`r`nexit /b $ExitCode`r`n"
        [System.IO.File]::WriteAllText((Join-Path $Directory "$Name.cmd"), $body, [System.Text.Encoding]::ASCII)
        return $record
    }

    # A python.cmd that answers --version, -m pip --version with the given
    # exit code, and -c (the hooks line's import) with ImportExit.
    function Initialize-Python {
        param([string]$Directory, [int]$PipExit, [int]$ImportExit = 0)
        [void][System.IO.Directory]::CreateDirectory($Directory)
        $record = Join-Path $Directory 'python.calls.txt'
        $body = "@echo off`r`necho %*>>`"$record`"`r`nif `"%1`"==`"-m`" goto pip`r`n" +
            "if `"%1`"==`"-c`" exit /b $ImportExit`r`necho Python 3.11.9`r`nexit /b 0`r`n" +
            ":pip`r`necho pip 24.2 from C:\py\Lib\site-packages\pip (python 3.11)`r`nexit /b $PipExit`r`n"
        [System.IO.File]::WriteAllText((Join-Path $Directory 'python.cmd'), $body, [System.Text.Encoding]::ASCII)
        return $record
    }
}

Describe 'The toolchain probe' {
    BeforeEach {
        $script:savedPath = $env:PATH
        $script:savedExtensions = $env:PATHEXT
        $env:PATHEXT = '.COM;.EXE;.BAT;.CMD'
        $script:root = Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
    }
    AfterEach {
        $env:PATH = $script:savedPath
        $env:PATHEXT = $script:savedExtensions
    }

    It 'reports each tool PATH names with the first line it answers on either stream, pip by module, and the VC tools vswhere finds' {
        $first = Join-Path $script:root 'first'
        $quoted = Join-Path $script:root 'quoted dir'
        $pythonCalls = Initialize-Python $first 0
        [void](Initialize-Answer $first 'git' @('git version 2.46.0.windows.1', 'a second line') 0)
        [void](Initialize-Answer $first 'node' @('v20.17.0') 0)
        [void](Initialize-Answer $first 'poetry' @('Poetry (version 1.8.3)') 3 -Stderr)
        [void](Initialize-Answer $first 'tar' @() 0)
        $ffmpegCalls = Initialize-Answer $first 'ffmpeg' @('ffmpeg version 9.0.1-essentials_build-www.gyan.dev', 'built with gcc') 0
        [void](Initialize-Answer $quoted 'make' @('GNU Make 4.4.1', 'Built for Windows32') 0)
        $vswhereCalls = Initialize-Answer (Join-Path $script:root 'vs') 'vswhere' @('17.11.35312.102') 0
        $env:PATH = "$first;;`"$quoted`""
        $said = @(Invoke-Rendered 'dialect-toolchain-probe' @{
                VsWhere = (Join-Path $script:root 'vs\vswhere.cmd'); HooksRoute = (Join-Path $script:root 'no-route.json')
            })
        # The default seam asks this session's own token, so the expected
        # line is read here the same way, independently of the script.
        $principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
        $integrity = 'integrity=no=limited'
        if ($principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
            $integrity = 'integrity=yes=administrator'
        }
        $said | Should -Be @(
            'python=yes=Python 3.11.9', 'poetry=yes=Poetry (version 1.8.3)', 'git=yes=git version 2.46.0.windows.1',
            'make=yes=GNU Make 4.4.1', 'node=yes=v20.17.0', 'ffmpeg=yes=ffmpeg version 9.0.1-essentials_build-www.gyan.dev',
            'tar=yes=', 'cargo=no=', 'winget=no=', 'choco=no=',
            'pip=yes=pip 24.2 from C:\py\Lib\site-packages\pip (python 3.11)', 'cxx=yes=17.11.35312.102', 'gpu=no=',
            'testdb=no=', 'docker=no=', 'hooks=no=', $integrity)
        [System.IO.File]::ReadAllLines($pythonCalls) | Should -Be @('--version', '-m pip --version')
        [System.IO.File]::ReadAllLines($ffmpegCalls) | Should -Be @('--version')
        [System.IO.File]::ReadAllText($vswhereCalls).Trim() |
            Should -BeExactly '-products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationVersion'
    }
    It 'reports a python found only under WindowsApps absent, pip with it, and never runs it' {
        $alias = Join-Path $script:root 'Microsoft\WindowsApps'
        $calls = Initialize-Python $alias 0
        $env:PATH = $alias
        $said = @(Invoke-Rendered 'dialect-toolchain-probe' @{ VsWhere = (Join-Path $script:root 'absent\vswhere.exe') })
        $said[0] | Should -BeExactly 'python=no='
        $said[5] | Should -BeExactly 'ffmpeg=no='
        $said[10] | Should -BeExactly 'pip=no='
        $said[11] | Should -BeExactly 'cxx=no='
        Test-Path -LiteralPath $calls | Should -BeFalse
    }
    It 'reports pip absent when the interpreter has none, and no VC tools when vswhere finds none' {
        $tools = Join-Path $script:root 'tools'
        [void](Initialize-Python $tools 1)
        [void](Initialize-Answer (Join-Path $script:root 'vs') 'vswhere' @() 0)
        $env:PATH = $tools
        $said = @(Invoke-Rendered 'dialect-toolchain-probe' @{ VsWhere = (Join-Path $script:root 'vs\vswhere.cmd') })
        $said[0] | Should -BeExactly 'python=yes=Python 3.11.9'
        $said[10] | Should -BeExactly 'pip=no='
        $said[11] | Should -BeExactly 'cxx=no='
    }
    It 'reports the CUDA device nvidia-smi lists, and none when it fails or lists nothing (MCPs 939ec5c7)' {
        $absent = Join-Path $script:root 'absent\vswhere.exe'
        $card = Join-Path $script:root 'card'
        $calls = Initialize-Answer $card 'nvidia-smi' @('NVIDIA GeForce RTX 3070 Ti Laptop GPU, 8.6') 0
        $env:PATH = $card
        $found = @(Invoke-Rendered 'dialect-toolchain-probe' @{ VsWhere = $absent })
        $found[12] | Should -BeExactly 'gpu=yes=NVIDIA GeForce RTX 3070 Ti Laptop GPU, 8.6'
        $found[13] | Should -BeExactly 'testdb=no='
        [System.IO.File]::ReadAllText($calls).Trim() | Should -BeExactly '--query-gpu=name,compute_cap --format=csv,noheader'
        $failing = Join-Path $script:root 'failing'
        [void](Initialize-Answer $failing 'nvidia-smi' @('NVIDIA-SMI has failed') 9)
        $env:PATH = $failing
        @(Invoke-Rendered 'dialect-toolchain-probe' @{ VsWhere = $absent })[12] | Should -BeExactly 'gpu=no='
        $silent = Join-Path $script:root 'silent'
        [void](Initialize-Answer $silent 'nvidia-smi' @() 0)
        $env:PATH = $silent
        @(Invoke-Rendered 'dialect-toolchain-probe' @{ VsWhere = $absent })[12] | Should -BeExactly 'gpu=no='
    }
    It 'reports the hooks route only when the file exists and one import of the check tools succeeds (MCPs ec895824)' {
        $absent = Join-Path $script:root 'absent\vswhere.exe'
        $route = Join-Path $script:root 'home\.claude\corvis-hooks.json'
        [void][System.IO.Directory]::CreateDirectory((Split-Path $route))
        [System.IO.File]::WriteAllText($route, '{}')
        $ready = Join-Path $script:root 'ready'
        $readyCalls = Initialize-Python $ready 0 0
        $env:PATH = $ready
        @(Invoke-Rendered 'dialect-toolchain-probe' @{ VsWhere = $absent; HooksRoute = $route })[15] |
            Should -BeExactly "hooks=yes=$route"
        [System.IO.File]::ReadAllLines($readyCalls) |
            Should -Be @('--version', '-m pip --version', '-c "import ruff, mypy, pytest, xdist, pytest_cov"')
        $lacking = Join-Path $script:root 'lacking'
        [void](Initialize-Python $lacking 0 1)
        $env:PATH = $lacking
        @(Invoke-Rendered 'dialect-toolchain-probe' @{ VsWhere = $absent; HooksRoute = $route })[15] |
            Should -BeExactly 'hooks=no='
        $env:PATH = Join-Path $script:root 'empty'
        @(Invoke-Rendered 'dialect-toolchain-probe' @{ VsWhere = $absent; HooksRoute = $route })[15] |
            Should -BeExactly 'hooks=no='
    }
    It 'reports an administrator token yes and a filtered one no, as the elevated runner reads them' {
        $env:PATH = Join-Path $script:root 'empty'
        $absent = Join-Path $script:root 'absent\vswhere.exe'
        $admin = @(Invoke-Rendered 'dialect-toolchain-probe' @{ VsWhere = $absent; Administrator = { $true } })
        $admin[-1] | Should -BeExactly 'integrity=yes=administrator'
        $limited = @(Invoke-Rendered 'dialect-toolchain-probe' @{ VsWhere = $absent; Administrator = { $false } })
        $limited[-1] | Should -BeExactly 'integrity=no=limited'
    }
}
