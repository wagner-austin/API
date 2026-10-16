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
        # go has no --version flag; it is asked 'version' (MCPs 1da15750).
        $goCalls = Initialize-Answer $first 'go' @('go version go1.27.1 windows/amd64') 0
        [void](Initialize-Answer $quoted 'make' @('GNU Make 4.4.1', 'Built for Windows32') 0)
        $vswhereCalls = Initialize-Answer (Join-Path $script:root 'vs') 'vswhere' @('17.11.35312.102') 0
        $env:PATH = "$first;;`"$quoted`""
        # Chrome and pdftotext are named like the hooks route, at paths this
        # case lays nothing under, so a Chrome or a Git for Windows installed
        # on the machine running the suite does not read as a node's.
        $said = @(Invoke-Rendered 'dialect-toolchain-probe' @{
                VsWhere = (Join-Path $script:root 'vs\vswhere.cmd'); HooksRoute = (Join-Path $script:root 'no-route.json')
                Chrome = @((Join-Path $script:root 'no-chrome\chrome.exe'))
                Pdftotext = (Join-Path $script:root 'no-git\pdftotext.exe')
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
            'go=yes=go version go1.27.1 windows/amd64', 'tar=yes=', 'cargo=no=', 'winget=no=', 'choco=no=',
            'pip=yes=pip 24.2 from C:\py\Lib\site-packages\pip (python 3.11)', 'cxx=yes=17.11.35312.102', 'gpu=no=',
            'testdb=no=', 'docker=no=', 'hooks=no=', 'chrome=no=', 'pdftotext=no=', $integrity)
        [System.IO.File]::ReadAllLines($pythonCalls) | Should -Be @('--version', '-m pip --version')
        [System.IO.File]::ReadAllLines($ffmpegCalls) | Should -Be @('--version')
        [System.IO.File]::ReadAllLines($goCalls) | Should -Be @('version')
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
        $said[6] | Should -BeExactly 'go=no='
        $said[11] | Should -BeExactly 'pip=no='
        $said[12] | Should -BeExactly 'cxx=no='
        Test-Path -LiteralPath $calls | Should -BeFalse
    }
    It 'reports pip absent when the interpreter has none, and no VC tools when vswhere finds none' {
        $tools = Join-Path $script:root 'tools'
        [void](Initialize-Python $tools 1)
        [void](Initialize-Answer (Join-Path $script:root 'vs') 'vswhere' @() 0)
        $env:PATH = $tools
        $said = @(Invoke-Rendered 'dialect-toolchain-probe' @{ VsWhere = (Join-Path $script:root 'vs\vswhere.cmd') })
        $said[0] | Should -BeExactly 'python=yes=Python 3.11.9'
        $said[11] | Should -BeExactly 'pip=no='
        $said[12] | Should -BeExactly 'cxx=no='
    }
    It 'reports cargo only when fmt, clippy and llvm-cov answer and uv is on the PATH (MCPs b1089284)' {
        $absent = Join-Path $script:root 'absent\vswhere.exe'
        # A cargo.cmd that answers --version and each subcommand it is given,
        # and refuses any other as cargo does, recording every call.
        function Initialize-Cargo {
            param([string]$Directory, [string[]]$Subcommands)
            [void][System.IO.Directory]::CreateDirectory($Directory)
            $record = Join-Path $Directory 'cargo.calls.txt'
            $routes = @($Subcommands | ForEach-Object { "if `"%1`"==`"$_`" goto sub" }) -join "`r`n"
            $body = "@echo off`r`necho %*>>`"$record`"`r`nif `"%1`"==`"--version`" goto version`r`n" +
                "$routes`r`necho error: no such command 1>&2`r`nexit /b 101`r`n" +
                ":version`r`necho cargo 1.99.0 5f94df478 2026-08-27`r`nexit /b 0`r`n" +
                ":sub`r`necho %1 1.0.0`r`nexit /b 0`r`n"
            [System.IO.File]::WriteAllText((Join-Path $Directory 'cargo.cmd'), $body, [System.Text.Encoding]::ASCII)
            return $record
        }
        $whole = Join-Path $script:root 'whole'
        $calls = Initialize-Cargo $whole @('fmt', 'clippy', 'llvm-cov')
        [void](Initialize-Answer $whole 'uv' @('uv 0.13.0') 0)
        $env:PATH = $whole
        @(Invoke-Rendered 'dialect-toolchain-probe' @{ VsWhere = $absent })[8] |
            Should -BeExactly 'cargo=yes=cargo 1.99.0 5f94df478 2026-08-27'
        [System.IO.File]::ReadAllLines($calls) |
            Should -Be @('fmt --version', 'clippy --version', 'llvm-cov --version', '--version')
        $minimal = Join-Path $script:root 'minimal'
        [void](Initialize-Cargo $minimal @('clippy', 'llvm-cov'))
        [void](Initialize-Answer $minimal 'uv' @('uv 0.13.0') 0)
        $env:PATH = $minimal
        @(Invoke-Rendered 'dialect-toolchain-probe' @{ VsWhere = $absent })[8] | Should -BeExactly 'cargo=no='
        $noUv = Join-Path $script:root 'no-uv'
        [void](Initialize-Cargo $noUv @('fmt', 'clippy', 'llvm-cov'))
        $env:PATH = $noUv
        @(Invoke-Rendered 'dialect-toolchain-probe' @{ VsWhere = $absent })[8] | Should -BeExactly 'cargo=no='
    }
    It 'reports the CUDA device nvidia-smi lists, and none when it fails or lists nothing (MCPs 939ec5c7)' {
        $absent = Join-Path $script:root 'absent\vswhere.exe'
        $card = Join-Path $script:root 'card'
        $calls = Initialize-Answer $card 'nvidia-smi' @('NVIDIA GeForce RTX 3070 Ti Laptop GPU, 8.6') 0
        $env:PATH = $card
        $found = @(Invoke-Rendered 'dialect-toolchain-probe' @{ VsWhere = $absent })
        $found[13] | Should -BeExactly 'gpu=yes=NVIDIA GeForce RTX 3070 Ti Laptop GPU, 8.6'
        $found[14] | Should -BeExactly 'testdb=no='
        [System.IO.File]::ReadAllText($calls).Trim() | Should -BeExactly '--query-gpu=name,compute_cap --format=csv,noheader'
        $failing = Join-Path $script:root 'failing'
        [void](Initialize-Answer $failing 'nvidia-smi' @('NVIDIA-SMI has failed') 9)
        $env:PATH = $failing
        @(Invoke-Rendered 'dialect-toolchain-probe' @{ VsWhere = $absent })[13] | Should -BeExactly 'gpu=no='
        $silent = Join-Path $script:root 'silent'
        [void](Initialize-Answer $silent 'nvidia-smi' @() 0)
        $env:PATH = $silent
        @(Invoke-Rendered 'dialect-toolchain-probe' @{ VsWhere = $absent })[13] | Should -BeExactly 'gpu=no='
    }
    It 'reports the image of the corvis-fleet-testdb container docker answers, and none when it fails or answers nothing (MCPs daae17f2)' {
        $absent = Join-Path $script:root 'absent\vswhere.exe'
        $running = Join-Path $script:root 'running'
        $calls = Initialize-Answer $running 'docker' @('pgvector/pgvector:pg16-bookworm') 0
        $env:PATH = $running
        @(Invoke-Rendered 'dialect-toolchain-probe' @{ VsWhere = $absent })[14] |
            Should -BeExactly 'testdb=yes=pgvector/pgvector:pg16-bookworm'
        [System.IO.File]::ReadAllText($calls).Trim() |
            Should -BeExactly 'container inspect --format "{{.Config.Image}}" corvis-fleet-testdb'
        $missing = Join-Path $script:root 'missing'
        [void](Initialize-Answer $missing 'docker' @('Error: No such container: corvis-fleet-testdb') 1 -Stderr)
        $env:PATH = $missing
        @(Invoke-Rendered 'dialect-toolchain-probe' @{ VsWhere = $absent })[14] | Should -BeExactly 'testdb=no='
        $silent = Join-Path $script:root 'silent'
        [void](Initialize-Answer $silent 'docker' @() 0)
        $env:PATH = $silent
        @(Invoke-Rendered 'dialect-toolchain-probe' @{ VsWhere = $absent })[14] | Should -BeExactly 'testdb=no='
    }
    It 'reports the hooks route only when the file exists and the check tools are at their pins (MCPs ec895824, 09a34e03)' {
        $absent = Join-Path $script:root 'absent\vswhere.exe'
        $route = Join-Path $script:root 'home\.claude\corvis-hooks.json'
        [void][System.IO.Directory]::CreateDirectory((Split-Path $route))
        [System.IO.File]::WriteAllText($route, '{}')
        $ready = Join-Path $script:root 'ready'
        $readyCalls = Initialize-Python $ready 0 0
        $env:PATH = $ready
        @(Invoke-Rendered 'dialect-toolchain-probe' @{ VsWhere = $absent; HooksRoute = $route })[16] |
            Should -BeExactly "hooks=yes=$route"
        [System.IO.File]::ReadAllLines($readyCalls) |
            Should -Be @('--version', '-m pip --version', '-c "import sys, importlib.metadata as m; a = sys.argv[1:]; sys.exit(any(m.version(a[i]) != a[i + 1] for i in range(0, len(a), 2)))" ruff 0.17.1 mypy 2.4.0 pytest 9.1.1 pytest-xdist 3.8.0 pytest-cov 7.1.0 coverage 7.16.2 pytest-timeout 2.4.0')
        $lacking = Join-Path $script:root 'lacking'
        [void](Initialize-Python $lacking 0 1)
        $env:PATH = $lacking
        @(Invoke-Rendered 'dialect-toolchain-probe' @{ VsWhere = $absent; HooksRoute = $route })[16] |
            Should -BeExactly 'hooks=no='
        $env:PATH = Join-Path $script:root 'empty'
        @(Invoke-Rendered 'dialect-toolchain-probe' @{ VsWhere = $absent; HooksRoute = $route })[16] |
            Should -BeExactly 'hooks=no='
    }
    It 'reports Chrome from the first candidate path that exists with its file version, never from PATH, and none when no candidate exists (MCPs 2f596185)' {
        $env:PATH = Join-Path $script:root 'empty'
        $absent = Join-Path $script:root 'absent\vswhere.exe'
        # A real version resource: the probe reads the file, it does not launch it.
        $installed = Join-Path $script:root 'Program Files\Google\Chrome\Application\chrome.exe'
        [void][System.IO.Directory]::CreateDirectory((Split-Path $installed))
        [System.IO.File]::Copy("$env:SystemRoot\System32\cmd.exe", $installed)
        $expected = [System.Diagnostics.FileVersionInfo]::GetVersionInfo($installed).ProductVersion
        $perUser = Join-Path $script:root 'Local\Google\Chrome\Application\chrome.exe'
        $candidates = @((Join-Path $script:root 'Program Files (x86)\Google\Chrome\Application\chrome.exe'), $installed, $perUser)
        @(Invoke-Rendered 'dialect-toolchain-probe' @{ VsWhere = $absent; Chrome = $candidates })[17] |
            Should -BeExactly "chrome=yes=$expected"
        # A chrome on the PATH counts for nothing: Playwright opens the path, not the PATH.
        $onPath = Join-Path $script:root 'onpath'
        [void](Initialize-Answer $onPath 'chrome' @('Google Chrome 155.0.8059.39') 0)
        $env:PATH = $onPath
        @(Invoke-Rendered 'dialect-toolchain-probe' @{ VsWhere = $absent; Chrome = @($perUser) })[17] |
            Should -BeExactly 'chrome=no='
    }
    It 'reports pdftotext from the Git for Windows path first, then from PATH, and none when neither has it (MCPs 6dd73ada)' {
        $env:PATH = Join-Path $script:root 'empty'
        $absent = Join-Path $script:root 'absent\vswhere.exe'
        # poppler answers -v on stderr; cmd's own 2>&1 carries it into the line.
        $git = Join-Path $script:root 'Git\mingw64\bin'
        $gitCalls = Initialize-Answer $git 'pdftotext' @('pdftotext version 4.00') 0 -Stderr
        $onPath = Join-Path $script:root 'onpath'
        [void](Initialize-Answer $onPath 'pdftotext' @('pdftotext version 24.02.0') 0)
        $env:PATH = $onPath
        @(Invoke-Rendered 'dialect-toolchain-probe' @{ VsWhere = $absent; Pdftotext = (Join-Path $git 'pdftotext.cmd') })[18] |
            Should -BeExactly 'pdftotext=yes=pdftotext version 4.00'
        [System.IO.File]::ReadAllLines($gitCalls) | Should -Be @('-v')
        $missing = Join-Path $script:root 'no-git\pdftotext.exe'
        @(Invoke-Rendered 'dialect-toolchain-probe' @{ VsWhere = $absent; Pdftotext = $missing })[18] |
            Should -BeExactly 'pdftotext=yes=pdftotext version 24.02.0'
        $env:PATH = Join-Path $script:root 'empty'
        @(Invoke-Rendered 'dialect-toolchain-probe' @{ VsWhere = $absent; Pdftotext = $missing })[18] |
            Should -BeExactly 'pdftotext=no='
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
