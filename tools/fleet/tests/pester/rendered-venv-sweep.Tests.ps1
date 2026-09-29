Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# The fleet cache's virtualenv sweep, executed from its committed copy under
# rendered/ (fleet.core.venv_sweep, MCPs board task 7b07c5d2). Its directory
# is a parameter defaulting to the example stage root's, so each case lays out
# poetry virtualenvs under $TestDrive the way a Windows node holds them: a
# folder per virtualenv, its packages under Lib\site-packages, and a .pth line
# per source path the editable install recorded.

BeforeAll {
    . (Join-Path $PSScriptRoot 'rendered-fixtures.ps1')

    function New-Virtualenv {
        param([string]$Venvs, [string]$Name, [string[]]$Lines, [switch]$NoSitePackages)
        $venv = Join-Path $Venvs $Name
        [void][System.IO.Directory]::CreateDirectory($venv)
        [System.IO.File]::WriteAllText((Join-Path $venv 'pyvenv.cfg'), 'home = C:\Python311')
        if (-not $NoSitePackages) {
            $site = Join-Path $venv 'Lib\site-packages'
            [void][System.IO.Directory]::CreateDirectory($site)
            [System.IO.File]::WriteAllBytes((Join-Path $site 'weights.bin'), [byte[]]::new(2MB))
            [System.IO.File]::WriteAllLines((Join-Path $site 'project.pth'), $Lines)
        }
        return $venv
    }
}

Describe 'The virtualenv sweep' {
    It 'removes a virtualenv whose recorded sources are all gone and keeps every other kind' {
        $venvs = Join-Path $TestDrive 'cache\pypoetry\virtualenvs'
        $alive = Join-Path $TestDrive 'stage\run-alive\src'
        [void][System.IO.Directory]::CreateDirectory($alive)
        $gone = Join-Path $TestDrive 'stage\run-retired\src'
        $orphan = New-Virtualenv $venvs 'demo-orphan-py3.11' @($gone, 'import _virtualenv')
        $serving = New-Virtualenv $venvs 'demo-alive-py3.11' @($gone, $alive)
        $unknown = New-Virtualenv $venvs 'demo-unknown-py3.11' @('import _virtualenv', 'relative\path')
        $creating = New-Virtualenv $venvs 'demo-creating-py3.11' @() -NoSitePackages

        Invoke-Rendered 'dialect-venv-sweep' @{ Venvs = $venvs } | Should -BeExactly 'venv-sweep: removed 1 of 4 (2 MB)'

        [System.IO.Directory]::Exists($orphan) | Should -BeFalse
        [System.IO.Directory]::Exists($serving) | Should -BeTrue
        [System.IO.Directory]::Exists($unknown) | Should -BeTrue
        [System.IO.Directory]::Exists($creating) | Should -BeTrue
    }
    It 'reads a source path written with forward slashes as absolute too' {
        $venvs = Join-Path $TestDrive 'slashes\virtualenvs'
        $orphan = New-Virtualenv $venvs 'demo-py3.11' @('C:/fleet/stage/never-staged-here/src')
        Invoke-Rendered 'dialect-venv-sweep' @{ Venvs = $venvs } | Should -BeExactly 'venv-sweep: removed 1 of 1 (2 MB)'
        [System.IO.Directory]::Exists($orphan) | Should -BeFalse
    }
    It 'reports nothing removed when the node has no virtualenvs directory yet' {
        Invoke-Rendered 'dialect-venv-sweep' @{ Venvs = (Join-Path $TestDrive 'no-cache\virtualenvs') } |
            Should -BeExactly 'venv-sweep: removed 0 of 0 (0 MB)'
    }
}
