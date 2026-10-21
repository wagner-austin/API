Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# The removal of the cache generations no run reads, executed from its
# committed copy under rendered/ (fleet.core.cache_generations, MCPs board
# task 346abc45). Its stage root is a parameter defaulting to the example
# stage root, so each case lays out a stage root under $TestDrive the way a
# Windows node holds one: generations named by the epoch second they started
# under cache\tools, the current one named in cache\tools\current, and a
# live run's directory holding the cache-generation marker its build wrote.

BeforeAll {
    . (Join-Path $PSScriptRoot 'rendered-fixtures.ps1')

    function Initialize-Generation {
        param([string]$Tools, [long]$SecondsAgo)
        $name = [string]([DateTimeOffset]::UtcNow.ToUnixTimeSeconds() - $SecondsAgo)
        $artifacts = Join-Path $Tools "$name\pypoetry\artifacts"
        [void][System.IO.Directory]::CreateDirectory($artifacts)
        [System.IO.File]::WriteAllBytes((Join-Path $artifacts 'torch.whl'), [byte[]]::new(2MB))
        return $name
    }

    function Initialize-Stage {
        $stage = Join-Path $TestDrive ('stage-' + [guid]::NewGuid().ToString('N'))
        $tools = Join-Path $stage 'cache\tools'
        [void][System.IO.Directory]::CreateDirectory($tools)
        return [pscustomobject]@{ Stage = $stage; Tools = $tools }
    }
}

Describe 'The cache generations removal' {
    It 'removes only a generation that is not current, that no run names and that has been out of use over an hour' {
        $layout = Initialize-Stage
        $unread = Initialize-Generation $layout.Tools (30 * 86400)
        $marked = Initialize-Generation $layout.Tools (16 * 86400)
        $current = Initialize-Generation $layout.Tools 7200
        $racing = Initialize-Generation $layout.Tools 60
        [System.IO.File]::WriteAllText((Join-Path $layout.Tools 'current'), $current)
        $run = Join-Path $layout.Stage 'MCPs-search-sedona-1792000000'
        [void][System.IO.Directory]::CreateDirectory($run)
        [System.IO.File]::WriteAllText((Join-Path $run 'cache-generation'), $marked)
        [void][System.IO.Directory]::CreateDirectory((Join-Path $layout.Stage 'logs'))

        Invoke-Rendered 'dialect-cache-generations.ps1' @{ StageRoot = $layout.Stage } |
            Should -BeExactly "cache-generations: $($layout.Stage)/cache/tools: generation $current current, removed 1 of 4 (2 MB)"

        Test-Path -LiteralPath (Join-Path $layout.Tools $unread) | Should -BeFalse
        foreach ($kept in $marked, $current, $racing) {
            Test-Path -LiteralPath (Join-Path $layout.Tools $kept) | Should -BeTrue
        }
    }
    It 'keeps every generation within the hour after the current one started' {
        $layout = Initialize-Stage
        $previous = Initialize-Generation $layout.Tools (15 * 86400)
        $current = Initialize-Generation $layout.Tools 600
        [System.IO.File]::WriteAllText((Join-Path $layout.Tools 'current'), $current)

        Invoke-Rendered 'dialect-cache-generations.ps1' @{ StageRoot = $layout.Stage } |
            Should -BeExactly "cache-generations: $($layout.Stage)/cache/tools: generation $current current, removed 0 of 2 (0 MB)"
        Test-Path -LiteralPath (Join-Path $layout.Tools $previous) | Should -BeTrue
    }
    It 'says so when no build on the node has started a generation' {
        $stage = Join-Path $TestDrive 'fresh-node'
        Invoke-Rendered 'dialect-cache-generations.ps1' @{ StageRoot = $stage } |
            Should -BeExactly "cache-generations: $stage/cache/tools: none yet"
    }
    It 'stops at an entry that is no generation, naming it' {
        $layout = Initialize-Stage
        $current = Initialize-Generation $layout.Tools 7200
        [System.IO.File]::WriteAllText((Join-Path $layout.Tools 'current'), $current)
        [void][System.IO.Directory]::CreateDirectory((Join-Path $layout.Tools 'npm'))

        { Invoke-Rendered 'dialect-cache-generations.ps1' @{ StageRoot = $layout.Stage } } |
            Should -Throw "cache-generations: *npm is no generation"
    }
    It 'defaults its stage root to the example dispatch''s' {
        Get-RenderedDefault 'dialect-cache-generations.ps1' 'StageRoot' | Should -BeExactly 'C:/fleet/stage'
    }
}
