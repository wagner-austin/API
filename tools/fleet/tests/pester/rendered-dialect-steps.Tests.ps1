Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# The Windows dialect's checked scripts, executed from their committed
# renders under rendered/ (fleet.core.dialect_windows.checked_script, MCPs
# board task d69786fa, A2): the extract, and the git steps that make a
# staged tree and a companion one-commit repositories. Their one piece of
# logic is "run each step, and end with the first non-zero status", so tar
# and git are stand-ins passed through the parameters the renders name them
# by, recording the exact argument vector each received.

BeforeAll {
    . (Join-Path $PSScriptRoot 'rendered-fixtures.ps1')
    $script:target = 'C:/fleet/stage/MCPs-packages-maketools-1790000000'
    $script:companion = 'C:/fleet/stage/MCPs'

    function Read-Record {
        param([string]$Path)
        return [string[]]@([System.IO.File]::ReadAllLines($Path) | ForEach-Object { $_.TrimEnd() })
    }
}

Describe 'The extract' {
    It 'unpacks the landed archive into the dispatch with the node''s clock' {
        $tar = Initialize-StandIn 0 -Append
        Invoke-Rendered 'dialect-extract' @{ Tar = $tar.Path }
        Read-Record $tar.Record | Should -Be @("-xzmf $($script:target)/tree.tgz -C $($script:target)")
    }
    It 'ends with tar''s own status when tar refuses' {
        $tar = Initialize-StandIn 2 -Append
        { Invoke-Rendered 'dialect-extract' @{ Tar = $tar.Path } } | Should -Throw 'TEST_RENDER_EXITED: dialect-extract exited 2'
    }
}

Describe 'The staged tree''s repository' {
    It 'initialises, indexes under the tree''s own ignore rules, and commits under the export identity' {
        $git = Initialize-StandIn 0 -Append
        Invoke-Rendered 'dialect-init-repository' @{ Git = $git.Path }
        Read-Record $git.Record | Should -Be @(
            "-C $($script:target) init --quiet",
            "-C $($script:target) add --all",
            "-C $($script:target) -c user.name=fleet -c user.email=fleet@corvis.invalid commit --quiet --message `"fleet export MCPs-packages-maketools-1790000000`""
        )
    }
    It 'stops at the first step git refuses, with git''s status' {
        $git = Initialize-StandIn 128 -Append
        { Invoke-Rendered 'dialect-init-repository' @{ Git = $git.Path } } | Should -Throw 'TEST_RENDER_EXITED: dialect-init-repository exited 128'
        Read-Record $git.Record | Should -Be @("-C $($script:target) init --quiet")
    }
}

Describe 'A companion''s repository' {
    It 'indexes every file of the export with --force and names the commit it came from' {
        $git = Initialize-StandIn 0 -Append
        Invoke-Rendered 'dialect-companion-repository' @{ Git = $git.Path }
        Read-Record $git.Record | Should -Be @(
            "-C $($script:companion) init --quiet",
            "-C $($script:companion) add --all --force",
            "-C $($script:companion) -c user.name=fleet -c user.email=fleet@corvis.invalid commit --quiet --message `"fleet companion export 5f389cb3d9bdd2e9b49e8df6683a6fee71a359b2`""
        )
    }
    It 'stops at the first step git refuses, with git''s status' {
        $git = Initialize-StandIn 1 -Append
        { Invoke-Rendered 'dialect-companion-repository' @{ Git = $git.Path } } | Should -Throw 'TEST_RENDER_EXITED: dialect-companion-repository exited 1'
        Read-Record $git.Record | Should -Be @("-C $($script:companion) init --quiet")
    }
}
