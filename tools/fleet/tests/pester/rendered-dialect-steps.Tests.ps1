Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# The Windows dialect's checked scripts, executed from their committed
# renders under rendered/ (fleet.core.dialect_windows.checked_script, MCPs
# board task d69786fa, A2): the extract, the git steps that make a staged
# tree a one-commit repository, and those that clone a companion from its
# bundle (MCPs board task 2026dfbc). Their one piece of
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
    It 'unpacks the archive landed beside the dispatch into it, with the node''s clock' {
        $tar = Initialize-StandIn 0 -Append
        Invoke-Rendered 'dialect-extract' @{ Tar = $tar.Path }
        Read-Record $tar.Record | Should -Be @("-xzmf $($script:target).stage/tree.tgz -C $($script:target)")
    }
    It 'ends with tar''s own status when tar refuses' {
        $tar = Initialize-StandIn 2 -Append
        { Invoke-Rendered 'dialect-extract' @{ Tar = $tar.Path } } | Should -Throw 'TEST_RENDER_EXITED: dialect-extract exited 2'
    }
}

Describe 'The staged tree''s repository' {
    It 'initialises, indexes every file the commit carries, and commits under the export identity' {
        $git = Initialize-StandIn 0 -Append
        Invoke-Rendered 'dialect-init-repository' @{ Git = $git.Path }
        Read-Record $git.Record | Should -Be @(
            "-C $($script:target) init --quiet",
            "-C $($script:target) add --all --force",
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
    It 'clones the bundle onto origin/main and main, then proves HEAD is the bundled commit' {
        # A real clone with its history (MCPs board task 2026dfbc), whose
        # origin/main still lets a check that runs MCPs' published maketools
        # read ../MCPs on a node (MCPs board task a8ee9b21).
        $git = Initialize-StandIn 0 -Append
        Invoke-Rendered 'dialect-companion-repository' @{ Git = $git.Path }
        Read-Record $git.Record | Should -Be @(
            "-C $($script:companion) init --quiet",
            "-C $($script:companion) fetch --quiet --no-tags $($script:companion).stage/tree.tgz +refs/fleet/companion:refs/remotes/origin/main",
            "-C $($script:companion) checkout --quiet -B main refs/remotes/origin/main",
            "-C $($script:companion) merge-base --is-ancestor HEAD 5f389cb3d9bdd2e9b49e8df6683a6fee71a359b2",
            "-C $($script:companion) merge-base --is-ancestor 5f389cb3d9bdd2e9b49e8df6683a6fee71a359b2 HEAD"
        )
    }
    It 'stops at the first step git refuses, with git''s status' {
        $git = Initialize-StandIn 1 -Append
        { Invoke-Rendered 'dialect-companion-repository' @{ Git = $git.Path } } | Should -Throw 'TEST_RENDER_EXITED: dialect-companion-repository exited 1'
        Read-Record $git.Record | Should -Be @("-C $($script:companion) init --quiet")
    }
}
