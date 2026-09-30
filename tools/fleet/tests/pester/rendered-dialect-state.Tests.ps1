Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# The Windows dialect's probes and the scripts that make, empty and read a
# dispatch's directory, executed from their committed copies under rendered/
# (fleet.core.rendered_powershell, MCPs board task d69786fa, A2). Every
# location is a parameter defaulting to the example dispatch's, so each case
# points it at a directory under $TestDrive; the probes read this machine and
# run as rendered.

BeforeAll {
    . (Join-Path $PSScriptRoot 'rendered-fixtures.ps1')

    function Read-ObserveDocument {
        param([string]$Directory)
        $said = @(Invoke-Rendered 'dialect-observe-sessions' @{ SessionsDirectory = $Directory })
        $said.Count | Should -Be 1
        return ($said[0] | ConvertFrom-Json)
    }
}

Describe 'The capacity probe' {
    It 'reports free memory, free disk and logical cores, each as one key=value line' {
        $said = @(Invoke-Rendered 'dialect-capacity-probe' @{})
        $said.Count | Should -Be 3
        $said[0] | Should -Match '^free_ram_gb=[0-9,]+\.[0-9]{3}$'
        $said[1] | Should -Match '^free_disk_gb=[0-9,]+\.[0-9]{3}$'
        $said[2] | Should -BeExactly "logical_cores=$([Environment]::ProcessorCount)"
    }
}

Describe 'The session observer' {
    It 'reports no records, as win32 and the lowercased host, for a profile that never ran Claude Code' {
        $document = Read-ObserveDocument (Join-Path $TestDrive 'never\.claude\sessions')
        $document.platform | Should -BeExactly 'win32'
        $document.hostname | Should -BeExactly $env:COMPUTERNAME.ToLowerInvariant()
        @($document.records).Count | Should -Be 0
    }
    It 'reports every registration verbatim, and nothing that is not JSON' {
        $sessions = Join-Path $TestDrive 'ran\.claude\sessions'
        [void][System.IO.Directory]::CreateDirectory($sessions)
        [System.IO.File]::WriteAllText((Join-Path $sessions '4242.json'), '{"sessionId":"d31e5228-3269-4a22-a27f-e535cbef1894","pid":4242,"pidDomain":"win32:serendipity"}')
        [System.IO.File]::WriteAllText((Join-Path $sessions 'notes.txt'), 'not a record')
        $document = Read-ObserveDocument $sessions
        @($document.records).Count | Should -Be 1
        $document.records[0].sessionId | Should -BeExactly 'd31e5228-3269-4a22-a27f-e535cbef1894'
        $document.records[0].pid | Should -Be 4242
        $document.records[0].pidDomain | Should -BeExactly 'win32:serendipity'
    }
    It 'reads the profile''s own sessions directory when none is named, as a node runs it' {
        $userProfile = Join-Path $TestDrive 'profile'
        $sessions = Join-Path $userProfile '.claude\sessions'
        [void][System.IO.Directory]::CreateDirectory($sessions)
        [System.IO.File]::WriteAllText((Join-Path $sessions '7.json'), '{"pid":7}')
        $saved = $env:USERPROFILE
        try {
            # $HOME follows USERPROFILE in a new process (measured on the hub
            # 2026-09-21), so the child reads the profile laid out here.
            $env:USERPROFILE = $userProfile
            $said = & "$PSHOME\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File (Join-Path $script:rendered 'dialect-observe-sessions.ps1')
            $exit = $LASTEXITCODE
        } finally {
            $env:USERPROFILE = $saved
        }
        $exit | Should -Be 0
        $document = @($said)[0] | ConvertFrom-Json
        @($document.records).Count | Should -Be 1
        $document.records[0].pid | Should -Be 7
    }
}

Describe 'A dispatch directory' {
    It 'is made with its parents, brackets taken literally, and making it again changes nothing' {
        $directory = Join-Path $TestDrive 'stage\run-[1]\nested'
        Invoke-Rendered 'dialect-make-directory' @{ Directory = $directory }
        Invoke-Rendered 'dialect-make-directory' @{ Directory = $directory }
        [System.IO.Directory]::Exists($directory) | Should -BeTrue
    }
    It 'is emptied read-only files and all, and made when it was never there' {
        $directory = Join-Path $TestDrive 'companion'
        [void][System.IO.Directory]::CreateDirectory((Join-Path $directory '.git\objects'))
        $object = Join-Path $directory '.git\objects\ab'
        [System.IO.File]::WriteAllText($object, 'loose object')
        [System.IO.File]::SetAttributes($object, [System.IO.FileAttributes]::ReadOnly)
        Invoke-Rendered 'dialect-reset-directory' @{ Directory = $directory }
        @([System.IO.Directory]::GetFileSystemEntries($directory)).Count | Should -Be 0
        $fresh = Join-Path $TestDrive 'never-made'
        Invoke-Rendered 'dialect-reset-directory' @{ Directory = $fresh }
        [System.IO.Directory]::Exists($fresh) | Should -BeTrue
    }
}

Describe 'Retiring a settled dispatch' {
    BeforeAll {
        function Initialize-SettledRun {
            param([string]$Root, [switch]$Transcript)
            $target = Join-Path $Root 'run-[1]'
            $objects = Join-Path $target '.git\objects'
            [void][System.IO.Directory]::CreateDirectory($objects)
            $object = Join-Path $objects 'ab'
            [System.IO.File]::WriteAllText($object, 'loose object')
            [System.IO.File]::SetAttributes($object, [System.IO.FileAttributes]::ReadOnly)
            $log = Join-Path $target 'result.txt.log'
            if ($Transcript) {
                [System.IO.File]::WriteAllText($log, '887 passed')
            }
            # The staging directory beside the export: the archive and the
            # scripts that staged it, none of them inside the tree.
            $staging = Join-Path $Root 'run-[1].stage'
            [void][System.IO.Directory]::CreateDirectory($staging)
            [System.IO.File]::WriteAllText((Join-Path $staging 'tree.tgz'), 'archive')
            $scripts = @('mkdir-run.ps1', 'mkdir-run.stage.ps1', 'stop-run.ps1', 'retire-run.ps1') | ForEach-Object { Join-Path $Root $_ }
            foreach ($script in $scripts) {
                [System.IO.File]::WriteAllText($script, '# a root script')
            }
            [void][System.IO.Directory]::CreateDirectory((Join-Path $Root 'cache'))
            return @{
                Target = $target
                Staging = $staging
                Log = $log
                Retained = Join-Path $Root 'logs\run-[1].log'
                Script0 = $scripts[0]
                Script1 = $scripts[1]
                Script2 = $scripts[2]
                Script3 = $scripts[3]
            }
        }
    }
    It 'keeps the transcript, removes the tree read-only files and all and its staging directory, and removes only the run''s root scripts' {
        $root = Join-Path $TestDrive 'kept'
        $run = Initialize-SettledRun -Root $root -Transcript
        Invoke-Rendered 'dialect-retire' $run
        [System.IO.File]::ReadAllText($run.Retained) | Should -BeExactly '887 passed'
        [System.IO.Directory]::Exists($run.Target) | Should -BeFalse
        @([System.IO.Directory]::GetFileSystemEntries($root) | ForEach-Object { Split-Path -Leaf $_ } | Sort-Object) |
            Should -Be @('cache', 'logs')
    }
    It 'retires a run that wrote no transcript, and a second retire finds nothing left and is not an error' {
        $root = Join-Path $TestDrive 'untranscribed'
        $run = Initialize-SettledRun -Root $root
        Invoke-Rendered 'dialect-retire' $run
        Invoke-Rendered 'dialect-retire' $run
        @([System.IO.Directory]::GetFileSystemEntries((Join-Path $root 'logs'))).Count | Should -Be 0
        [System.IO.Directory]::Exists($run.Target) | Should -BeFalse
        [System.IO.Directory]::Exists($run.Staging) | Should -BeFalse
    }
    It 'deletes the run''s own scheduled task, which nothing else ever removed, and a second retire finds none' {
        # A real root-folder task under a name no fleet run can carry, the
        # way launch_script leaves one behind for every finished run.
        $name = "fleet-pester-retire-$([guid]::NewGuid().ToString('N'))"
        & schtasks.exe /Create /TN $name /TR 'cmd.exe /c exit 0' /SC ONCE /ST 23:59 /F | Out-Null
        $LASTEXITCODE | Should -Be 0
        $run = Initialize-SettledRun -Root (Join-Path $TestDrive 'tasked')
        $run.TaskName = $name
        Invoke-Rendered 'dialect-retire' $run
        @(Get-ScheduledTask -TaskPath '\' | Where-Object { $_.TaskName -eq $name }).Count | Should -Be 0
        Invoke-Rendered 'dialect-retire' $run
        @(Get-ScheduledTask -TaskPath '\' | Where-Object { $_.TaskName -eq $name }).Count | Should -Be 0
    }
}

Describe 'What the node reports back' {
    It 'digests the landed archive in lower case' {
        $target = Join-Path $TestDrive 'digested'
        [void][System.IO.Directory]::CreateDirectory($target)
        [System.IO.File]::WriteAllText((Join-Path $target 'tree.tgz'), 'abc')
        Invoke-Rendered 'dialect-digest' @{ Target = $target } |
            Should -BeExactly 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
    }
    It 'prints nothing while the build has written no result' {
        $target = Join-Path $TestDrive 'running'
        [void][System.IO.Directory]::CreateDirectory($target)
        @(Invoke-Rendered 'dialect-result' @{ Target = $target }).Count | Should -Be 0
    }
    It 'prints the status and the UTC epoch second the build wrote it' {
        $target = Join-Path $TestDrive 'finished'
        [void][System.IO.Directory]::CreateDirectory($target)
        $result = Join-Path $target 'result.txt'
        [System.IO.File]::WriteAllText($result, "3`r`n")
        [System.IO.File]::SetLastWriteTimeUtc($result, [datetime]::new(2026, 9, 27, 17, 0, 0, [System.DateTimeKind]::Utc))
        Invoke-Rendered 'dialect-result' @{ Target = $target } | Should -BeExactly '3 1790528400'
    }
}
