Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# The Windows dialect's probes and the scripts that make, empty and read a
# dispatch's directory, executed from their committed copies under rendered/
# (fleet.core.rendered_powershell, MCPs board task d69786fa, A2); the result
# render, which reads a build's scheduled task, is rendered-dialect-result's
# (MCPs board task 4e3afe4f). Every
# location is a parameter defaulting to the example dispatch's, so each case
# points it at a directory under $TestDrive; the probes read this machine and
# run as rendered.

BeforeAll {
    . (Join-Path $PSScriptRoot 'rendered-fixtures.ps1')

    function Read-ObserveDocument {
        param([string]$Directory)
        $said = @(Invoke-Rendered 'dialect-observe-sessions.ps1' @{ SessionsDirectory = $Directory })
        $said.Count | Should -Be 1
        return ($said[0] | ConvertFrom-Json)
    }
}

Describe 'The capacity probe' {
    BeforeAll {
        # The reservations live under the node account's profile (MCPs board
        # task 84a2b690, fleet.core.reservations), so each case points
        # USERPROFILE at a profile under $TestDrive and puts it back.
        function Invoke-ReservedProbe {
            param([string]$ProfileDirectory, [hashtable]$Files)
            $held = Join-Path $ProfileDirectory '.fleet\reservations'
            [void][System.IO.Directory]::CreateDirectory($held)
            foreach ($name in $Files.Keys) {
                [System.IO.File]::WriteAllText((Join-Path $held $name), $Files[$name])
            }
            $saved = $env:USERPROFILE
            try {
                $env:USERPROFILE = $ProfileDirectory
                return @(Invoke-Rendered 'dialect-capacity-probe.ps1' @{})
            } finally {
                $env:USERPROFILE = $saved
            }
        }
    }
    It 'reports free memory, free disk, logical cores and reserved cores, each as one key=value line' {
        $said = @(Invoke-ReservedProbe (Join-Path $TestDrive 'none') @{})
        $said.Count | Should -Be 4
        $said[0] | Should -Match '^free_ram_gb=[0-9,]+\.[0-9]{3}$'
        $said[1] | Should -Match '^free_disk_gb=[0-9,]+\.[0-9]{3}$'
        $said[2] | Should -BeExactly "logical_cores=$([Environment]::ProcessorCount)"
        $said[3] | Should -BeExactly 'reserved_cores=0'
    }
    It 'reads no reservations for a profile that has no reservations directory' {
        $userProfile = Join-Path $TestDrive 'bare'
        [void][System.IO.Directory]::CreateDirectory($userProfile)
        $saved = $env:USERPROFILE
        try {
            $env:USERPROFILE = $userProfile
            $said = @(Invoke-Rendered 'dialect-capacity-probe.ps1' @{})
        } finally {
            $env:USERPROFILE = $saved
        }
        $said[-1] | Should -BeExactly 'reserved_cores=0'
    }
    It 'sums the cores of every live reservation and charges an expired one nothing' {
        $said = @(Invoke-ReservedProbe (Join-Path $TestDrive 'held') @{
            'vm.reservation'    = "holder=vm smoke`ncores=4`n`nuntil_unix=99999999999`n"
            'build.reservation' = "holder=build`r`ncores=3`r`nuntil_unix=99999999999`r`n"
            'old.reservation'   = "holder=old`ncores=2`nuntil_unix=1`n"
            'notes.txt'         = "cores=9`nuntil_unix=99999999999`n"
        })
        $said[-1] | Should -BeExactly 'reserved_cores=7'
    }
    It 'names a reservation it cannot read and ends the probe' -TestCases @(
        @{ Text = "holder=vm`ncores=four`nuntil_unix=99999999999`n" }
        @{ Text = "holder=vm`nuntil_unix=99999999999`n" }
        @{ Text = "holder=vm`ncores=4`n" }
        @{ Text = "holder=vm`ncores=4`nuntil_unix=-1`n" }
    ) {
        param([string]$Text)
        $userProfile = Join-Path $TestDrive ([guid]::NewGuid().ToString('N'))
        $said = @(Invoke-ReservedProbe $userProfile @{ 'bad.reservation' = $Text })
        $said[-1] | Should -BeExactly "reservation_unreadable=$(Join-Path $userProfile '.fleet\reservations\bad.reservation')"
        ($said -match '^reserved_cores=').Count | Should -Be 0
    }
}

Describe 'The session observer' {
    It 'reports no records, as win32, the lowercased host and its pidDomain, for a profile that never ran Claude Code' {
        $document = Read-ObserveDocument (Join-Path $TestDrive 'never\.claude\sessions')
        $document.platform | Should -BeExactly 'win32'
        $document.hostname | Should -BeExactly $env:COMPUTERNAME.ToLowerInvariant()
        $document.pidDomain | Should -BeExactly ('win32:' + $env:COMPUTERNAME.ToLowerInvariant())
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
        Invoke-Rendered 'dialect-make-directory.ps1' @{ Directory = $directory }
        Invoke-Rendered 'dialect-make-directory.ps1' @{ Directory = $directory }
        [System.IO.Directory]::Exists($directory) | Should -BeTrue
    }
    It 'is emptied read-only files and all, and made when it was never there' {
        $directory = Join-Path $TestDrive 'companion'
        [void][System.IO.Directory]::CreateDirectory((Join-Path $directory '.git\objects'))
        $object = Join-Path $directory '.git\objects\ab'
        [System.IO.File]::WriteAllText($object, 'loose object')
        [System.IO.File]::SetAttributes($object, [System.IO.FileAttributes]::ReadOnly)
        Invoke-Rendered 'dialect-reset-directory.ps1' @{ Directory = $directory }
        @([System.IO.Directory]::GetFileSystemEntries($directory)).Count | Should -Be 0
        $fresh = Join-Path $TestDrive 'never-made'
        Invoke-Rendered 'dialect-reset-directory.ps1' @{ Directory = $fresh }
        [System.IO.Directory]::Exists($fresh) | Should -BeTrue
    }
}

Describe 'Retiring a settled dispatch' {
    BeforeAll {
        function Test-RootTaskRegistered {
            <#
            .SYNOPSIS
                Whether a task of this name is registered in the scheduler's
                root folder, read through Schedule.Service as the retire
                render reads it, not through Get-ScheduledTask, which reads
                every task's definition and fails while another is deleted.
            .PARAMETER Name
                The task's name.
            #>
            param([string]$Name)
            $scheduler = New-Object -ComObject Schedule.Service
            $scheduler.Connect()
            return @($scheduler.GetFolder('\').GetTasks(1) | Where-Object { $_.Name -eq $Name }).Count -gt 0
        }
        function Initialize-FileHolder {
            <#
            .SYNOPSIS
                A process that holds a file open the way a build's leftover
                holds its transcript: open for writing, shared for reading
                and writing as cmd.exe's >> shares it, and not for delete,
                so the file can be neither moved nor removed while it runs.
            .PARAMETER Path
                The file it holds.
            .PARAMETER Release
                A file whose appearance makes it exit, closing its handle;
                without one it holds the file for ten minutes.
            .OUTPUTS
                System.Diagnostics.Process, returned once it holds the file.
            #>
            param([string]$Path, [string]$Release)
            $ready = "$Path.holding"
            $wait = @{ $true = 'Start-Sleep 600'; $false = "while (-not (Test-Path -LiteralPath '$Release')) { Start-Sleep -Milliseconds 100 }" }[$Release -eq '']
            $command = "`$held = [IO.File]::Open('$Path', 'Open', 'Write', 'ReadWrite'); [IO.File]::WriteAllText('$ready', 'held'); $wait"
            $holder = Start-Process -FilePath "$env:SystemRoot\System32\WindowsPowerShell\v1.0\powershell.exe" -ArgumentList '-NoProfile', '-Command', $command -PassThru -WindowStyle Hidden
            while (-not (Test-Path -LiteralPath $ready)) {
                $holder.HasExited | Should -BeFalse
                Start-Sleep -Milliseconds 100
            }
            Remove-Item -LiteralPath $ready
            return $holder
        }
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
        Invoke-Rendered 'dialect-retire.ps1' $run
        [System.IO.File]::ReadAllText($run.Retained) | Should -BeExactly '887 passed'
        [System.IO.Directory]::Exists($run.Target) | Should -BeFalse
        @([System.IO.Directory]::GetFileSystemEntries($root) | ForEach-Object { Split-Path -Leaf $_ } | Sort-Object) |
            Should -Be @('cache', 'logs')
    }
    It 'ends a leftover process holding the transcript, names it by pid and image, then keeps the transcript (MCPs e40bca34)' {
        # On sedona two orphaned bash.exe of a hung install step held a
        # run's transcript, and every retire stopped on Move-Item's
        # IOException for 26 hours; the retire now ends the holder first.
        $root = Join-Path $TestDrive 'held'
        $run = Initialize-SettledRun -Root $root -Transcript
        $holder = Initialize-FileHolder $run.Log
        { Move-Item -LiteralPath $run.Log -Destination (Join-Path $root 'moved.log') } |
            Should -Throw -ExpectedMessage '*being used by another process*'
        $said = @(Invoke-Rendered 'dialect-retire.ps1' $run)
        $said | Should -HaveCount 1
        $said[0] | Should -BeLike ("FLEET_RETIRE_HOLDER_ENDED: pid $($holder.Id) powershell.exe (*Start-Sleep 600*) held " + [WildcardPattern]::Escape($run.Log))
        $holder.HasExited | Should -BeTrue
        [System.IO.File]::ReadAllText($run.Retained) | Should -BeExactly '887 passed'
        [System.IO.Directory]::Exists($run.Target) | Should -BeFalse
    }
    It 'refuses by name to end a holder of a kind it may not end, leaving it and the transcript where they are' {
        # A service or critical process is not the run's to end; with no
        # kind allowed, an ordinary holder reads as one.
        $root = Join-Path $TestDrive 'protected'
        $run = Initialize-SettledRun -Root $root -Transcript
        $holder = Initialize-FileHolder $run.Log
        $run.EndableTypes = [int[]]@()
        { Invoke-Rendered 'dialect-retire.ps1' $run } |
            Should -Throw -ExpectedMessage ("FLEET_RETIRE_HOLDER_PROTECTED: pid $($holder.Id) powershell.exe (*), of Restart Manager type 5, holds " + [WildcardPattern]::Escape($run.Log))
        $holder.HasExited | Should -BeFalse
        Stop-Process -Id $holder.Id -Force
        $holder.WaitForExit()
        [System.IO.File]::ReadAllText($run.Log) | Should -BeExactly '887 passed'
        Test-Path -LiteralPath $run.Retained | Should -BeFalse
    }
    It 'retires a run that wrote no transcript, and a second retire finds nothing left and is not an error' {
        $root = Join-Path $TestDrive 'untranscribed'
        $run = Initialize-SettledRun -Root $root
        Invoke-Rendered 'dialect-retire.ps1' $run
        Invoke-Rendered 'dialect-retire.ps1' $run
        @([System.IO.Directory]::GetFileSystemEntries((Join-Path $root 'logs'))).Count | Should -Be 0
        [System.IO.Directory]::Exists($run.Target) | Should -BeFalse
        [System.IO.Directory]::Exists($run.Staging) | Should -BeFalse
    }
    It 'removes a tree past MAX_PATH and a junction in it, leaving what the junction points at (MCPs 74b13c20)' {
        # Remove-Item in Windows PowerShell 5.1 cannot reach past 260
        # characters on a node that does not enable long paths, which wedged
        # loki's runner on 2026-10-04; the tree is made through cmd's
        # verbatim path so this case needs no such node setting.
        $root = Join-Path $TestDrive 'deep'
        $run = Initialize-SettledRun -Root $root
        $deep = '\\?\' + $run.Target + '\' + ((1..8 | ForEach-Object { 'd' * 40 }) -join '\')
        & cmd.exe /d /c mkdir $deep
        $LASTEXITCODE | Should -Be 0
        & cmd.exe /d /c "echo x> `"$deep\f.txt`""
        $LASTEXITCODE | Should -Be 0
        $outside = Join-Path $TestDrive 'outside'
        [void][System.IO.Directory]::CreateDirectory($outside)
        [System.IO.File]::WriteAllText((Join-Path $outside 'keep.txt'), 'keep')
        & cmd.exe /d /c mklink /J (Join-Path $run.Target 'link') $outside | Out-Null
        $LASTEXITCODE | Should -Be 0
        Invoke-Rendered 'dialect-retire.ps1' $run
        [System.IO.Directory]::Exists($run.Target) | Should -BeFalse
        [System.IO.File]::ReadAllText((Join-Path $outside 'keep.txt')) | Should -BeExactly 'keep'
    }
    It 'stops by name when a directory survives its removal, counting a file too long to ask Restart Manager about' {
        # rd exits 0 whatever it leaves; this one removes nothing, so both
        # files stay, one of them past the 259 characters Restart Manager
        # registers, and no process holds either.
        $root = Join-Path $TestDrive 'survived'
        $run = Initialize-SettledRun -Root $root
        $deep = '\\?\' + $run.Target + '\' + ((1..6 | ForEach-Object { 'd' * 40 }) -join '\')
        & cmd.exe /d /c mkdir $deep
        $LASTEXITCODE | Should -Be 0
        & cmd.exe /d /c "echo x> `"$deep\f.txt`""
        $LASTEXITCODE | Should -Be 0
        $inert = Join-Path $TestDrive 'inert.cmd'
        [System.IO.File]::WriteAllText($inert, "@echo off`r`nexit /b 0`r`n", [System.Text.Encoding]::ASCII)
        $run.Cmd = $inert
        $run.HeldSeconds = 0
        { Invoke-Rendered 'dialect-retire.ps1' $run } |
            Should -Throw -ExpectedMessage ([WildcardPattern]::Escape("FLEET_RETIRE_INCOMPLETE: rd exited 0 and left $($run.Target) after 0 s, 2 files left, 1 of them too long for Restart Manager to ask about, held by no process Restart Manager names"))
        [System.IO.Directory]::Exists($run.Target) | Should -BeTrue
    }
    It 'removes a run directory again once another process lets go of a file in it, naming the holder it waited on (MCPs a703ec42)' {
        # On loki on 2026-10-21 a process the failure never named held the
        # staging directory's tree.tgz when the retire ran, and rd left the
        # directory. The stand-in runs the real rd and, the second time rd
        # leaves the directory, tells the holder to let go, so the retire
        # meets the hold, names it, and waits it out.
        $root = Join-Path $TestDrive 'held-briefly'
        $run = Initialize-SettledRun -Root $root
        $archive = Join-Path $run.Staging 'tree.tgz'
        $release = Join-Path $TestDrive 'held-briefly.release'
        $refused = Join-Path $TestDrive 'held-briefly.refused'
        $holder = Initialize-FileHolder $archive -Release $release
        { [System.IO.File]::Delete($archive) } | Should -Throw -ExpectedMessage '*being used by another process*'
        $rd = Initialize-Batch -Name 'cmd' -Body @(
            '"%SystemRoot%\System32\cmd.exe" %*',
            "if exist `"%~6`" if exist `"$refused`" echo released> `"$release`"",
            "if exist `"%~6`" echo refused> `"$refused`"",
            'exit /b 0'
        )
        $run.Cmd = $rd.Path
        $said = @(Invoke-Rendered 'dialect-retire.ps1' $run)
        $said | Should -HaveCount 1
        $said[0] | Should -BeLike ([WildcardPattern]::Escape("FLEET_RETIRE_HELD: rd exited 0 and left $($run.Staging), 1 files left, 0 of them too long for Restart Manager to ask about, held by pid $($holder.Id) powershell.exe (") + '*' + [WildcardPattern]::Escape('); removing it again for up to 30 s'))
        $holder.WaitForExit(10000) | Should -BeTrue
        [System.IO.Directory]::Exists($run.Staging) | Should -BeFalse
        [System.IO.Directory]::Exists($run.Target) | Should -BeFalse
        @(Read-CallRecord $rd) | Should -HaveCount 4
    }
    It 'names the process still holding a run file when the wait runs out, leaving the directory (MCPs a703ec42)' {
        $root = Join-Path $TestDrive 'held-on'
        $run = Initialize-SettledRun -Root $root
        $holder = Initialize-FileHolder (Join-Path $run.Staging 'tree.tgz')
        $run.HeldSeconds = 2
        try {
            { Invoke-Rendered 'dialect-retire.ps1' $run } |
                Should -Throw -ExpectedMessage ([WildcardPattern]::Escape("FLEET_RETIRE_INCOMPLETE: rd exited 0 and left $($run.Staging) after 2 s, 1 files left, 0 of them too long for Restart Manager to ask about, held by pid $($holder.Id) powershell.exe (") + '*Start-Sleep 600*)')
            $holder.HasExited | Should -BeFalse
        } finally {
            Stop-Process -Id $holder.Id -Force
            $holder.WaitForExit()
        }
        [System.IO.Directory]::Exists($run.Target) | Should -BeFalse
        [System.IO.File]::ReadAllText((Join-Path $run.Staging 'tree.tgz')) | Should -BeExactly 'archive'
    }
    It 'deletes the run''s own scheduled task, which nothing else ever removed, and a second retire finds none' {
        # A real root-folder task under a name no fleet run can carry, the
        # way launch_script leaves one behind for every finished run.
        $name = "fleet-pester-retire-$([guid]::NewGuid().ToString('N'))"
        & schtasks.exe /Create /TN $name /TR 'cmd.exe /c exit 0' /SC ONCE /ST 23:59 /F | Out-Null
        $LASTEXITCODE | Should -Be 0
        $run = Initialize-SettledRun -Root (Join-Path $TestDrive 'tasked')
        $run.TaskName = $name
        Test-RootTaskRegistered $name | Should -BeTrue
        Invoke-Rendered 'dialect-retire.ps1' $run
        Test-RootTaskRegistered $name | Should -BeFalse
        Invoke-Rendered 'dialect-retire.ps1' $run
        Test-RootTaskRegistered $name | Should -BeFalse
    }
}

Describe 'What the node reports back' {
    It 'digests the landed archive in lower case' {
        $target = Join-Path $TestDrive 'digested'
        [void][System.IO.Directory]::CreateDirectory($target)
        [System.IO.File]::WriteAllText((Join-Path $target 'tree.tgz'), 'abc')
        Invoke-Rendered 'dialect-digest.ps1' @{ Target = $target } |
            Should -BeExactly 'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'
    }
}
