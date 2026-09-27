Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# The Windows process forest the load sampler reads, from its committed
# render under rendered/ (fleet.core.runner_load, MCPs board task d69786fa,
# A2). It is a read of this machine's own processes, so it runs as it is:
# the forest it prints must hold the process running this suite, with its
# real parent and command line, in the three columns the sampler parses.

BeforeAll {
    . (Join-Path $PSScriptRoot 'rendered-fixtures.ps1')
}

Describe 'The Windows process forest' {
    It 'prints every process as pid, parent pid and command line, this one among them' {
        $said = [string[]]@(Invoke-Rendered 'load-forest' @{})
        $said.Count | Should -BeGreaterThan 1
        @($said | Where-Object { $_ -notmatch '^\d+ \d+ ' }) | Should -Be @()
        $self = Get-CimInstance -ClassName Win32_Process -Filter "ProcessId=$PID"
        @($said | Where-Object { $_.StartsWith("$PID ") }) | Should -Be @("$PID $($self.ParentProcessId) $($self.CommandLine)")
    }
    It 'prints a process with no command line as its two ids and nothing after them' {
        $said = [string[]]@(Invoke-Rendered 'load-forest' @{})
        @($said | Where-Object { $_.StartsWith('4 ') }) | Should -Be @('4 0 ')
    }
}
