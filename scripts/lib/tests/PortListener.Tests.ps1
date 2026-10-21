Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# scripts/lib/PortListener.ps1, read against real sockets this process opens
# on ports the operating system hands out (board task 2a044a37).

BeforeAll {
    . (Join-Path (Split-Path -Parent $PSScriptRoot) 'PortListener.ps1')

    function Get-FreePort {
        $probe = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
        $probe.Start()
        $port = ([System.Net.IPEndPoint]$probe.LocalEndpoint).Port
        $probe.Stop()
        return $port
    }
}

Describe 'Get-PortListener' {
    It 'names this process for a port it listens on, once, on any address and on its own' {
        $held = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
        $held.Start()
        try {
            $port = ([System.Net.IPEndPoint]$held.LocalEndpoint).Port
            Get-PortListener -Port $port | Should -Be @([uint32]$PID)
            Get-PortListener -Port $port -Address '127.0.0.1' | Should -Be @([uint32]$PID)
        } finally {
            $held.Stop()
        }
    }
    It 'answers a free port with no process, not an error' {
        @(Get-PortListener -Port (Get-FreePort)).Count | Should -Be 0
    }
    It 'answers a port held on another address with no process for this one' {
        $held = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
        $held.Start()
        try {
            $port = ([System.Net.IPEndPoint]$held.LocalEndpoint).Port
            @(Get-PortListener -Port $port -Address '0.0.0.0').Count | Should -Be 0
        } finally {
            $held.Stop()
        }
    }
}
