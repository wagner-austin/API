<#
.SYNOPSIS
    Start and stop grandma-api's host-mode API and web server (make up, make
    down on Windows).
.DESCRIPTION
    The API is hypercorn over grandma_api.asgi:app with the web directory's
    certificate; the web side is the built frontend served by
    scripts.webserver. Both run under poetry, hidden, with their streams in
    logs\. The docker route (`make up-grandma` at the root) is the portable
    one; this is the host mode.

    READY MEANS LISTENING. Until 2026-09-27 the start waited thirty seconds
    for the API's port and went on to report success whether or not it had
    opened, and reported the web server ready after a fixed two-second
    sleep. Each now waits for its own port and refuses by name when it does
    not open (GRANDMA_API_NOT_READY, GRANDMA_WEB_NOT_READY), and the
    frontend build's exit code is read (GRANDMA_WEB_BUILD_FAILED) where it
    was discarded with its stderr (MCPs board task d69786fa).

    A port is answered by the listening sockets on it, read from the full
    table: Get-NetTCPConnection -LocalPort throws when nothing matches,
    which the old script silenced with SilentlyContinue along with every
    other failure to read the table.
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-GrandmaListener {
    <#
    .SYNOPSIS
        The processes listening on a local port.
    .PARAMETER Port
        The port.
    .OUTPUTS
        UInt32[]: owning process ids, each once, System (pid 4) excluded.
    #>
    [OutputType([uint32[]])]
    param([Parameter(Mandatory)][int]$Port)
    $owners = @(Get-NetTCPConnection -State Listen | Where-Object { $_.LocalPort -eq $Port -and $_.OwningProcess -gt 4 } |
        ForEach-Object { $_.OwningProcess } | Select-Object -Unique)
    return [uint32[]]$owners
}

function Wait-GrandmaListener {
    <#
    .SYNOPSIS
        Wait until something listens on a port, or refuse by name.
    .PARAMETER Port
        The port.
    .PARAMETER Seconds
        How long to wait.
    .PARAMETER Code
        The error code a port that never opens is named with.
    #>
    param([Parameter(Mandatory)][int]$Port, [Parameter(Mandatory)][int]$Seconds, [Parameter(Mandatory)][string]$Code)
    $deadline = [DateTime]::UtcNow.AddSeconds($Seconds)
    while (@(Get-GrandmaListener $Port).Count -eq 0) {
        if ([DateTime]::UtcNow -ge $deadline) {
            throw "${Code}: nothing listened on port $Port within ${Seconds}s; see the logs directory"
        }
        Start-Sleep -Milliseconds 250
    }
}

function Import-GrandmaEnvironment {
    <#
    .SYNOPSIS
        Load KEY=value lines from a .env file into this process's environment.
    .PARAMETER Path
        The .env file; absent means nothing to load.
    .OUTPUTS
        Int32: how many variables were set.
    #>
    [OutputType([int])]
    param([Parameter(Mandatory)][string]$Path)
    if (-not [System.IO.File]::Exists($Path)) {
        return 0
    }
    $count = 0
    foreach ($line in [System.IO.File]::ReadAllLines($Path)) {
        if ($line -match '^\s*([^#][^=]*?)\s*=\s*(.*)$') {
            [Environment]::SetEnvironmentVariable($Matches[1].Trim(), $Matches[2].Trim().Trim('"', "'"), 'Process')
            $count++
        }
    }
    return $count
}

function Start-GrandmaService {
    <#
    .SYNOPSIS
        Start whichever of the API and the web server is not listening.
    .PARAMETER ProjectRoot
        services\grandma-api.
    .PARAMETER ApiPort
        The API's port.
    .PARAMETER WebPort
        The web server's port.
    .PARAMETER Poetry
        The poetry executable.
    .PARAMETER Npm
        The npm executable.
    .PARAMETER ReadySeconds
        How long each side may take to listen.
    .OUTPUTS
        String[]: what was started, in order; none when both already ran.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([string[]])]
    param(
        [Parameter(Mandatory)][string]$ProjectRoot,
        [Parameter(Mandatory)][int]$ApiPort,
        [Parameter(Mandatory)][int]$WebPort,
        [Parameter(Mandatory)][string]$Poetry,
        [Parameter(Mandatory)][string]$Npm,
        [Parameter(Mandatory)][int]$ReadySeconds
    )
    $webDir = Join-Path $ProjectRoot 'web'
    $logDir = Join-Path $ProjectRoot 'logs'
    [void][System.IO.Directory]::CreateDirectory($logDir)
    $started = [System.Collections.Generic.List[string]]::new()
    if (@(Get-GrandmaListener $ApiPort).Count -eq 0 -and $PSCmdlet.ShouldProcess("port $ApiPort", 'Start the grandma-api API')) {
        $apiArguments = @('run', 'hypercorn', 'grandma_api.asgi:app', '--bind', "0.0.0.0:$ApiPort", '--reload',
            '--certfile', (Join-Path $webDir 'cert.pem'), '--keyfile', (Join-Path $webDir 'key.pem'))
        [void](Start-Process -FilePath $Poetry -ArgumentList $apiArguments -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $logDir 'api.log') -RedirectStandardError (Join-Path $logDir 'api-err.log'))
        Wait-GrandmaListener $ApiPort $ReadySeconds 'GRANDMA_API_NOT_READY'
        $started.Add("api:$ApiPort")
    }
    if (@(Get-GrandmaListener $WebPort).Count -eq 0 -and $PSCmdlet.ShouldProcess("port $WebPort", 'Build and start the grandma-api web server')) {
        Push-Location -LiteralPath $webDir
        try {
            $buildOutput = & $Npm run build
            $buildExit = $LASTEXITCODE
        } finally { Pop-Location }
        if ($buildExit -ne 0) {
            throw "GRANDMA_WEB_BUILD_FAILED: npm run build exited ${buildExit}: $($buildOutput -join "`n")"
        }
        [void](Start-Process -FilePath $Poetry -ArgumentList @('run', 'python', '-m', 'scripts.webserver', "$WebPort", $webDir) `
            -WorkingDirectory $ProjectRoot -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput (Join-Path $logDir 'web.log') -RedirectStandardError (Join-Path $logDir 'web-err.log'))
        Wait-GrandmaListener $WebPort $ReadySeconds 'GRANDMA_WEB_NOT_READY'
        $started.Add("web:$WebPort")
    }
    return [string[]]$started.ToArray()
}

function Stop-GrandmaService {
    <#
    .SYNOPSIS
        End the process tree listening on each port.
    .DESCRIPTION
        taskkill /T /F by pid, never by name. Exit 128 means the process was
        already gone, which a tree ended through its parent makes routine;
        any other non-zero is refused as GRANDMA_STOP_FAILED.
    .PARAMETER Ports
        The ports.
    .PARAMETER Taskkill
        taskkill.exe's absolute path.
    .OUTPUTS
        String[]: "pid on port" for each process ended.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    [OutputType([string[]])]
    param([Parameter(Mandatory)][int[]]$Ports, [Parameter(Mandatory)][string]$Taskkill)
    $ended = [System.Collections.Generic.List[string]]::new()
    foreach ($port in $Ports) {
        foreach ($owner in @(Get-GrandmaListener $port)) {
            if ($PSCmdlet.ShouldProcess("pid $owner on port $port", 'End the process tree')) {
                $said = & $Taskkill /F /T /PID $owner
                $killExit = $LASTEXITCODE
                if ($killExit -ne 0 -and $killExit -ne 128) {
                    throw "GRANDMA_STOP_FAILED: taskkill for pid $owner on port $port exited ${killExit}: $($said -join ' ')"
                }
                $ended.Add("$owner on $port")
            }
        }
    }
    return [string[]]$ended.ToArray()
}
