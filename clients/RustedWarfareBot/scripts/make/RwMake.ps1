<#
.SYNOPSIS
    What the RustedWarfareBot Makefile's PowerShell recipes share: building
    the agent, running a match beside the game, and spawning the match
    service's processes detached.
.DESCRIPTION
    agent.ps1, selftest.ps1, host.ps1 and watch.ps1 each compiled the agent
    with javac; host.ps1 and watch.ps1 were one launcher twice; door.ps1 and
    fleet-up.ps1 read the database password and spawned through WMI the
    same way. Until 2026-09-27 each carried its own copy, none read every
    native exit, and the port waits ran a connect in a catch that discarded
    every error (MCPs board task d69786fa).

    Every tool is a parameter: the Makefile passes javac, jar and java
    already, and the rest default to what the recipes ran by name, so the
    suite runs each recipe against stand-ins.

    A PORT IS OPEN WHEN SOMETHING LISTENS ON IT, read from the listening
    socket table rather than by connecting: a connect needs a catch for
    every refusal, and the agent's channel is a listening socket either way.
#>
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Invoke-RwNative {
    <#
    .SYNOPSIS
        Run one tool and refuse a non-zero exit by name.
    .PARAMETER File
        The tool.
    .PARAMETER Arguments
        Its arguments.
    .PARAMETER Code
        The error code a failure is named with.
    .OUTPUTS
        String[]: the tool's standard output.
    #>
    [OutputType([string[]])]
    param([Parameter(Mandatory)][string]$File, [Parameter(Mandatory)][string[]]$Arguments, [Parameter(Mandatory)][string]$Code)
    $output = & $File @Arguments
    $exit = $LASTEXITCODE
    if ($exit -ne 0) {
        throw "${Code}: $File exited ${exit}: $(@($output) -join "`n")"
    }
    return [string[]]@($output)
}

function Invoke-RwAgentCompile {
    <#
    .SYNOPSIS
        Compile the agent's sources into a directory, warnings as errors.
    .PARAMETER Root
        The RustedWarfareBot directory.
    .PARAMETER Javac
        javac.
    .PARAMETER Release
        The bytecode level.
    .PARAMETER ClassesDir
        Where the classes go, relative to Root.
    #>
    param([Parameter(Mandatory)][string]$Root, [Parameter(Mandatory)][string]$Javac,
        [Parameter(Mandatory)][string]$Release, [Parameter(Mandatory)][string]$ClassesDir)
    $classes = Join-Path $Root $ClassesDir
    [void][System.IO.Directory]::CreateDirectory($classes)
    $sources = @(Get-ChildItem -LiteralPath (Join-Path $Root 'agent\src\rwbot\agent') -Filter '*.java' -File | ForEach-Object { $_.FullName })
    [void](Invoke-RwNative $Javac (@('--release', $Release, '-Xlint:all', '-Werror', '-d', $classes) + $sources) 'RW_JAVAC_FAILED')
}

function Invoke-RwAgentJar {
    <#
    .SYNOPSIS
        Package compiled agent classes with the agent's manifest.
    .PARAMETER Root
        The RustedWarfareBot directory.
    .PARAMETER Jar
        jar.
    .PARAMETER JarPath
        The jar to write, relative to Root.
    .PARAMETER ClassesDir
        The compiled classes, relative to Root.
    #>
    param([Parameter(Mandatory)][string]$Root, [Parameter(Mandatory)][string]$Jar,
        [Parameter(Mandatory)][string]$JarPath, [Parameter(Mandatory)][string]$ClassesDir)
    [void](Invoke-RwNative $Jar @('cfm', (Join-Path $Root $JarPath), (Join-Path $Root 'agent\manifest.mf'), '-C', (Join-Path $Root $ClassesDir), '.') 'RW_JAR_FAILED')
}

function Remove-RwBuildPath {
    <#
    .SYNOPSIS
        Remove a scratch build file or directory when it exists.
    .PARAMETER Path
        The path.
    #>
    [CmdletBinding(SupportsShouldProcess)]
    param([Parameter(Mandatory)][string]$Path)
    if ((Test-Path -LiteralPath $Path) -and $PSCmdlet.ShouldProcess($Path, 'Remove a scratch build path')) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
}

function Get-RwListener {
    <#
    .SYNOPSIS
        The processes listening on a local port, each once.
    .PARAMETER Port
        The port.
    .OUTPUTS
        UInt32[].
    #>
    [OutputType([uint32[]])]
    param([Parameter(Mandatory)][int]$Port)
    return [uint32[]]@(Get-NetTCPConnection -State Listen | Where-Object { $_.LocalPort -eq $Port } |
        ForEach-Object { $_.OwningProcess } | Select-Object -Unique)
}

function Wait-RwListener {
    <#
    .SYNOPSIS
        Wait until something listens on a port, or refuse by name.
    .PARAMETER Port
        The port.
    .PARAMETER Seconds
        How long to wait.
    .PARAMETER Code
        The error code a port that never opens is named with.
    .PARAMETER Reason
        What a port that never opens means, for the message.
    #>
    param([Parameter(Mandatory)][int]$Port, [Parameter(Mandatory)][int]$Seconds,
        [Parameter(Mandatory)][string]$Code, [Parameter(Mandatory)][string]$Reason)
    $deadline = [DateTime]::UtcNow.AddSeconds($Seconds)
    while (@(Get-RwListener $Port).Count -eq 0) {
        if ([DateTime]::UtcNow -ge $deadline) {
            throw "${Code}: nothing listened on port $Port within ${Seconds}s: $Reason"
        }
        Start-Sleep -Milliseconds 250
    }
}

function Get-RwGameArgument {
    <#
    .SYNOPSIS
        The JVM and game arguments a watched or hosted match starts with.
    .PARAMETER AgentJar
        The agent jar's absolute path.
    .PARAMETER AgentArgs
        The agent's argument string.
    .PARAMETER GameLog
        The game's log, absolute.
    .PARAMETER Display
        Whether the game renders a window.
    .PARAMETER Width
        The window's width.
    .PARAMETER Height
        The window's height.
    .OUTPUTS
        String[].
    #>
    [OutputType([string[]])]
    param([Parameter(Mandatory)][string]$AgentJar, [Parameter(Mandatory)][string]$AgentArgs, [Parameter(Mandatory)][string]$GameLog,
        [Parameter(Mandatory)][bool]$Display, [Parameter(Mandatory)][int]$Width, [Parameter(Mandatory)][int]$Height)
    $screen = @('-nosound')
    if (-not $Display) {
        $screen = @('-nodisplay', '-nosound')
    }
    return [string[]](@('-Xmx1000M', '--add-opens', 'java.base/java.lang=ALL-UNNAMED', '--add-opens', 'java.base/java.util=ALL-UNNAMED',
            '-Djava.library.path=.', "-javaagent:$AgentJar=$AgentArgs", '-cp', 'game-lib.jar;libs/*', 'com.corrodinggames.rts.java.Main') +
        $screen + @('-width', "$Width", '-height', "$Height", '-log', $GameLog))
}

function Invoke-RwMatch {
    <#
    .SYNOPSIS
        Start the game with the agent attached, wait for the agent's channel,
        play the match with the planner, and stop the game whatever happened.
    .PARAMETER Java
        The game's java.exe.
    .PARAMETER GameArguments
        Its arguments (Get-RwGameArgument).
    .PARAMETER GameDir
        The game's directory, absolute; the game runs there.
    .PARAMETER PlayLog
        The game's log, absolute; its streams go beside it.
    .PARAMETER Port
        The agent's channel port.
    .PARAMETER WaitSeconds
        How long the channel may take to open.
    .PARAMETER WaitReason
        What a channel that never opens means.
    .PARAMETER Poetry
        poetry.
    .PARAMETER PlannerArguments
        What follows `poetry run python -m`.
    #>
    param(
        [Parameter(Mandatory)][string]$Java,
        [Parameter(Mandatory)][string[]]$GameArguments,
        [Parameter(Mandatory)][string]$GameDir,
        [Parameter(Mandatory)][string]$PlayLog,
        [Parameter(Mandatory)][int]$Port,
        [Parameter(Mandatory)][int]$WaitSeconds,
        [Parameter(Mandatory)][string]$WaitReason,
        [Parameter(Mandatory)][string]$Poetry,
        [Parameter(Mandatory)][string[]]$PlannerArguments
    )
    $game = Start-Process -FilePath $Java -ArgumentList $GameArguments -WorkingDirectory $GameDir -PassThru `
        -RedirectStandardOutput "$PlayLog.agent" -RedirectStandardError "$PlayLog.err"
    try {
        Wait-RwListener $Port $WaitSeconds 'RW_CHANNEL_NOT_OPEN' $WaitReason
        & $Poetry (@('run', 'python', '-m') + $PlannerArguments)
        $played = $LASTEXITCODE
        if ($played -ne 0) {
            throw "RW_PLANNER_FAILED: the planner exited $played"
        }
    } finally {
        # Waited for, so the JVM has let go of the agent jar before the
        # caller removes it.
        if (-not $game.HasExited) {
            Stop-Process -Id $game.Id -Force
        }
        $game.WaitForExit()
    }
}

function Get-RwDatabaseDsn {
    <#
    .SYNOPSIS
        The covenant database's DSN, with the password read from the
        platform-postgres container at launch and never written to disk.
    .PARAMETER Docker
        docker.
    .OUTPUTS
        String.
    #>
    [OutputType([string])]
    param([Parameter(Mandatory)][string]$Docker)
    $password = (@(Invoke-RwNative $Docker @('exec', 'platform-postgres', 'printenv', 'POSTGRES_PASSWORD') 'RW_DATABASE_PASSWORD_UNREADABLE') -join '').Trim()
    if ($password -eq '') {
        throw 'RW_DATABASE_PASSWORD_UNREADABLE: platform-postgres has an empty POSTGRES_PASSWORD'
    }
    return "host=127.0.0.1 port=55432 user=covenant password=$password dbname=covenant connect_timeout=10"
}

function Invoke-RwDetachedSpawn {
    <#
    .SYNOPSIS
        Start a command line through WMI, detached from this console.
    .DESCRIPTION
        WMI Create rather than Start-Process: a Start-Process child inherits
        the launching console's handles, so `make door` never got end of
        input and hung until the door died. WMI spawns with fresh handles.
    .PARAMETER CommandLine
        The command line.
    .PARAMETER Code
        The error code a refused spawn is named with.
    .OUTPUTS
        UInt32: the spawned process's id.
    #>
    [OutputType([uint32])]
    param([Parameter(Mandatory)][string]$CommandLine, [Parameter(Mandatory)][string]$Code)
    $spawn = Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{ CommandLine = $CommandLine }
    if ($spawn.ReturnValue -ne 0) {
        throw "${Code}: WMI Create returned $($spawn.ReturnValue)"
    }
    return [uint32]$spawn.ProcessId
}

function Get-RwWorkerName {
    <#
    .SYNOPSIS
        The match workers running now, by the name on their command line.
    .PARAMETER ProcessName
        The workers' image name.
    .PARAMETER NamePrefix
        What each worker's name starts with before its number.
    .OUTPUTS
        String[]: each name once, sorted.
    #>
    [OutputType([string[]])]
    param([Parameter(Mandatory)][string]$ProcessName, [Parameter(Mandatory)][string]$NamePrefix)
    $pattern = 'match_worker .* (' + [regex]::Escape($NamePrefix) + '\d+) '
    $names = foreach ($process in @(Get-CimInstance -ClassName Win32_Process -Filter "Name='$ProcessName'")) {
        if ($null -ne $process.CommandLine -and $process.CommandLine -match $pattern) { $Matches[1] }
    }
    return [string[]]@($names | Sort-Object -Unique)
}
