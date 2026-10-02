param(
    [string]$Distro = 'Ubuntu',
    [string]$PolicyKey = 'HKLM:\SOFTWARE\Microsoft\PowerShell\1\ShellIds\Microsoft.PowerShell',
    [string]$FileSystemKey = 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem',
    [string]$EnvironmentKey = 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Environment',
    [scriptblock]$GetService = { param([string]$Name) @(Get-CimInstance Win32_Service -Filter "Name='$Name'") },
    [scriptblock]$TestWorkdir = { param([string]$Path) Test-Path -LiteralPath $Path },
    [scriptblock]$GetProcesses = { @(Get-CimInstance Win32_Process) },
    [string]$Git = 'git',
    [string]$Schtasks = "$env:SystemRoot\System32\schtasks.exe",
    [string]$Wsl = "$env:SystemRoot\System32\wsl.exe",
    [string]$Cmd = "$env:SystemRoot\System32\cmd.exe"
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$env:WSL_UTF8 = '1'
function Write-Check {
    param([string]$CheckId, [bool]$Ok, [string]$Detail)
    if ($Ok) {
        Write-Output ('CHECK ' + $CheckId + ' OK')
    } else {
        $flat = $Detail -replace "[`r`n]+", ' '
        Write-Output ('CHECK ' + $CheckId + ' DRIFT ' + $flat)
    }
}
function Invoke-Probe {
    param([string]$Shell, [string]$Line)
    $said = & $Shell /d /s /c "$Line 2>&1"
    $exit = $LASTEXITCODE
    $all = [string[]]@($said | ForEach-Object { [string]$_ })
    return [pscustomobject]@{ Exit = $exit; Lines = $all; Text = (($all -join ' ') + ' (exit ' + $exit + ')') }
}
function Invoke-InDistro {
    param([string]$Shell, [string]$WslPath, [string]$Name, [string]$Command)
    return Invoke-Probe $Shell "`"$WslPath`" -d $Name -- $Command"
}
function Get-RunnerLeftover {
    param([object[]]$Processes, [int]$ServicePid, [string]$Root, [int]$OlderThanSeconds)
    $children = @{}
    foreach ($Process in $Processes) {
        $Parent = [int]$Process.ParentProcessId
        if (-not $children.ContainsKey($Parent)) {
            $children[$Parent] = [System.Collections.Generic.List[object]]::new()
        }
        $children[$Parent].Add($Process)
    }
    $tree = @{}
    $pending = [System.Collections.Generic.Queue[int]]::new()
    if ($ServicePid -ne 0) {
        $pending.Enqueue($ServicePid)
    }
    while ($pending.Count -gt 0) {
        $Id = $pending.Dequeue()
        $tree[$Id] = $true
        if ($children.ContainsKey($Id)) {
            foreach ($Child in $children[$Id]) {
                if (-not $tree.ContainsKey([int]$Child.ProcessId)) {
                    $pending.Enqueue([int]$Child.ProcessId)
                }
            }
        }
    }
    $working = @($Processes | Where-Object { $tree.ContainsKey([int]$_.ProcessId) -and [string]$_.Name -eq 'Runner.Worker.exe' })
    if ($working.Count -gt 0) {
        return @()
    }
    $cutoff = (Get-Date).AddSeconds(-$OlderThanSeconds)
    return @($Processes | Where-Object {
        -not $tree.ContainsKey([int]$_.ProcessId) -and $null -ne $_.CreationDate -and
        $_.CreationDate -lt $cutoff -and (
            ([string]$_.CommandLine).Replace('/', '\').IndexOf($Root, [StringComparison]::OrdinalIgnoreCase) -ge 0 -or
            ([string]$_.ExecutablePath).StartsWith($Root, [StringComparison]::OrdinalIgnoreCase))
    })
}
$Probe = Invoke-Probe $Cmd "`"$Schtasks`" /query /tn wsl-keepalive /fo csv"
$KeepaliveRow = (@($Probe.Lines | Select-Object -Skip 1 -First 1) -join '')
Write-Check 'keepalive:wsl-keepalive' ($KeepaliveRow -match '"Running"') ('schtasks row: ' + $Probe.Text)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro 'free -m'
$MemLine = (@($Probe.Lines | Where-Object { $_ -match '^Mem:' }) -join '')
$TotalMb = 0
if ($MemLine -match 'Mem:\s+(\d+)') {
    $TotalMb = [int]$Matches[1]
}
Write-Check 'memory-floor:26gb' ($TotalMb -ge 26000) ('the VM reports ' + $TotalMb + ' MB; free said: ' + $Probe.Text)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro 'df -BG --output=used /'
$DiskLine = (@($Probe.Lines | Select-Object -Skip 1 -First 1) -join '')
$UsedGb = -1
if ($DiskLine -match '(\d+)G') {
    $UsedGb = [int]$Matches[1]
}
Write-Check 'disk:/:ceiling-150gb:baseline-46gb@2026-09-26' ($UsedGb -ge 0 -and $UsedGb -le 150) ('the distro root uses ' + $UsedGb + ' GB against a ceiling of 150 GB; an idle rebuilt host used 46 GB on 2026-09-26; df said: ' + $Probe.Text)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "du -s -BG '/home/gharunner/.cache'"
$CacheGb = -1
if ((@($Probe.Lines | Select-Object -First 1) -join '') -match '^(\d+)G') {
    $CacheGb = [int]$Matches[1]
}
Write-Check 'cache:/home/gharunner/.cache:ceiling-60gb' ($Probe.Exit -eq 0 -and $CacheGb -ge 0 -and $CacheGb -le 60) ('/home/gharunner/.cache holds ' + $CacheGb + ' GB against a ceiling of 60 GB; du said: ' + $Probe.Text)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "du -s -BG '/home/gharunner/actions-runner-api-1/_work'"
$CacheGb = -1
if ((@($Probe.Lines | Select-Object -First 1) -join '') -match '^(\d+)G') {
    $CacheGb = [int]$Matches[1]
}
Write-Check 'cache:/home/gharunner/actions-runner-api-1/_work:ceiling-15gb' ($Probe.Exit -eq 0 -and $CacheGb -ge 0 -and $CacheGb -le 15) ('/home/gharunner/actions-runner-api-1/_work holds ' + $CacheGb + ' GB against a ceiling of 15 GB; du said: ' + $Probe.Text)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "du -s -BG '/home/gharunner/actions-runner-api-2/_work'"
$CacheGb = -1
if ((@($Probe.Lines | Select-Object -First 1) -join '') -match '^(\d+)G') {
    $CacheGb = [int]$Matches[1]
}
Write-Check 'cache:/home/gharunner/actions-runner-api-2/_work:ceiling-15gb' ($Probe.Exit -eq 0 -and $CacheGb -ge 0 -and $CacheGb -le 15) ('/home/gharunner/actions-runner-api-2/_work holds ' + $CacheGb + ' GB against a ceiling of 15 GB; du said: ' + $Probe.Text)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "du -s -BG '/home/gharunner/actions-runner/_work'"
$CacheGb = -1
if ((@($Probe.Lines | Select-Object -First 1) -join '') -match '^(\d+)G') {
    $CacheGb = [int]$Matches[1]
}
Write-Check 'cache:/home/gharunner/actions-runner/_work:ceiling-15gb' ($Probe.Exit -eq 0 -and $CacheGb -ge 0 -and $CacheGb -le 15) ('/home/gharunner/actions-runner/_work holds ' + $CacheGb + ' GB against a ceiling of 15 GB; du said: ' + $Probe.Text)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "du -s -BG '/home/gharunner/actions-runner-2/_work'"
$CacheGb = -1
if ((@($Probe.Lines | Select-Object -First 1) -join '') -match '^(\d+)G') {
    $CacheGb = [int]$Matches[1]
}
Write-Check 'cache:/home/gharunner/actions-runner-2/_work:ceiling-15gb' ($Probe.Exit -eq 0 -and $CacheGb -ge 0 -and $CacheGb -le 15) ('/home/gharunner/actions-runner-2/_work holds ' + $CacheGb + ' GB against a ceiling of 15 GB; du said: ' + $Probe.Text)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "du -s -BG '/home/gharunner/actions-runner-3/_work'"
$CacheGb = -1
if ((@($Probe.Lines | Select-Object -First 1) -join '') -match '^(\d+)G') {
    $CacheGb = [int]$Matches[1]
}
Write-Check 'cache:/home/gharunner/actions-runner-3/_work:ceiling-15gb' ($Probe.Exit -eq 0 -and $CacheGb -ge 0 -and $CacheGb -le 15) ('/home/gharunner/actions-runner-3/_work holds ' + $CacheGb + ' GB against a ceiling of 15 GB; du said: ' + $Probe.Text)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "du -s -BG '/home/gharunner/actions-runner-4/_work'"
$CacheGb = -1
if ((@($Probe.Lines | Select-Object -First 1) -join '') -match '^(\d+)G') {
    $CacheGb = [int]$Matches[1]
}
Write-Check 'cache:/home/gharunner/actions-runner-4/_work:ceiling-15gb' ($Probe.Exit -eq 0 -and $CacheGb -ge 0 -and $CacheGb -le 15) ('/home/gharunner/actions-runner-4/_work holds ' + $CacheGb + ' GB against a ceiling of 15 GB; du said: ' + $Probe.Text)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "du -s -BG '/home/gharunner/actions-runner-corvis-stick-1/_work'"
$CacheGb = -1
if ((@($Probe.Lines | Select-Object -First 1) -join '') -match '^(\d+)G') {
    $CacheGb = [int]$Matches[1]
}
Write-Check 'cache:/home/gharunner/actions-runner-corvis-stick-1/_work:ceiling-15gb' ($Probe.Exit -eq 0 -and $CacheGb -ge 0 -and $CacheGb -le 15) ('/home/gharunner/actions-runner-corvis-stick-1/_work holds ' + $CacheGb + ' GB against a ceiling of 15 GB; du said: ' + $Probe.Text)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "du -s -BG '/home/gharunner/actions-runner-treebot-1/_work'"
$CacheGb = -1
if ((@($Probe.Lines | Select-Object -First 1) -join '') -match '^(\d+)G') {
    $CacheGb = [int]$Matches[1]
}
Write-Check 'cache:/home/gharunner/actions-runner-treebot-1/_work:ceiling-15gb' ($Probe.Exit -eq 0 -and $CacheGb -ge 0 -and $CacheGb -le 15) ('/home/gharunner/actions-runner-treebot-1/_work holds ' + $CacheGb + ' GB against a ceiling of 15 GB; du said: ' + $Probe.Text)
$Policy = [string](Get-Item -LiteralPath $PolicyKey).GetValue('ExecutionPolicy')
Write-Check 'execution-policy:LocalMachine:RemoteSigned' ($Policy -eq 'RemoteSigned') ('the LocalMachine ExecutionPolicy value is: ' + $Policy)
$LongPaths = [string](Get-Item -LiteralPath $FileSystemKey).GetValue('LongPathsEnabled')
$Probe = Invoke-Probe $Cmd "`"$Git`" config --system --get core.longpaths"
$GitLongPaths = (@($Probe.Lines) -join '')
Write-Check 'long-paths:win32-and-git' ($LongPaths -eq '1' -and $GitLongPaths -eq 'true') ('LongPathsEnabled=' + $LongPaths + ' git core.longpaths=' + $Probe.Text)
$Held = [string](Get-Item -LiteralPath $EnvironmentKey).GetValue('POETRY_CACHE_DIR')
Write-Check 'machine-env:POETRY_CACHE_DIR' ($Held -ceq 'C:\fleet\poetry') ('the machine environment holds POETRY_CACHE_DIR=' + $Held + '; the roster says C:\fleet\poetry')
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "sh -c 'PATH=`$(cat /home/gharunner/actions-runner-api-1/.path) nvidia-smi --query-gpu=name --format=csv,noheader'"
$GpuName = (@($Probe.Lines | Select-Object -First 1) -join '')
Write-Check 'gpu:wagner-austin/API:wsl:lavender-wsl' ($Probe.Exit -eq 0 -and $GpuName.Trim().Length -gt 0) ('nvidia-smi on the runner PATH said: ' + $Probe.Text)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "sh -c 'PATH=`$(cat /home/gharunner/actions-runner-api-2/.path) nvidia-smi --query-gpu=name --format=csv,noheader'"
$GpuName = (@($Probe.Lines | Select-Object -First 1) -join '')
Write-Check 'gpu:wagner-austin/API:wsl:lavender-wsl-2' ($Probe.Exit -eq 0 -and $GpuName.Trim().Length -gt 0) ('nvidia-smi on the runner PATH said: ' + $Probe.Text)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "sh -c 'PATH=`$(cat /home/gharunner/actions-runner/.path) nvidia-smi --query-gpu=name --format=csv,noheader'"
$GpuName = (@($Probe.Lines | Select-Object -First 1) -join '')
Write-Check 'gpu:wagner-austin/MCPs:wsl:lavender-wsl' ($Probe.Exit -eq 0 -and $GpuName.Trim().Length -gt 0) ('nvidia-smi on the runner PATH said: ' + $Probe.Text)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "sh -c 'PATH=`$(cat /home/gharunner/actions-runner-2/.path) nvidia-smi --query-gpu=name --format=csv,noheader'"
$GpuName = (@($Probe.Lines | Select-Object -First 1) -join '')
Write-Check 'gpu:wagner-austin/MCPs:wsl:lavender-wsl-2' ($Probe.Exit -eq 0 -and $GpuName.Trim().Length -gt 0) ('nvidia-smi on the runner PATH said: ' + $Probe.Text)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "sh -c 'PATH=`$(cat /home/gharunner/actions-runner-3/.path) nvidia-smi --query-gpu=name --format=csv,noheader'"
$GpuName = (@($Probe.Lines | Select-Object -First 1) -join '')
Write-Check 'gpu:wagner-austin/MCPs:wsl:lavender-wsl-3' ($Probe.Exit -eq 0 -and $GpuName.Trim().Length -gt 0) ('nvidia-smi on the runner PATH said: ' + $Probe.Text)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "sh -c 'PATH=`$(cat /home/gharunner/actions-runner-4/.path) nvidia-smi --query-gpu=name --format=csv,noheader'"
$GpuName = (@($Probe.Lines | Select-Object -First 1) -join '')
Write-Check 'gpu:wagner-austin/MCPs:wsl:lavender-wsl-4' ($Probe.Exit -eq 0 -and $GpuName.Trim().Length -gt 0) ('nvidia-smi on the runner PATH said: ' + $Probe.Text)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "sh -c 'PATH=`$(cat /home/gharunner/actions-runner-corvis-stick-1/.path) nvidia-smi --query-gpu=name --format=csv,noheader'"
$GpuName = (@($Probe.Lines | Select-Object -First 1) -join '')
Write-Check 'gpu:wagner-austin/corvis-stick:wsl:lavender-wsl' ($Probe.Exit -eq 0 -and $GpuName.Trim().Length -gt 0) ('nvidia-smi on the runner PATH said: ' + $Probe.Text)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "sh -c 'PATH=`$(cat /home/gharunner/actions-runner-treebot-1/.path) nvidia-smi --query-gpu=name --format=csv,noheader'"
$GpuName = (@($Probe.Lines | Select-Object -First 1) -join '')
Write-Check 'gpu:wagner-austin/tree-bot:wsl:lavender-wsl' ($Probe.Exit -eq 0 -and $GpuName.Trim().Length -gt 0) ('nvidia-smi on the runner PATH said: ' + $Probe.Text)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "systemctl is-enabled 'ci-clean.timer'"
$State = (@($Probe.Lines | Select-Object -First 1) -join '')
Write-Check 'timer:ci-clean.timer' ($State -eq 'enabled') ('it said: ' + $Probe.Text)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "systemctl is-enabled 'fleet-runner-reaper.timer'"
$State = (@($Probe.Lines | Select-Object -First 1) -join '')
Write-Check 'timer:fleet-runner-reaper.timer' ($State -eq 'enabled') ('it said: ' + $Probe.Text)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "systemctl is-active 'actions.runner.wagner-austin-API.lavender-wsl.service'"
$State = (@($Probe.Lines | Select-Object -First 1) -join '')
Write-Check 'service:wsl:actions.runner.wagner-austin-API.lavender-wsl.service' ($State -eq 'active') ('it said: ' + $Probe.Text)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "test -d '/home/gharunner/actions-runner-api-1/_work'"
Write-Check 'workdir:wagner-austin/API:wsl:lavender-wsl' ($Probe.Exit -eq 0) ('test -d exited ' + $Probe.Exit)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "/usr/local/sbin/fleet-runner-reaper --audit 21600 'actions.runner.wagner-austin-API.lavender-wsl.service'"
$Leftover = (@($Probe.Lines | Select-Object -First 1) -join '')
Write-Check 'orphans:wagner-austin/API:wsl:lavender-wsl' ($Probe.Exit -eq 0 -and $Leftover -eq '0') ('processes older than 360 minutes outside actions.runner.wagner-austin-API.lavender-wsl.service with no job running; the reaper counted: ' + $Probe.Text)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "systemctl is-active 'actions.runner.wagner-austin-API.lavender-wsl-2.service'"
$State = (@($Probe.Lines | Select-Object -First 1) -join '')
Write-Check 'service:wsl:actions.runner.wagner-austin-API.lavender-wsl-2.service' ($State -eq 'active') ('it said: ' + $Probe.Text)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "test -d '/home/gharunner/actions-runner-api-2/_work'"
Write-Check 'workdir:wagner-austin/API:wsl:lavender-wsl-2' ($Probe.Exit -eq 0) ('test -d exited ' + $Probe.Exit)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "/usr/local/sbin/fleet-runner-reaper --audit 21600 'actions.runner.wagner-austin-API.lavender-wsl-2.service'"
$Leftover = (@($Probe.Lines | Select-Object -First 1) -join '')
Write-Check 'orphans:wagner-austin/API:wsl:lavender-wsl-2' ($Probe.Exit -eq 0 -and $Leftover -eq '0') ('processes older than 360 minutes outside actions.runner.wagner-austin-API.lavender-wsl-2.service with no job running; the reaper counted: ' + $Probe.Text)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "systemctl is-active 'actions.runner.wagner-austin-MCPs.lavender-wsl.service'"
$State = (@($Probe.Lines | Select-Object -First 1) -join '')
Write-Check 'service:wsl:actions.runner.wagner-austin-MCPs.lavender-wsl.service' ($State -eq 'active') ('it said: ' + $Probe.Text)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "test -d '/home/gharunner/actions-runner/_work'"
Write-Check 'workdir:wagner-austin/MCPs:wsl:lavender-wsl' ($Probe.Exit -eq 0) ('test -d exited ' + $Probe.Exit)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "/usr/local/sbin/fleet-runner-reaper --audit 21600 'actions.runner.wagner-austin-MCPs.lavender-wsl.service'"
$Leftover = (@($Probe.Lines | Select-Object -First 1) -join '')
Write-Check 'orphans:wagner-austin/MCPs:wsl:lavender-wsl' ($Probe.Exit -eq 0 -and $Leftover -eq '0') ('processes older than 360 minutes outside actions.runner.wagner-austin-MCPs.lavender-wsl.service with no job running; the reaper counted: ' + $Probe.Text)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "systemctl is-active 'actions.runner.wagner-austin-MCPs.lavender-wsl-2.service'"
$State = (@($Probe.Lines | Select-Object -First 1) -join '')
Write-Check 'service:wsl:actions.runner.wagner-austin-MCPs.lavender-wsl-2.service' ($State -eq 'active') ('it said: ' + $Probe.Text)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "test -d '/home/gharunner/actions-runner-2/_work'"
Write-Check 'workdir:wagner-austin/MCPs:wsl:lavender-wsl-2' ($Probe.Exit -eq 0) ('test -d exited ' + $Probe.Exit)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "/usr/local/sbin/fleet-runner-reaper --audit 21600 'actions.runner.wagner-austin-MCPs.lavender-wsl-2.service'"
$Leftover = (@($Probe.Lines | Select-Object -First 1) -join '')
Write-Check 'orphans:wagner-austin/MCPs:wsl:lavender-wsl-2' ($Probe.Exit -eq 0 -and $Leftover -eq '0') ('processes older than 360 minutes outside actions.runner.wagner-austin-MCPs.lavender-wsl-2.service with no job running; the reaper counted: ' + $Probe.Text)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "systemctl is-active 'actions.runner.wagner-austin-MCPs.lavender-wsl-3.service'"
$State = (@($Probe.Lines | Select-Object -First 1) -join '')
Write-Check 'service:wsl:actions.runner.wagner-austin-MCPs.lavender-wsl-3.service' ($State -eq 'active') ('it said: ' + $Probe.Text)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "test -d '/home/gharunner/actions-runner-3/_work'"
Write-Check 'workdir:wagner-austin/MCPs:wsl:lavender-wsl-3' ($Probe.Exit -eq 0) ('test -d exited ' + $Probe.Exit)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "/usr/local/sbin/fleet-runner-reaper --audit 21600 'actions.runner.wagner-austin-MCPs.lavender-wsl-3.service'"
$Leftover = (@($Probe.Lines | Select-Object -First 1) -join '')
Write-Check 'orphans:wagner-austin/MCPs:wsl:lavender-wsl-3' ($Probe.Exit -eq 0 -and $Leftover -eq '0') ('processes older than 360 minutes outside actions.runner.wagner-austin-MCPs.lavender-wsl-3.service with no job running; the reaper counted: ' + $Probe.Text)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "systemctl is-active 'actions.runner.wagner-austin-MCPs.lavender-wsl-4.service'"
$State = (@($Probe.Lines | Select-Object -First 1) -join '')
Write-Check 'service:wsl:actions.runner.wagner-austin-MCPs.lavender-wsl-4.service' ($State -eq 'active') ('it said: ' + $Probe.Text)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "test -d '/home/gharunner/actions-runner-4/_work'"
Write-Check 'workdir:wagner-austin/MCPs:wsl:lavender-wsl-4' ($Probe.Exit -eq 0) ('test -d exited ' + $Probe.Exit)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "/usr/local/sbin/fleet-runner-reaper --audit 21600 'actions.runner.wagner-austin-MCPs.lavender-wsl-4.service'"
$Leftover = (@($Probe.Lines | Select-Object -First 1) -join '')
Write-Check 'orphans:wagner-austin/MCPs:wsl:lavender-wsl-4' ($Probe.Exit -eq 0 -and $Leftover -eq '0') ('processes older than 360 minutes outside actions.runner.wagner-austin-MCPs.lavender-wsl-4.service with no job running; the reaper counted: ' + $Probe.Text)
$Service = @(& $GetService 'actions.runner.wagner-austin-MCPs.lavender')
$ServiceState = (@($Service | ForEach-Object { [string]$_.State }) -join '')
Write-Check 'service:windows:actions.runner.wagner-austin-MCPs.lavender' ($ServiceState -eq 'Running') ('Win32_Service State: ' + $ServiceState)
Write-Check 'workdir:wagner-austin/MCPs:windows:lavender' ([bool](& $TestWorkdir 'C:/actions-runner/_work')) ('Test-Path C:/actions-runner/_work')
$Account = (@($Service | ForEach-Object { [string]$_.StartName }) -join '')
Write-Check 'account:windows:actions.runner.wagner-austin-MCPs.lavender:LocalSystem' ($Account -eq 'LocalSystem') ('Win32_Service StartName: ' + $Account)
$ServicePid = 0
foreach ($Row in $Service) {
    $ServicePid = [int]$Row.ProcessId
}
$Leftover = @(Get-RunnerLeftover @(& $GetProcesses) $ServicePid 'C:\actions-runner\' 21600)
Write-Check 'orphans:wagner-austin/MCPs:windows:lavender' ($Leftover.Count -eq 0) ([string]$Leftover.Count + ' process(es) under C:\actions-runner\ outside actions.runner.wagner-austin-MCPs.lavender, older than 360 minutes with no job running: ' + ((@($Leftover | Select-Object -First 5 | ForEach-Object { [string]$_.Name + ' pid ' + [string]$_.ProcessId })) -join ', '))
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "systemctl is-active 'actions.runner.wagner-austin-corvis-stick.lavender-wsl.service'"
$State = (@($Probe.Lines | Select-Object -First 1) -join '')
Write-Check 'service:wsl:actions.runner.wagner-austin-corvis-stick.lavender-wsl.service' ($State -eq 'active') ('it said: ' + $Probe.Text)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "test -d '/home/gharunner/actions-runner-corvis-stick-1/_work'"
Write-Check 'workdir:wagner-austin/corvis-stick:wsl:lavender-wsl' ($Probe.Exit -eq 0) ('test -d exited ' + $Probe.Exit)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "/usr/local/sbin/fleet-runner-reaper --audit 21600 'actions.runner.wagner-austin-corvis-stick.lavender-wsl.service'"
$Leftover = (@($Probe.Lines | Select-Object -First 1) -join '')
Write-Check 'orphans:wagner-austin/corvis-stick:wsl:lavender-wsl' ($Probe.Exit -eq 0 -and $Leftover -eq '0') ('processes older than 360 minutes outside actions.runner.wagner-austin-corvis-stick.lavender-wsl.service with no job running; the reaper counted: ' + $Probe.Text)
$Service = @(& $GetService 'actions.runner.wagner-austin-corvis-stick.lavender')
$ServiceState = (@($Service | ForEach-Object { [string]$_.State }) -join '')
Write-Check 'service:windows:actions.runner.wagner-austin-corvis-stick.lavender' ($ServiceState -eq 'Running') ('Win32_Service State: ' + $ServiceState)
Write-Check 'workdir:wagner-austin/corvis-stick:windows:lavender' ([bool](& $TestWorkdir 'C:/actions-runner-corvis-stick/_work')) ('Test-Path C:/actions-runner-corvis-stick/_work')
$Account = (@($Service | ForEach-Object { [string]$_.StartName }) -join '')
Write-Check 'account:windows:actions.runner.wagner-austin-corvis-stick.lavender:LocalSystem' ($Account -eq 'LocalSystem') ('Win32_Service StartName: ' + $Account)
$ServicePid = 0
foreach ($Row in $Service) {
    $ServicePid = [int]$Row.ProcessId
}
$Leftover = @(Get-RunnerLeftover @(& $GetProcesses) $ServicePid 'C:\actions-runner-corvis-stick\' 21600)
Write-Check 'orphans:wagner-austin/corvis-stick:windows:lavender' ($Leftover.Count -eq 0) ([string]$Leftover.Count + ' process(es) under C:\actions-runner-corvis-stick\ outside actions.runner.wagner-austin-corvis-stick.lavender, older than 360 minutes with no job running: ' + ((@($Leftover | Select-Object -First 5 | ForEach-Object { [string]$_.Name + ' pid ' + [string]$_.ProcessId })) -join ', '))
$Service = @(& $GetService 'actions.runner.wagner-austin-chat.lavender')
$ServiceState = (@($Service | ForEach-Object { [string]$_.State }) -join '')
Write-Check 'service:windows:actions.runner.wagner-austin-chat.lavender' ($ServiceState -eq 'Running') ('Win32_Service State: ' + $ServiceState)
Write-Check 'workdir:wagner-austin/chat:windows:lavender' ([bool](& $TestWorkdir 'C:/actions-runner-chat/_work')) ('Test-Path C:/actions-runner-chat/_work')
$Account = (@($Service | ForEach-Object { [string]$_.StartName }) -join '')
Write-Check 'account:windows:actions.runner.wagner-austin-chat.lavender:LocalSystem' ($Account -eq 'LocalSystem') ('Win32_Service StartName: ' + $Account)
$ServicePid = 0
foreach ($Row in $Service) {
    $ServicePid = [int]$Row.ProcessId
}
$Leftover = @(Get-RunnerLeftover @(& $GetProcesses) $ServicePid 'C:\actions-runner-chat\' 21600)
Write-Check 'orphans:wagner-austin/chat:windows:lavender' ($Leftover.Count -eq 0) ([string]$Leftover.Count + ' process(es) under C:\actions-runner-chat\ outside actions.runner.wagner-austin-chat.lavender, older than 360 minutes with no job running: ' + ((@($Leftover | Select-Object -First 5 | ForEach-Object { [string]$_.Name + ' pid ' + [string]$_.ProcessId })) -join ', '))
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "systemctl is-active 'actions.runner.wagner-austin-tree-bot.lavender-wsl.service'"
$State = (@($Probe.Lines | Select-Object -First 1) -join '')
Write-Check 'service:wsl:actions.runner.wagner-austin-tree-bot.lavender-wsl.service' ($State -eq 'active') ('it said: ' + $Probe.Text)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "test -d '/home/gharunner/actions-runner-treebot-1/_work'"
Write-Check 'workdir:wagner-austin/tree-bot:wsl:lavender-wsl' ($Probe.Exit -eq 0) ('test -d exited ' + $Probe.Exit)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "/usr/local/sbin/fleet-runner-reaper --audit 21600 'actions.runner.wagner-austin-tree-bot.lavender-wsl.service'"
$Leftover = (@($Probe.Lines | Select-Object -First 1) -join '')
Write-Check 'orphans:wagner-austin/tree-bot:wsl:lavender-wsl' ($Probe.Exit -eq 0 -and $Leftover -eq '0') ('processes older than 360 minutes outside actions.runner.wagner-austin-tree-bot.lavender-wsl.service with no job running; the reaper counted: ' + $Probe.Text)
$Service = @(& $GetService 'actions.runner.wagner-austin-tree-bot.lavender')
$ServiceState = (@($Service | ForEach-Object { [string]$_.State }) -join '')
Write-Check 'service:windows:actions.runner.wagner-austin-tree-bot.lavender' ($ServiceState -eq 'Running') ('Win32_Service State: ' + $ServiceState)
Write-Check 'workdir:wagner-austin/tree-bot:windows:lavender' ([bool](& $TestWorkdir 'C:/actions-runner-tree-bot/_work')) ('Test-Path C:/actions-runner-tree-bot/_work')
$Account = (@($Service | ForEach-Object { [string]$_.StartName }) -join '')
Write-Check 'account:windows:actions.runner.wagner-austin-tree-bot.lavender:LocalSystem' ($Account -eq 'LocalSystem') ('Win32_Service StartName: ' + $Account)
$ServicePid = 0
foreach ($Row in $Service) {
    $ServicePid = [int]$Row.ProcessId
}
$Leftover = @(Get-RunnerLeftover @(& $GetProcesses) $ServicePid 'C:\actions-runner-tree-bot\' 21600)
Write-Check 'orphans:wagner-austin/tree-bot:windows:lavender' ($Leftover.Count -eq 0) ([string]$Leftover.Count + ' process(es) under C:\actions-runner-tree-bot\ outside actions.runner.wagner-austin-tree-bot.lavender, older than 360 minutes with no job running: ' + ((@($Leftover | Select-Object -First 5 | ForEach-Object { [string]$_.Name + ' pid ' + [string]$_.ProcessId })) -join ', '))
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "test -e '/opt/corvis/rw-game/game-lib.jar'"
Write-Check 'asset:/opt/corvis/rw-game/game-lib.jar' ($Probe.Exit -eq 0) ('test -e exited ' + $Probe.Exit)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "sha256sum '/opt/corvis/rw-game/game-lib.jar'"
$Sum = (@($Probe.Lines | Select-Object -First 1) -join '')
Write-Check 'sha256:/opt/corvis/rw-game/game-lib.jar' ($Sum.StartsWith('8a550a37e2d8a5430866090d4e7d5892f9010b47f52a5a09350fc66c620deec9')) ('sha256sum said: ' + $Probe.Text)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "test -e '/opt/llama.cpp/convert_lora_to_gguf.py'"
Write-Check 'asset:/opt/llama.cpp/convert_lora_to_gguf.py' ($Probe.Exit -eq 0) ('test -e exited ' + $Probe.Exit)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "test -e '/data'"
Write-Check 'asset:/data' ($Probe.Exit -eq 0) ('test -e exited ' + $Probe.Exit)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "test -w '/data'"
Write-Check 'writable:/data' ($Probe.Exit -eq 0) ('test -w exited ' + $Probe.Exit)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "test -e '/opt/google/chrome/chrome'"
Write-Check 'asset:/opt/google/chrome/chrome' ($Probe.Exit -eq 0) ('test -e exited ' + $Probe.Exit)
$Probe = Invoke-InDistro $Cmd $Wsl $Distro "test -e '/opt/corvis/model-cache/Xenova/gte-small/onnx/model.onnx'"
Write-Check 'asset:/opt/corvis/model-cache/Xenova/gte-small/onnx/model.onnx' ($Probe.Exit -eq 0) ('test -e exited ' + $Probe.Exit)
exit 0
