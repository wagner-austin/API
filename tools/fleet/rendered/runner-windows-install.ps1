param(
    [string]$RunnerZipUrl = 'https://github.com/actions/runner/releases/download/v2.337.0/actions-runner-win-x64-2.337.0.zip',
    [string]$Sc = "$env:SystemRoot\System32\sc.exe",
    [hashtable]$Tokens = @{},
    [hashtable[]]$Installs = @(
        @{ Repo = 'wagner-austin/chat'; Name = 'loki'; Labels = 'loki'; Root = 'C:/actions-runner/chat'; Service = 'actions.runner.wagner-austin-chat.loki' }
    )
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
foreach ($install in $Installs) {
    $root = $install.Root
    if (-not (Test-Path -LiteralPath "$root/config.cmd")) {
        [void][IO.Directory]::CreateDirectory($root)
        Invoke-WebRequest -Uri $RunnerZipUrl -OutFile "$root/runner.zip" -UseBasicParsing
        Expand-Archive -LiteralPath "$root/runner.zip" -DestinationPath $root -Force
        Remove-Item -LiteralPath "$root/runner.zip"
        "runner: unpacked $RunnerZipUrl into $root"
    }
    if (-not (Test-Path -LiteralPath "$root/.runner")) {
        if (-not $Tokens.ContainsKey($install.Repo)) {
            throw "RUNNER_TOKEN_MISSING: no registration token for $($install.Repo); fleet-runners --rebuild yes mints one"
        }
        & "$root/config.cmd" --unattended --url "https://github.com/$($install.Repo)" --token $Tokens[$install.Repo] --name $install.Name --labels $install.Labels --replace --runasservice --windowslogonaccount 'NT AUTHORITY\SYSTEM'
        if ($LASTEXITCODE -ne 0) {
            throw "RUNNER_CONFIG_FAILED: config.cmd for $($install.Repo) exited $LASTEXITCODE"
        }
        "runner: registered $($install.Repo) runner $($install.Name)"
    }
    $null = & $Sc failure $install.Service reset= 86400 actions= restart/60000/restart/60000/restart/60000
    if ($LASTEXITCODE -ne 0) {
        throw "RUNNER_SERVICE_FAILED: sc.exe failure $($install.Service) exited $LASTEXITCODE"
    }
    $state = (& $Sc query $install.Service) -join ' '
    if ($LASTEXITCODE -ne 0) {
        throw "RUNNER_SERVICE_FAILED: sc.exe query $($install.Service) exited $LASTEXITCODE"
    }
    if ($state -notmatch 'STATE\s+:\s+4\s+RUNNING') {
        $null = & $Sc start $install.Service
        if ($LASTEXITCODE -ne 0) {
            throw "RUNNER_SERVICE_FAILED: sc.exe start $($install.Service) exited $LASTEXITCODE"
        }
    }
    "runner: $($install.Repo) runner $($install.Name) runs as $($install.Service), restarted on failure"
}
