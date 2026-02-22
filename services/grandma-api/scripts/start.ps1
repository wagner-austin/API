$ErrorActionPreference = "Stop"

$projectRoot = "C:\Users\Test\PROJECTS\API\services\grandma-api"
$webDir = "$projectRoot\web"
$logDir = "$projectRoot\logs"

# Create logs dir
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }

# Load .env
$envFile = "$projectRoot\.env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]*?)\s*=\s*(.*)$') {
            $key = $matches[1].Trim()
            $value = $matches[2].Trim().Trim('"', "'")
            [Environment]::SetEnvironmentVariable($key, $value, 'Process')
        }
    }
}

$apiPort = if ($env:PORT) { [int]$env:PORT } else { 8090 }
$webPort = 8091

Write-Host "Starting grandma-api..." -ForegroundColor Cyan

# Check if already running
$apiRunning = Get-NetTCPConnection -LocalPort $apiPort -ErrorAction SilentlyContinue
$webRunning = Get-NetTCPConnection -LocalPort $webPort -ErrorAction SilentlyContinue

if ($apiRunning -and $webRunning) {
    Write-Host "Already running on ports $apiPort and $webPort" -ForegroundColor Green
    exit 0
}

# Start API
if (-not $apiRunning) {
    Write-Host "  Starting API on port $apiPort..." -ForegroundColor Gray
    $certFile = "$webDir\cert.pem"
    $keyFile = "$webDir\key.pem"

    Start-Process -FilePath "poetry" `
        -ArgumentList "run", "hypercorn", "grandma_api.asgi:app", "--bind", "0.0.0.0:$apiPort", "--reload", "--certfile", $certFile, "--keyfile", $keyFile `
        -WorkingDirectory $projectRoot `
        -RedirectStandardOutput "$logDir\api.log" `
        -RedirectStandardError "$logDir\api-err.log" `
        -WindowStyle Hidden

    # Wait for port
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 1
        if (Get-NetTCPConnection -LocalPort $apiPort -ErrorAction SilentlyContinue) {
            Write-Host "  API ready" -ForegroundColor Green
            break
        }
    }
}

# Build and start web
if (-not $webRunning) {
    Write-Host "  Building frontend..." -ForegroundColor Gray
    Push-Location $webDir
    npm run build 2>&1 | Out-Null
    Pop-Location

    Write-Host "  Starting web on port $webPort..." -ForegroundColor Gray
    Start-Process -FilePath "poetry" `
        -ArgumentList "run", "python", "-m", "scripts.webserver", $webPort, $webDir `
        -WorkingDirectory $projectRoot `
        -RedirectStandardOutput "$logDir\web.log" `
        -RedirectStandardError "$logDir\web-err.log" `
        -WindowStyle Hidden

    Start-Sleep -Seconds 2
    Write-Host "  Web ready" -ForegroundColor Green
}

Write-Host ""
Write-Host "Services running:" -ForegroundColor Green
Write-Host "  API: https://localhost:$apiPort"
Write-Host "  Web: https://localhost:$webPort"
Write-Host ""
Write-Host "Password: grandma" -ForegroundColor Cyan
