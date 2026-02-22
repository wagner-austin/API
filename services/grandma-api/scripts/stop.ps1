$ErrorActionPreference = "Continue"

Write-Host "Stopping grandma-api services..." -ForegroundColor Cyan

# Kill anything on our ports
@(8090, 8091) | ForEach-Object {
    $port = $_
    $conn = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
    if ($conn) {
        $pids = $conn.OwningProcess | Where-Object { $_ -gt 4 } | Select-Object -Unique
        foreach ($p in $pids) {
            Write-Host "  Killing PID $p on port $port" -ForegroundColor Gray
            cmd /c "taskkill /F /T /PID $p 2>nul"
        }
    }
}

# Wait for ports to clear
Start-Sleep -Seconds 2

Write-Host "Done" -ForegroundColor Green
