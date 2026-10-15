Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$os = Get-CimInstance Win32_OperatingSystem
$drive = Get-PSDrive C
"free_ram_gb={0:N3}" -f ($os.FreePhysicalMemory / 1MB)
"free_disk_gb={0:N3}" -f ($drive.Free / 1GB)
"logical_cores={0}" -f (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
$reservations = Join-Path $env:USERPROFILE '.fleet/reservations'
$now = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
$reserved = 0
if (Test-Path -LiteralPath $reservations) {
  foreach ($file in Get-ChildItem -LiteralPath $reservations -Filter '*.reservation') {
    $fields = @{}
    foreach ($line in Get-Content -LiteralPath $file.FullName) {
      $pair = $line -split '=', 2
      if ($pair.Count -eq 2) { $fields[$pair[0]] = $pair[1] }
    }
    if ($fields['cores'] -notmatch '^[0-9]+$' -or $fields['until_unix'] -notmatch '^[0-9]+$') {
      'reservation_unreadable={0}' -f $file.FullName
      exit 0
    }
    if ([int64]$fields['until_unix'] -gt $now) { $reserved += [int]$fields['cores'] }
  }
}
'reserved_cores={0}' -f $reserved
