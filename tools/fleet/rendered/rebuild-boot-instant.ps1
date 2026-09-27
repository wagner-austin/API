Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToUniversalTime().ToString('o')
