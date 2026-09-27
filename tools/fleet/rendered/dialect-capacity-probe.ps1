Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$os = Get-CimInstance Win32_OperatingSystem
$drive = Get-PSDrive C
"free_ram_gb={0:N3}" -f ($os.FreePhysicalMemory / 1MB)
"free_disk_gb={0:N3}" -f ($drive.Free / 1GB)
"logical_cores={0}" -f (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
