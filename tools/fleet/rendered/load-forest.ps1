Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
foreach ($Process in @(Get-CimInstance -ClassName Win32_Process)) {
    '{0} {1} {2}' -f $Process.ProcessId, $Process.ParentProcessId, $Process.CommandLine
}
