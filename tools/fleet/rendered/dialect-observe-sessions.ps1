param(
    [string]$SessionsDirectory = "$HOME\.claude\sessions"
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$records = @()
if (Test-Path -LiteralPath $SessionsDirectory) {
    foreach ($file in Get-ChildItem -LiteralPath $SessionsDirectory -Filter '*.json' -File) {
        $records += , (Get-Content -Raw -LiteralPath $file.FullName | ConvertFrom-Json)
    }
}
$hostname = $env:COMPUTERNAME.ToLowerInvariant()
$document = [pscustomobject]@{
    platform  = 'win32'
    hostname  = $hostname
    pidDomain = 'win32:' + $hostname
    records   = @($records)
}
$document | ConvertTo-Json -Depth 8 -Compress
