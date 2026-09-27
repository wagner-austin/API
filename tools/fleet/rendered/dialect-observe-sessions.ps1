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
$document = [pscustomobject]@{
    platform = 'win32'
    hostname = $env:COMPUTERNAME.ToLowerInvariant()
    records  = @($records)
}
$document | ConvertTo-Json -Depth 8 -Compress
