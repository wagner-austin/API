param(
    [string]$Target = 'C:/fleet/stage/MCPs-packages-maketools-1790000000'
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$result = "$Target/result.txt"
if (Test-Path -LiteralPath $result) {
    $file = Get-Item -LiteralPath $result
    $code = (Get-Content -Raw -LiteralPath $result).Trim()
    $epoch = [int]($file.LastWriteTimeUtc - [datetime]'1970-01-01').TotalSeconds
    "$code $epoch"
}
