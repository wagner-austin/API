param(
    [string]$Log = 'C:/fleet/stage/MCPs-packages-maketools-1790000000/result.txt.log',
    [int]$Lines = 200
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if (Test-Path -LiteralPath $Log) {
    $stream = [System.IO.File]::Open($Log, 'Open', 'Read', 'ReadWrite')
    $mark = New-Object byte[] 3
    $marked = $stream.Read($mark, 0, 3)
    $encoding = [System.Text.Encoding]::UTF8
    $skip = 0
    if (($marked -ge 2) -and ($mark[0] -eq 255) -and ($mark[1] -eq 254)) {
        $encoding = [System.Text.Encoding]::Unicode
        $skip = 2
    }
    if (($marked -eq 3) -and ($mark[0] -eq 239) -and ($mark[1] -eq 187) -and ($mark[2] -eq 191)) {
        $skip = 3
    }
    $start = [Math]::Max([long]$skip, $stream.Length - 262144)
    if (($skip -eq 2) -and (($start % 2) -eq 1)) {
        $start = $start + 1
    }
    $null = $stream.Seek($start, 'Begin')
    $bytes = New-Object byte[] ($stream.Length - $start)
    $read = 0
    do {
        $got = $stream.Read($bytes, $read, $bytes.Length - $read)
        $read = $read + $got
    } while (($got -gt 0) -and ($read -lt $bytes.Length))
    $stream.Close()
    $text = $encoding.GetString($bytes, 0, $read).TrimEnd([char[]]"`r`n")
    $parts = $text -split "`r?`n"
    if ($start -gt $skip) {
        $parts = $parts | Select-Object -Skip 1
    }
    $parts | Select-Object -Last $Lines
}
