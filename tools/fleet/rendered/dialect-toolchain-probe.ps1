param(
    [string]$Cmd = "$env:SystemRoot\System32\cmd.exe",
    [string]$VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe",
    [scriptblock]$Administrator = { Test-Administrator }
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
function Test-Administrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}
function Find-Tool {
    param([string]$Name)
    foreach ($directory in @($env:PATH -split ';' | Where-Object { $_ -ne '' })) {
        foreach ($extension in @($env:PATHEXT -split ';' | Where-Object { $_ -ne '' })) {
            $candidate = [System.IO.Path]::Combine($directory.Trim('"'), "$Name$extension")
            if ([System.IO.File]::Exists($candidate)) {
                return $candidate
            }
        }
    }
    return ''
}
function Invoke-Answer {
    param([string]$Shell, [string]$Path, [string]$Arguments)
    $lines = & $Shell /d /s /c "`"$Path`" $Arguments 2>&1"
    $succeeded = $LASTEXITCODE -eq 0
    $first = [string](@(@($lines) + '') | Select-Object -First 1)
    return [pscustomobject]@{ Succeeded = $succeeded; First = $first.Trim() }
}
$python = Find-Tool 'python'
if ($python -like '*\Microsoft\WindowsApps\*') {
    $python = ''
}
$tools = @(
    'python', 'poetry', 'git', 'make', 'node', 'ffmpeg', 'tar', 'cargo', 'winget', 'choco'
)
foreach ($tool in $tools) {
    $found = $python
    if ($tool -ne 'python') {
        $found = Find-Tool $tool
    }
    if ($found -ne '') {
        "$tool=yes=" + (Invoke-Answer $Cmd $found '--version').First
    } else {
        "$tool=no="
    }
}
$pip = 'pip=no='
if ($python -ne '') {
    $answer = Invoke-Answer $Cmd $python '-m pip --version'
    if ($answer.Succeeded) {
        $pip = "pip=yes=$($answer.First)"
    }
}
$pip
$query = '-products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 ' +
    '-property installationVersion'
$vc = ''
if ([System.IO.File]::Exists($VsWhere)) {
    $vc = (Invoke-Answer $Cmd $VsWhere $query).First
}
if ($vc -ne '') {
    "cxx=yes=$vc"
} else {
    'cxx=no='
}
'docker=no='
if ([bool](& $Administrator)) {
    'integrity=yes=administrator'
} else {
    'integrity=no=limited'
}
