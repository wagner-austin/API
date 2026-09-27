Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# The Windows transcript tail, executed from its committed render under
# rendered/ against transcripts in each encoding a node's writers produce
# (MCPs board task d69786fa, A2). The regression it carries is 704f324f:
# Get-Content -Tail on a three-megabyte UTF-16 transcript outlived the
# collector's 120-second deadline on sedona, so the script reads one bounded
# window from the end instead.

BeforeAll {
    . (Join-Path $PSScriptRoot 'rendered-fixtures.ps1')

    # Lines 'line 0000000 passed in the fixture' onward, CRLF-terminated,
    # 36 characters each before the break.
    function Get-NumberedText {
        param([int]$Count)
        $builder = [System.Text.StringBuilder]::new()
        for ($index = 0; $index -lt $Count; $index++) {
            [void]$builder.Append(('line {0:D7} passed in the fixture' -f $index)).Append("`r`n")
        }
        return $builder.ToString()
    }

    function Write-Transcript {
        param([byte[]]$Mark, [byte[]]$Body, [byte[]]$Tail = @())
        $path = Join-Path $TestDrive ('result-' + [guid]::NewGuid().ToString('N') + '.txt.log')
        [System.IO.File]::WriteAllBytes($path, [byte[]]($Mark + $Body + $Tail))
        return $path
    }

    function Read-Tail {
        param([string]$Log, [int]$Lines)
        return [string[]]@(Invoke-Rendered 'dialect-log-tail' @{ Log = $Log; Lines = $Lines })
    }

    $script:utf16Mark = [byte[]](255, 254)
    $script:utf8Mark = [byte[]](239, 187, 191)
}

Describe 'The transcript tail' {
    It 'reads a transcript the build''s own redirection wrote, without its mark' {
        $log = Join-Path $TestDrive 'redirected.txt.log'
        Write-Output 'first line' *>> $log
        Write-Output 'FLEET-CHECK status=passed' *>> $log
        [System.IO.File]::ReadAllBytes($log)[0..1] | Should -Be $script:utf16Mark
        Read-Tail $log 200 | Should -Be @('first line', 'FLEET-CHECK status=passed')
    }
    It 'returns the last lines of a three-megabyte UTF-16 transcript quickly' {
        $log = Write-Transcript $script:utf16Mark ([System.Text.Encoding]::Unicode.GetBytes((Get-NumberedText 45000)))
        (Get-Item -LiteralPath $log).Length | Should -BeGreaterThan 3000000
        $clock = [System.Diagnostics.Stopwatch]::StartNew()
        Read-Tail $log 3 | Should -Be @(
            'line 0044997 passed in the fixture',
            'line 0044998 passed in the fixture',
            'line 0044999 passed in the fixture'
        )
        $clock.Elapsed.TotalSeconds | Should -BeLessThan 60
    }
    It 'drops the line its window opens inside' {
        # 72 bytes a line in UTF-16, and 262144 is no multiple of 72, so the
        # window opens inside line 36359 and the 3640 lines after it are read.
        $log = Write-Transcript $script:utf16Mark ([System.Text.Encoding]::Unicode.GetBytes((Get-NumberedText 40000)))
        $printed = Read-Tail $log 100000
        $printed.Count | Should -Be 3640
        $printed[0] | Should -BeExactly 'line 0036360 passed in the fixture'
        $printed[-1] | Should -BeExactly 'line 0039999 passed in the fixture'
    }
    It 'decodes an odd-length UTF-16 transcript on character boundaries' {
        $log = Write-Transcript $script:utf16Mark ([System.Text.Encoding]::Unicode.GetBytes((Get-NumberedText 40000))) ([byte[]](0))
        (Read-Tail $log 3)[0..1] | Should -Be @('line 0039998 passed in the fixture', 'line 0039999 passed in the fixture')
    }
    It 'reads a marked UTF-8 transcript without its mark' {
        $log = Write-Transcript $script:utf8Mark ([System.Text.Encoding]::UTF8.GetBytes((Get-NumberedText 3)))
        Read-Tail $log 200 | Should -Be @(
            'line 0000000 passed in the fixture',
            'line 0000001 passed in the fixture',
            'line 0000002 passed in the fixture'
        )
    }
    It 'reads an unmarked transcript as UTF-8' {
        $log = Write-Transcript ([byte[]]@()) ([System.Text.Encoding]::UTF8.GetBytes("r$([char]0xE9)sum$([char]0xE9)`nFLEET-CHECK status=failed`n"))
        Read-Tail $log 1 | Should -Be @('FLEET-CHECK status=failed')
    }
    It 'prints nothing for a build that wrote no transcript' {
        @(Read-Tail (Join-Path $TestDrive 'never.txt.log') 200).Count | Should -Be 0
    }
}
