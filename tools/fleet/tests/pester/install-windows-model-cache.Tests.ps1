Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# tools/fleet/scripts/install-windows-model-cache.ps1, the Windows fleet
# node's gte-small directory (MCPs board task f28396f8). Each case runs the
# real script against a source directory it writes under $TestDrive, named
# as a file:// base URL so the same WebClient download the Hub gets is the
# one measured, into a root of its own; the weights' minimum is lowered to
# the bytes a case writes.

BeforeAll {
    $script:install = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) 'scripts\install-windows-model-cache.ps1'
    $script:files = @('config.json', 'tokenizer.json', 'tokenizer_config.json', 'onnx/model.onnx')
    $script:weights = '0123456789abcdef'

    function Initialize-Source {
        <#
        .SYNOPSIS
            A directory holding the named files, each with its text, as a
            file:// base URL.
        .PARAMETER Entries
            File name, with forward slashes, to its text.
        .OUTPUTS
            String: the base URL.
        #>
        param([hashtable]$Entries)
        $directory = Join-Path $TestDrive ('source-' + [guid]::NewGuid().ToString('N'))
        foreach ($name in $Entries.Keys) {
            $path = Join-Path $directory ($name -replace '/', '\')
            [void][System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($path))
            [System.IO.File]::WriteAllText($path, $Entries[$name], [System.Text.Encoding]::ASCII)
        }
        return ([Uri]$directory).AbsoluteUri
    }

    function Initialize-FullSource {
        <#
        .SYNOPSIS
            A source holding every file, the weights 16 bytes.
        .OUTPUTS
            String: the base URL.
        #>
        $entries = @{}
        foreach ($name in $script:files) {
            $entries[$name] = "fetched $name"
        }
        $entries['onnx/model.onnx'] = $script:weights
        return Initialize-Source $entries
    }

    function Initialize-Root {
        <#
        .SYNOPSIS
            A models root whose Xenova/gte-small holds the given files.
        .PARAMETER Entries
            File name, with forward slashes, to its text.
        .OUTPUTS
            String: the root.
        #>
        param([hashtable]$Entries)
        $root = Join-Path $TestDrive ('root-' + [guid]::NewGuid().ToString('N'))
        foreach ($name in $Entries.Keys) {
            $path = Join-Path $root ('Xenova\gte-small\' + ($name -replace '/', '\'))
            [void][System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($path))
            [System.IO.File]::WriteAllText($path, $Entries[$name], [System.Text.Encoding]::ASCII)
        }
        return $root
    }

    function Get-ModelText {
        <#
        .SYNOPSIS
            The text of one file under the root's Xenova/gte-small.
        .OUTPUTS
            String.
        #>
        param([string]$Root, [string]$Name)
        return [System.IO.File]::ReadAllText((Join-Path $Root ('Xenova\gte-small\' + ($Name -replace '/', '\'))))
    }

    function Get-ModelFileList {
        <#
        .SYNOPSIS
            Every file under the root, relative to it with forward slashes,
            sorted.
        .OUTPUTS
            String[].
        #>
        param([string]$Root)
        $prefix = (Join-Path $Root '').Length
        return @([System.IO.Directory]::GetFiles($Root, '*', 'AllDirectories') |
                ForEach-Object { $_.Substring($prefix) -replace '\\', '/' } | Sort-Object)
    }

    # The script fails by throwing; a non-zero $LASTEXITCODE left behind
    # would be a native it ran, and it runs none, so one is itself an error.
    function Invoke-Install {
        param([hashtable]$Parameters)
        $global:LASTEXITCODE = 0
        & $script:install @Parameters
        if ($LASTEXITCODE -ne 0) {
            throw "TEST_INSTALL_EXITED: install-windows-model-cache exited $LASTEXITCODE"
        }
    }
}

Describe 'install-windows-model-cache' {
    It 'fetches every file into a fresh root, renames each from its partial, measures the weights and counts the fetches' {
        $root = Join-Path $TestDrive 'fresh'
        $said = @(Invoke-Install @{ Root = $root; BaseUrl = (Initialize-FullSource); MinimumWeightBytes = 16 })
        $said | Should -Be @("fleet-model-cache: measured Xenova/gte-small under $root, model.onnx 16 bytes, fetched 4 of 4 file(s)")
        Get-ModelFileList $root | Should -Be @(
            'Xenova/gte-small/config.json', 'Xenova/gte-small/onnx/model.onnx',
            'Xenova/gte-small/tokenizer.json', 'Xenova/gte-small/tokenizer_config.json')
        Get-ModelText $root 'config.json' | Should -BeExactly 'fetched config.json'
        Get-ModelText $root 'onnx/model.onnx' | Should -BeExactly $script:weights
    }
    It 'fetches only what the directory lacks or holds empty, and keeps the rest as it is' {
        $root = Initialize-Root @{ 'config.json' = 'kept'; 'tokenizer.json' = '' }
        $said = @(Invoke-Install @{ Root = $root; BaseUrl = (Initialize-FullSource); MinimumWeightBytes = 16 })
        $said | Should -Be @("fleet-model-cache: measured Xenova/gte-small under $root, model.onnx 16 bytes, fetched 3 of 4 file(s)")
        Get-ModelText $root 'config.json' | Should -BeExactly 'kept'
        Get-ModelText $root 'tokenizer.json' | Should -BeExactly 'fetched tokenizer.json'
        Get-ModelFileList $root | Should -Be @(
            'Xenova/gte-small/config.json', 'Xenova/gte-small/onnx/model.onnx',
            'Xenova/gte-small/tokenizer.json', 'Xenova/gte-small/tokenizer_config.json')
    }
    It 'fetches nothing from a filled directory and measures it, so a rerun is the audit' {
        $root = Initialize-Root @{
            'config.json' = 'c'; 'tokenizer.json' = 't'; 'tokenizer_config.json' = 'tc'; 'onnx/model.onnx' = $script:weights
        }
        $said = @(Invoke-Install @{ Root = $root; BaseUrl = 'file:///absent'; MinimumWeightBytes = 16 })
        $said | Should -Be @("fleet-model-cache: measured Xenova/gte-small under $root, model.onnx 16 bytes, fetched 0 of 4 file(s)")
    }
    It 'refuses weights under the minimum by name, so a short hand copy is never taken for the model' {
        $root = Initialize-Root @{
            'config.json' = 'c'; 'tokenizer.json' = 't'; 'tokenizer_config.json' = 'tc'; 'onnx/model.onnx' = 'short'
        }
        $weights = Join-Path $root 'Xenova\gte-small\onnx\model.onnx'
        { Invoke-Install @{ Root = $root; BaseUrl = 'file:///absent'; MinimumWeightBytes = 16 } } |
            Should -Throw -ExpectedMessage "MODEL_CACHE_TRUNCATED: $weights is 5 bytes, under the 16 the weights take; remove it and rerun"
    }
    It 'lets a failed download through with no partial file left, keeping what it fetched before it' {
        $source = Initialize-Source @{ 'config.json' = 'c'; 'tokenizer.json' = 't' }
        $root = Join-Path $TestDrive 'failing'
        { Invoke-Install @{ Root = $root; BaseUrl = $source; MinimumWeightBytes = 16 } } | Should -Throw
        Get-ModelFileList $root | Should -Be @('Xenova/gte-small/config.json', 'Xenova/gte-small/tokenizer.json')
    }
}
