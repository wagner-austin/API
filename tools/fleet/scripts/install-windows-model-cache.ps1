<#
.SYNOPSIS
    Give a Windows fleet node the gte-small model directory that MCPs
    packages/embed and mcp-proxy's live tool-search block load from its
    fleet cache, so the node's first check of either reads the model
    instead of refusing by name.

.DESCRIPTION
    Why (MCPs board tasks 1d8e2e0e, f7085447 and f28396f8): since 2026-10-07
    packages/embed loads Xenova/gte-small from <CORVIS_FLEET_CACHE>/models
    with remote fetch switched off, and a node whose directory lacks the
    model fails the block with EMBED_MODEL_MISSING naming the directory,
    never a fetch to huggingface.co. Every node's directory was filled by
    hand that day; colossus (enrolled 2026-10-09) and loki (made a testdb
    node the same day) each took their first mcp-proxy check with no
    directory and failed it. The linux provisioners fill it in MCPs
    scripts/host/lib/fleet-node.sh (step_fleet_model_cache); this script is
    the Windows node's fill, run over ssh as the build account beside
    install-windows-testdb.ps1, the step that gives the node its test
    database.

    WHAT IS FETCHED. The four files a feature-extraction pipeline opens:
    config.json, tokenizer.json, tokenizer_config.json and onnx/model.onnx,
    the fp32 weights, from the Hub's resolve URL, each only where the
    directory lacks it or holds it empty. A download lands in a .partial
    file and is renamed into place, so an interrupted run leaves no half
    file a rerun would take for finished; a download that throws removes
    its .partial before the error propagates.

    WHAT IS MEASURED. The weights' size, because a WebClient download that
    ends early throws, but a file already in the directory may be short
    from an earlier hand copy: under MinimumWeightBytes the script throws
    MODEL_CACHE_TRUNCATED naming the file, and the operator removes it and
    reruns. A rerun on a filled directory fetches nothing and measures the
    same, so it is the audit.

.PARAMETER Root
    The fleet cache's models directory, <stage root>/cache/models, where a
    fleet build's CORVIS_FLEET_CACHE points (MCPs packages/embed
    resolveModelDirectory).

.PARAMETER Repo
    The model's Hub id, which is also its directory under Root.

.PARAMETER Files
    The files the pipeline opens, relative to the model's directory, with
    forward slashes as the Hub names them.

.PARAMETER BaseUrl
    Where each file is fetched from, the Hub's resolve URL of the repo's
    main branch; the test names a file:// directory.

.PARAMETER MinimumWeightBytes
    The least the weights may be: the real file is 133,093,490 bytes.
#>
[CmdletBinding()]
param(
    [string]$Root = 'C:\fleet\stage\cache\models',
    [string]$Repo = 'Xenova/gte-small',
    [string[]]$Files = @('config.json', 'tokenizer.json', 'tokenizer_config.json', 'onnx/model.onnx'),
    [string]$BaseUrl = 'https://huggingface.co/Xenova/gte-small/resolve/main',
    [long]$MinimumWeightBytes = 100000000
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$model = Join-Path $Root ($Repo -replace '/', '\')
$fetched = 0
foreach ($file in $Files) {
    $path = Join-Path $model ($file -replace '/', '\')
    if ([System.IO.File]::Exists($path) -and (Get-Item -LiteralPath $path).Length -gt 0) {
        continue
    }
    [void][System.IO.Directory]::CreateDirectory([System.IO.Path]::GetDirectoryName($path))
    $partial = "$path.partial"
    # File.Delete on a path that is absent is a no-op, so the empty file
    # the exists check let through and the partial a failed download may
    # or may not have opened are both removed without a branch.
    try {
        (New-Object System.Net.WebClient).DownloadFile("$BaseUrl/$file", $partial)
    } catch {
        [System.IO.File]::Delete($partial)
        throw
    }
    [System.IO.File]::Delete($path)
    [System.IO.File]::Move($partial, $path)
    $fetched += 1
}

$weights = Join-Path $model 'onnx\model.onnx'
$size = (Get-Item -LiteralPath $weights).Length
if ($size -lt $MinimumWeightBytes) {
    throw "MODEL_CACHE_TRUNCATED: $weights is $size bytes, under the $MinimumWeightBytes the weights take; remove it and rerun"
}
Write-Output "fleet-model-cache: measured $Repo under $Root, model.onnx $size bytes, fetched $fetched of $($Files.Count) file(s)"
