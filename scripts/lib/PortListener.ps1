Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# Which processes listen on a local TCP port: the one read behind every API
# script that waits for a port to open or refuses one already held
# (clients/RustedWarfareBot's RwMake.ps1 and tools/fleet's
# install-windows-testdb.ps1, which each carried their own copy until board
# task 2a044a37; services/grandma-api's GrandmaService.ps1 was a third until
# its Windows host mode was deleted for board task 703fec5a).
#
# READ FROM CIM, NOT FROM Get-NetTCPConnection. The cmdlet reads the same
# class, MSFT_NetTCPConnection, but answers a query matching nothing by
# writing an error, and its -State Listen did so with listeners present in
# MCPs' PowerShell harness workers on austinpc on 2026-10-21, so a free port
# and a held one both read as a failure there. Get-CimInstance answers a
# filter matching nothing with no objects, so a free port is an empty answer
# and any real failure to read the table still throws.

# MSFT_NetTCPConnection's State for a listening socket.
$script:PortListenState = 2

function Get-PortListener {
    <#
    .SYNOPSIS
        The processes listening on a local TCP port, each once.
    .PARAMETER Port
        The port.
    .PARAMETER Address
        Only sockets bound to this local address; any address when empty.
    .OUTPUTS
        UInt32[]: the owning process ids, in the order the table lists them;
        none when nothing listens.
    #>
    [OutputType([uint32[]])]
    param([Parameter(Mandatory)][int]$Port, [string]$Address = '')
    $filter = "LocalPort = $Port AND State = $($script:PortListenState)"
    if ($Address -ne '') {
        $filter = "LocalAddress = '$Address' AND $filter"
    }
    return [uint32[]]@(Get-CimInstance -Namespace ROOT/StandardCimv2 -ClassName MSFT_NetTCPConnection -Filter $filter |
        ForEach-Object { $_.OwningProcess } | Select-Object -Unique)
}
