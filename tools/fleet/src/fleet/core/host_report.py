"""What a WSL node's Windows host sees when the node itself does not answer.

MCPs board task 45a4f22b. From 11:36Z on 2026-09-29 every tick for
lavender-wsl logged only ``did not answer ... Connection timed out during
banner exchange``, while lavender itself answered ssh with 0.3 GB free and
its distro Running: CI jobs on the same VM had starved sshd. Finding that
took a session measuring the host by hand. A node that declares its host
(:attr:`fleet.contracts.node.NodeConfig.wsl_host`) now gets both facts in
the tick's own line: the host's free memory and disk from the Windows
capacity probe every tick already runs, and each distro's state from
``wsl.exe -l -v``.

No new remote script: the capacity probe is the dialect's constant, and the
WSL listing is a native argv, so nothing here is rendered PowerShell.
"""

from __future__ import annotations

import pathlib

from typing_extensions import TypedDict

from fleet.contracts.workspace import FleetWorkspace, require_node
from fleet.core import probe, records, remote

#: The listing's argv. ``-v`` adds each distro's state and WSL version.
WSL_LIST_ARGV: tuple[str, ...] = ("wsl.exe", "-l", "-v")


class Distro(TypedDict):
    """One row of ``wsl.exe -l -v``.

    Attributes:
        name: The distro's name.
        state: Its state as WSL prints it, e.g. ``Running`` or ``Stopped``.
    """

    name: str
    state: str


def parse_wsl_list(output: str) -> list[Distro]:
    """Read ``wsl.exe -l -v`` as it arrives over ssh.

    wsl.exe writes UTF-16LE with no byte-order mark whatever the console
    says, so the text the command runner decodes as UTF-8 carries a NUL
    after every character (measured on lavender, 2026-09-29); they are
    dropped before the columns are read.

    Args:
        output: The command's standard output.

    Returns:
        Each distro with its state, in listing order; the header row and
        the default marker ``*`` are not part of any row.
    """
    rows: list[Distro] = []
    for line in output.replace("\x00", "").splitlines():
        words = line.split()
        if words and words[0] == "*":
            words = words[1:]
        if len(words) < 2 or words[0] == "NAME":
            continue
        rows.append(Distro(name=words[0], state=words[1]))
    return rows


def describe_wsl_host(workspace: FleetWorkspace, ledger: pathlib.Path, host_name: str) -> str:
    """Say what a WSL node's host reports, for a tick that could not reach the node.

    Args:
        workspace: The workspace, which declares the host.
        ledger: The workspace's ledger, for the dispatches live on the host
            that the capacity probe's reading carries.
        host_name: The host's workspace name, the node's ``wsl_host``.

    Returns:
        One clause naming the host's free memory and disk and each distro's
        state, or why the host could not say.
    """
    host = require_node(workspace, host_name)
    probed = probe.attempt_probe(host, live_runs=records.live_runs(ledger, node=host_name))
    state = probed["state"]
    if state is None:
        return f"its host {host_name} did not answer either: {probed['reason']}"
    listed = remote.attempt_ssh(host["host"], WSL_LIST_ARGV)
    failure = listed["failure"]
    if failure is not None:
        distros = f"wsl.exe -l -v failed: {failure['message']}"
    else:
        rows = parse_wsl_list(listed["output"])
        distros = (
            ", ".join(f"{row['name']} {row['state']}" for row in rows)
            if rows
            else "no distro listed"
        )
    return (
        f"its host {host_name} answers with {state['free_ram_gb']:.1f} GB of "
        f"{host['ram_gb']:.1f} GB RAM and {state['free_disk_gb']:.1f} GB disk free; "
        f"wsl: {distros}"
    )


__all__ = ["WSL_LIST_ARGV", "Distro", "describe_wsl_host", "parse_wsl_list"]
