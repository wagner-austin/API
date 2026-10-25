"""Which shipped projects require Windows, and why (MCPs board tasks 6c89be43 and e35812cc).

loki is the fleet's one Windows node, at one check at a time, and it
declares no GPU. Every project that required windows could run nowhere
else, and one that required windows and gpu could run nowhere at all once
austinpc, the Windows box with the 3090 Ti, was reinstalled as the Linux
node demure on 2026-10-22. On 2026-10-24 the operator asked "what other
shit is stale?" and "why is loki getting so many jobs?? like we should have
barely any windows code."

services/Model-Trainer, libs/cleargbm_rs and libs/covenant_nn held no
Windows code (6c89be43). Of the 23 projects still requiring windows on
2026-10-25 (e35812cc), seven held none either: clients/TankpitBot/web
passed make check on Linux as it stood; Dashboards, Dashboards/rabbit,
metabolomics-dashboard and LSTM needed only their Makefiles ported from
SHELL := powershell.exe to the two-shell form of API scripts/make/shell.mk
(Dashboards 3418f26, metabolomics-dashboard aa6ad90, LSTM 2532210), with
Dashboards' to_bash_path read the same on Linux (16ac1b9); idle needed its
pre-commit hook committed executable (96a857b); and services/grandma-api
already asserts it runs no PowerShell (tests/test_linux_only.py).

So this file pins both halves against the shipped registry: the projects
that still require windows are exactly :data:`WINDOWS_REASONS`, each with
what in its own code needs Windows, and each of :data:`LINUX_PROJECTS`
requires no windows and an enabled Linux node, idle with memory to spare,
grants it workers. A tool tag (ffmpeg, chrome, hooks, go, pdftotext) is one
no declaration holds; a runner adds it on the tick its probe finds the
tool, so it is counted here as present and the project's fleet check on
Linux is what shows a node has it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fleet.contracts.node import LiveLoad, NodeConfig, NodeState
from fleet.contracts.tags import TOOL_TAG, NodeTag, node_tags
from fleet.contracts.workspace import FleetWorkspace, decode_fleet_workspace
from fleet.core.capacity import assess
from fleet.core.workspace_files import read_workspace_document

#: Every shipped project that requires windows, and what in its own code
#: needs a Windows host, read on 2026-10-25 (board task e35812cc).
#: MCPs' own three Windows projects, MCPs/execution, MCPs/packages/claude-hooks
#: and MCPs/scripts/ps-harness, left the registry on 2026-10-25 when the
#: operator ruled "so no windows shit then ok" (MCPs board task 5b040fdb): no
#: MCPs check waits on loki.
WINDOWS_REASONS = {
    "tools/ps-harness": (
        "runs MCPs' ps-harness over every tracked API .ps1, registering real S4U scheduled "
        "tasks on the elevated runner (its Makefile)"
    ),
    "chat": (
        "its Makefile runs under powershell.exe and its 32 .ps1 scripts are covered by Pester "
        "and PSScriptAnalyzer suites"
    ),
    "chat/execution": (
        "its Pester deploy suite (execution/tests/Deploy.Execution.ps1) installs "
        "cloudflared-windows-amd64.exe and reads the host's System32 nvidia-smi.exe"
    ),
    "clients/NavProbe": (
        "scripts/_test_hooks.py calls kernel32 through ctypes.WinDLL, which mypy and the "
        "tests reach only on Windows; on Linux mypy reports six errors there"
    ),
    "clients/TankpitBot": (
        "the Windows half of a pair with clients/TankpitBot-linux: its host_windows cases "
        "test Windows' own file-replace and process-exit behaviour (tests/_host.py)"
    ),
    "clients/TankpitBot-execution": (
        "its live cases play the public demo in the installed Microsoft Edge every Windows "
        "node ships (tests/live/_visitor.py)"
    ),
    "corvis-stick": (
        "its root Makefile runs under powershell.exe and its 61 .ps1 scripts are executed by "
        "Pester to 100 percent of commands and branch arms"
    ),
    "corvis-stick/execution": (
        "the Windows lane of the drive's real-machine smoke (execution/Makefile); "
        "corvis-stick/execution-linux is the Linux lane"
    ),
    "libs/instrument_io": (
        "its six Thermo .raw integration cases run the committed .NET "
        "ThermoRawFileParser.exe, which executes natively only on Windows, and no Linux node "
        "carries mono; on Linux they skip"
    ),
    "slime": (
        "scripts/performance-cores.ts reads Windows CPU sets and sets affinity through "
        "PowerShell, and only its real cases cover it under the 100 percent gate; on Linux "
        "they fail spawnSync powershell.exe ENOENT"
    ),
    "services/procart-api": (
        "tests/test_scripts_live_preview.py opens a real pygame window to exercise the display "
        "adapter, and a Linux fleet build has no display (fleet job a08291f3 on colossus: 'The "
        "video driver did not add any displays'); it leaves windows when board task 5e5c6116 "
        "gives Linux builds one"
    ),
    "tools/fleet-execution": (
        "the Windows lane of tools/fleet's host cases, which run rendered PowerShell for "
        "real; tools/fleet-execution-linux is the Linux lane"
    ),
    "tree-bot": (
        "its Makefile runs under powershell.exe and its 100 percent gate includes five "
        "folder-picker cases that run only on Windows (tests/ui/test_folder_picker.py)"
    ),
}

#: The projects whose windows tag was dropped because nothing in them needs it.
LINUX_PROJECTS = (
    "Dashboards",
    "Dashboards/rabbit",
    "LSTM",
    "clients/TankpitBot/web",
    "idle",
    "libs/cleargbm_rs",
    "libs/covenant_nn",
    "metabolomics-dashboard",
    "services/Model-Trainer",
    "services/grandma-api",
)


def _shipped() -> FleetWorkspace:
    """The registry this package ships.

    Returns:
        The decoded workspace.
    """
    return decode_fleet_workspace(
        read_workspace_document(Path(__file__).resolve().parents[1] / "registry")
    )


def _idle(node: NodeConfig) -> NodeState:
    """A node with nothing of the fleet's on it and memory to spare.

    Args:
        node: The node's declaration.

    Returns:
        Its state with 24 GB free, no live run, no reservation and no
        pressure read.
    """
    return NodeState(
        host=node["host"],
        free_ram_gb=24.0,
        free_disk_gb=100.0,
        live=LiveLoad(runs=0, workers=0, ram_gb=0.0),
        ci_slice=None,
        reserved_cores=0,
        action_cache=None,
        cpu_pressure_pct=None,
        io_pressure_pct=None,
    )


def test_the_projects_requiring_windows_are_exactly_those_with_a_reason() -> None:
    windows = sorted(
        name
        for name, project in _shipped()["projects"].items()
        if NodeTag.WINDOWS in project["required_tags"]
    )
    assert windows == sorted(WINDOWS_REASONS)


@pytest.mark.parametrize("project_name", LINUX_PROJECTS)
def test_the_project_requires_no_windows_and_an_enabled_linux_node_grants_it(
    project_name: str,
) -> None:
    workspace = _shipped()
    project = workspace["projects"][project_name]
    detected = frozenset(project["required_tags"]) & frozenset(TOOL_TAG.values())

    assert NodeTag.WINDOWS not in project["required_tags"]
    granted = [
        name
        for name, node in sorted(workspace["nodes"].items())
        if node["platform"] == "linux"
        and node["enabled"]
        and assess(node, _idle(node), project, node_tags(node) | detected)["workers"] > 0
    ]
    assert granted, f"no enabled Linux node grants {project_name} workers"
