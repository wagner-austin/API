"""MCPs pcsession-mcp needs more workers than lenovold affords (MCPs board task 007bfc7e).

A push's gate checks every project it touches, and the push lands when the
last verdict is in. On 2026-11-07 the gate of MCPs 9bffb9bb5 queued
pcsession-mcp at 00:56:34Z, lenovold claimed it at 00:57:46Z at 2 workers,
and it closed at 01:01:10Z: a 52 s workspace build and a 134 s check of 359
cases whose slowest file took 1 s, the last verdict of the gate by 109 s,
while sedona, serendipity, lacrymal, lavender and pendragon ran the same
check in 15 to 62 s over the twenty runs before it. The push took 388.4 s
from its gate to every host deployed. API 76bcb74c1 kept MCPs/packages/db
and 02eca1644 MCPs/packages/session-audit-linux off lenovold the same way.
"""

from __future__ import annotations

from fleet.contracts.workspace import decode_fleet_workspace, require_project
from tests._registry import shipped_document


def test_pcsession_mcp_needs_more_workers_than_lenovold_affords() -> None:
    """Three workers, so lenovold's runner, which affords two, is never offered it."""
    workspace = decode_fleet_workspace(shipped_document())
    assert require_project(workspace, "MCPs/pcsession-mcp")["minimum_workers"] == 3
