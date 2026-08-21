"""The complete outcome vocabulary: six per-kind unions + the total union.

Every outcome label the bot can record, one Literal per action kind,
mirroring the real resolution signals of the HFSM completion gates,
the 0x52 rejection paths, the stall timeouts, and the executor
discards (the class the 2026-07-06 20:47:31 deadlock hid in -- now
first-class recorded outcomes).

All six unions live in this one top-level module (rather than each
kind's module in the ``outcome`` subpackage) so the ring and emit
plumbing can import the total union without triggering the
subpackage ``__init__`` -- keeps the import graph acyclic.
"""

from __future__ import annotations

from typing import Literal

ScanOutcome = Literal[
    "radar_complete",
    "superseded",
    "stall_timeout",
    "command_rejected",
]
"""Scan resolutions: radar sweep done, stalled out, or 0x52-rejected."""

MoveOutcome = Literal[
    "position_reached",
    "superseded",
    "movement_rejected",
    "command_rejected",
    "stall_timeout",
]
"""Move resolutions. The former ``discarded_hostile_mine`` executor
discard was removed 2026-07-20: hostile mines are composed into the
decision terrain (``compose_decision_terrain``), so the planner cannot
produce a mined walk destination in the first place."""

TeleportOutcome = Literal[
    "landed_exact",
    "superseded",
    "landed_inexact",
    "command_rejected",
    "stall_timeout",
]
"""Teleport resolutions. The ``discarded_*`` labels are the executor
validation classes from the rejection-loop audit. The former
``discarded_hostile_mine`` was removed 2026-07-20: it was wrong physics
(the server displaces off mined tiles on landing, so a mined teleport
target is safe) and it created the silent loop the audit predicted."""

CollectOutcome = Literal[
    "position_reached",
    "superseded",
    "container_consumed",
    "movement_rejected",
    "command_rejected",
    "pickup_empty",
    "clamped_transfer",
    "inventory_full",
    "stall_timeout",
]
"""Collect resolutions. ``discarded_no_container`` /
``discarded_kind_mismatch`` are the executor pickup-validation
discards (container gone, or fuel/equipment kind disagreement).

The three typed 0x52 resolutions (2026-07-19) replace the blanket
``command_rejected`` for their codes so the ledger distinguishes what
actually happened: ``pickup_empty`` (code 4 -- the container was
drained; belief removed), ``clamped_transfer`` (code 5 -- the server
transferred ``min(volume, headroom)`` and kept the remainder; this is
a SUCCESS, not a failure -- the 5-min soak 2026-07-19 gained +2472
fuel across four of these while the ledger filed them as rejections),
and ``inventory_full`` (code 7 -- authoritative all-slots-full
statement; beliefs reconciled). ``command_rejected`` remains for the
genuine refusals (code 0 geometry, code 1 can't-go)."""

MapOpenOutcome = Literal[
    "map_data_processed",
    "superseded",
    "stall_timeout",
    "command_rejected",
]
"""Map-open resolutions."""

ShootOutcome = Literal[
    "hit",
    "superseded",
    "miss",
    "command_rejected",
]
"""Shoot resolutions from the per-shot ammo-consumption ledger
(consumption = hit, user contract 2026-07-02) plus the 0x52 rejection
and the executor's target-not-tracked discard (the Phase 0 residue:
the race guard against the tank vanishing between plan and dispatch)."""

ScopeOutcome = Literal[
    "confirmed",
    "superseded",
    "stall_timeout",
]
"""Scope-pan resolutions ([[viewport-shift-protocol]]): the answering
0x5A confirmed the shifted window (median exactly one server tick,
p95 two, across 759 archived pans), or the pan stalled out. Promoted
from fire-and-forget 2026-08-20: an untracked pan let the next tick's
radar or map_open dispatch into the scope-pending window the server
silently drops commands in — half of all scan stalls ever recorded."""

ActionOutcome = Literal[
    # scan
    "radar_complete",
    # move / collect shared
    "position_reached",
    "movement_rejected",
    # teleport
    "landed_exact",
    "landed_inexact",
    # collect
    "container_consumed",
    "pickup_empty",
    "clamped_transfer",
    "inventory_full",
    # map_open
    "map_data_processed",
    # shoot
    "hit",
    "miss",
    # scope
    "confirmed",
    # shared
    "command_rejected",
    "stall_timeout",
    "superseded",
]
"""Union of all seven per-kind outcome vocabularies."""

LIVENESS_STALL_STREAK = 12
"""Consecutive zero-dispatch replans of one kind that mean a livelock.

A zero-duration ``superseded`` is a decision the planner replaced
before anything was dispatched — one replan is normal, a streak means
the planner keeps deriving an identical plan some downstream veto
keeps refusing without feedback (the planner/veto gap class,
[[fleet-coordination]] gatherer livelock). Empirical basis, 459-run
archive sweep 2026-08-20: the healthy ceiling is 7 (combat re-aiming
while a shot resolves); the one livelock in the archive ran 93. Set
above the healthy ceiling with margin, far below the pathology.
Consumed live (``liveness_stall`` diagnostic at the crossing) and
post-run (the issue report's streak scan)."""

__all__ = [
    "LIVENESS_STALL_STREAK",
    "ActionOutcome",
    "CollectOutcome",
    "MapOpenOutcome",
    "MoveOutcome",
    "ScanOutcome",
    "ScopeOutcome",
    "ShootOutcome",
    "TeleportOutcome",
]
