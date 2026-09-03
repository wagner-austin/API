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
    "fired",
    "command_rejected",
]
"""Shoot resolutions from the per-shot ammo-consumption ledger
(consumption = hit, user contract 2026-07-02) plus the 0x52 rejection
and the executor's target-not-tracked discard (the Phase 0 residue:
the race guard against the tank vanishing between plan and dispatch).

``fired`` (2026-08-21) is the ground-aimed shot's resolution: a
clearance shot targets a tile, not a tank, so hit/miss semantics never
apply — its own 0x53 echo is the server's receipt that the shot was
accepted, billed, and fired. Before this label existed, shoot was the
ONLY action kind with no completion path for a whole class of its
dispatches: every clearance shot's decision died ``superseded``
(soak bot-20260821-013519: 13 wire dispatches, 12/12 superseded,
0 completions), which the liveness counter misread as a livelock —
the detector's first live catch was a catch of itself."""

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

ActionOutcome = (
    ScanOutcome
    | MoveOutcome
    | TeleportOutcome
    | CollectOutcome
    | MapOpenOutcome
    | ShootOutcome
    | ScopeOutcome
)
"""Union of all seven per-kind outcome vocabularies.

COMPOSED from them since 2026-09-03, not re-listed. It used to be a
hand-written flat Literal that repeated every member, which made the
seven per-kind unions dead exports — documented, published in
``__all__``, and annotated nowhere — while this list quietly became
the only thing anyone read. The two agreed exactly (17 members, no
difference either way) at the moment they were joined, so composing
changed no type and removed the copy that could have drifted.

Adding an outcome to a kind now widens this automatically, and a
label that belongs to no kind cannot be added here at all."""

LIVENESS_STALL_STREAK = 12
"""Consecutive zero-dispatch replans of one kind that mean a livelock.

A superseded close counts toward the streak ONLY when the closed
decision's command never reached the wire (the executor marks every
real dispatch via ``mark_decision_dispatched``) — one undispatched
replan is normal, a streak means the planner keeps deriving an
identical plan some downstream veto keeps refusing without feedback
(the planner/veto gap class, [[fleet-coordination]] gatherer
livelock). A superseded close of a DISPATCHED decision resets the
streak instead: the planner's output demonstrably reached the wire,
so no livelock is in progress (2026-08-21 correction — before the
dispatch gate, combat re-aims and outcome-less clearance shots
counted as "zero dispatches" and the counter's first live catch was
a false positive on 12 dispatched-and-echoed shots). Empirical
basis, 459-run archive sweep 2026-08-20: the pre-gate healthy
ceiling was 7 (dispatched combat re-aims, which no longer count);
the one livelock in the archive ran 93 genuinely undispatched
replans. Set above the old ceiling with margin, far below the
pathology. Consumed live (``liveness_stall`` diagnostic at the
crossing) and post-run (the issue report's streak scan)."""

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
