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
    "discarded_hostile_mine",
]
"""Move resolutions. ``discarded_hostile_mine`` is the executor
discard (destination is a known hostile-mine tile) -- previously a
silent ``emit_ai`` line."""

TeleportOutcome = Literal[
    "landed_exact",
    "superseded",
    "landed_inexact",
    "command_rejected",
    "stall_timeout",
    "discarded_hostile_mine",
    "discarded_combat_target_stale",
    "discarded_resource_target_stale",
    "discarded_resource_target_invalid",
]
"""Teleport resolutions. The four ``discarded_*`` labels are the
executor validation classes from the rejection-loop audit."""

CollectOutcome = Literal[
    "position_reached",
    "superseded",
    "container_consumed",
    "movement_rejected",
    "command_rejected",
    "stall_timeout",
    "discarded_no_container",
    "discarded_kind_mismatch",
]
"""Collect resolutions. ``discarded_no_container`` /
``discarded_kind_mismatch`` are the executor pickup-validation
discards (container gone, or fuel/equipment kind disagreement)."""

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
    "discarded_target_not_tracked",
]
"""Shoot resolutions from the per-shot ammo-consumption ledger
(consumption = hit, user contract 2026-07-02) plus the 0x52 rejection
and the executor's target-not-tracked discard (the Phase 0 residue:
the race guard against the tank vanishing between plan and dispatch)."""

ActionOutcome = Literal[
    # scan
    "radar_complete",
    # move / collect shared
    "position_reached",
    "movement_rejected",
    # teleport
    "landed_exact",
    "landed_inexact",
    "discarded_combat_target_stale",
    "discarded_resource_target_stale",
    "discarded_resource_target_invalid",
    # move / teleport shared
    "discarded_hostile_mine",
    # collect
    "container_consumed",
    "discarded_no_container",
    "discarded_kind_mismatch",
    # map_open
    "map_data_processed",
    # shoot
    "hit",
    "miss",
    "discarded_target_not_tracked",
    # shared
    "command_rejected",
    "stall_timeout",
    "superseded",
]
"""Union of all six per-kind outcome vocabularies."""

__all__ = [
    "ActionOutcome",
    "CollectOutcome",
    "MapOpenOutcome",
    "MoveOutcome",
    "ScanOutcome",
    "ShootOutcome",
    "TeleportOutcome",
]
