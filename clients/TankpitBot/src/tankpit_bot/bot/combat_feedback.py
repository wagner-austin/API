"""Combat feedback type for protocol-based hit/miss detection.

CombatFeedback is determined by the 0x53 ShootEvent dispatch: when we
shoot, the dispatcher checks tile occupancy at the wire-reported target
tile. If a tank is at that tile, it's a "hit"; if not, it's a "miss".
Kills are handled separately by the 0x41 Deactivation message which
sets victim position to (0,0).

(Function names ``mark_combat_hit`` / ``check_and_clear_combat_hit`` /
``peek_combat_hit`` predate the 2026-06-19 decoder unification; they
refer to the deleted container ``CombatHit`` decoder by historical
name only. Their semantics are the ShootEvent tile-occupancy check.)
"""

from __future__ import annotations

from typing import Literal

CombatFeedback = Literal["hit", "miss", ""]


__all__ = [
    "CombatFeedback",
]
