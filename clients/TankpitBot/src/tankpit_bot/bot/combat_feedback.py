"""Combat feedback type for protocol-based shot-outcome detection.

CombatFeedback is determined by the 0x53 ShootEvent dispatch: the
per-shot ``weapon`` byte is the server's ammo-consumption ledger --
``weapon > 0`` debited a consumable and the shot landed ("hit"),
``weapon = 0`` was a free single at empty ground ("miss"). "rejected"
means the server refused the shoot command outright with a 0x52
Supervisor error (e.g. code 0 "You can't do this" for an aim outside
the viewport) -- no ShootEvent and no ammo delta ever arrive for a
rejected shot, so it is neither a hit nor a miss (live run 2026-07-03
20:34: five rejected pursuit shots at an off-viewport aim produced
zero wire feedback and looped for 4 s each). Kills are handled
separately by the 0x41 Deactivation message.

(Function names ``mark_combat_hit`` / ``check_and_clear_combat_hit`` /
``peek_combat_hit`` predate the 2026-06-19 decoder unification; they
refer to the deleted container ``CombatHit`` decoder by historical
name only.)
"""

from __future__ import annotations

from typing import Literal

CombatFeedback = Literal["hit", "miss", "rejected", ""]


__all__ = [
    "CombatFeedback",
]
