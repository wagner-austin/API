"""Combat feedback type for protocol-based hit/miss detection.

CombatFeedback is determined by the protocol decoder: when we shoot,
if a CombatHit message arrives where we are the attacker, it's a "hit".
If no CombatHit arrives, it's a "miss". Kills are handled separately
by the Deactivation protocol message which sets victim position to (0,0).
"""

from __future__ import annotations

from typing import Literal

CombatFeedback = Literal["hit", "miss", ""]


__all__ = [
    "CombatFeedback",
]
