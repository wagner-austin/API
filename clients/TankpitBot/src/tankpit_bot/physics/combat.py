"""Combat timing physics.

One fact lives here today: the server-side homing-reroute TTL. The
consumption-equals-hit rule (dual/missile/homing debit exactly 1
round per LANDED shot; the ammo counter IS the hit detector) is a
bookkeeping identity, not a constant — it is documented in
``wiki/pages/shoot-event-format.md`` and consumed by the ledger's
0x49 reconciliation, so it carries no symbol here.
"""

from __future__ import annotations

REROUTE_TTL_MS = 12_000
"""ESTIMATE — how long after a target's 0x58 TankRemove the server
keeps rerouting id-targeted shots to the departed tank. Measured
boundary is [11.0, 13.0] s fire-time (run 2026-07-19 22:30: hits at
+0.65..+11.0 s all debited ammo; +13.0 s drew no debit). 12_000 ms is
the midpoint; the ``tank_removed`` diagnostic timestamps every 0x58
so future pursuit misses narrow it.
Wiki: [[shoot-event-format]]#reroute-ttl-ms."""

__all__ = [
    "REROUTE_TTL_MS",
]
