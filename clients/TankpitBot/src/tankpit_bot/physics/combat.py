"""Combat timing physics.

One fact lives here today: the server-side homing-reroute TTL. The
consumption-equals-hit rule (dual/missile/homing debit exactly 1
round per LANDED shot; the ammo counter IS the hit detector) is a
bookkeeping identity, not a constant — it is documented in
``wiki/pages/shoot-event-format.md`` and consumed by the ledger's
0x49 reconciliation, so it carries no symbol here.
"""

from __future__ import annotations

REROUTE_TTL_MS = 12_920
"""How long after a living target's 0x58 TankRemove the server keeps
rerouting id-targeted shots to it. Corpus-swept 2026-07-22 across all
246 sessions (echo-paired: each sent id-shot at a removed-and-dark id
matched to its own 0x53, weapon=3 debit == hit): 704 hits and 137
misses, with hits dense up to +12.91 s, ZERO hits later, and a dense
miss wall from +12.93 s. Boundary [12.91, 12.93] s fire-time;
12_920 ms is the midpoint. Supersedes the single-run 2026-07-19
estimate (boundary [11.0, 13.0] s, midpoint 12_000 — which donated
one guaranteed pursuit hit per chase by quitting ~0.9 s early).
Wiki: [[shoot-event-format]]#reroute-ttl-ms."""

__all__ = [
    "REROUTE_TTL_MS",
]
