"""The knowledge-floor law: when a scan stamp still answers a question.

Two constructors for one threshold. A coverage predicate asks "is this
stamp still knowledge?"; the answer is ``stamp >= floor``, and the two
laws below build the floor. The settled-knowledge law
([[flag-triage-20260902]] rows 3-5) is the session default — planners
reach it through
:meth:`~tankpit_bot.sniffer.world_service_beliefs.WorldServiceBeliefsMixin.knowledge_floor_ms`,
which supplies the foreign-human watermark; the pure-clock law remains
for questions that must stay time-bounded regardless of world dynamism.
"""

from __future__ import annotations

FORAGE_COVERAGE_TTL_MS = 180000
"""How long a scan mark answers "worth another RADAR?" UNDER HUMAN
PRESENCE.

The clock is the FALLBACK arm of the settled-knowledge law
(:func:`settled_knowledge_floor_ms`, [[flag-triage-20260902]] rows
3-5), not the mechanism: ground changes only when someone changes it,
and the archive falsified every autonomous-change theory —
[[game-economy]]: container "respawn" was our own exposure 605/605,
refills are discrete deposits with corr(Δv,Δt) = -0.13; and
[[enemy-bot-behavior]]: practice bots never deliberately collect.
So a scan made after the last foreign human was seen stays valid
until one appears, and this interval prices staleness only while a
human's unobserved consumption is actually possible. The pure-clock
predicate re-scanned a static Practice room forever: 49% of a live
session's radars landed within 7 tiles of an earlier radar, and 139
frontier teleports never left the current viewport."""

HARVEST_MEMORY_TTL_MS = 600000
"""How long harvested/barren ground stays vetoed UNDER HUMAN PRESENCE.

The forage TTL above answers "is this ground worth another RADAR" and
deliberately ages out fast. Harvest memory answers "did we already
LEARN this ground is worthless" and must outlive it: run
bot-20260729-232252 re-hopped picked-clean viewports the moment the
180 s coverage expired (63% zero-yield hops,
[[flag-triage-20260729]] F2). Two vetoes share this window: drained
container beliefs (`_landing_viewport_known_empty`) and barren scan
memory (`is_viewport_scanned_within` — scanned recently, revealed
nothing). Like the forage TTL, this clock is the settled-knowledge
law's fallback arm ([[flag-triage-20260902]]): barren ground stays
barren until a foreign human — the only agent that changes ground —
is actually seen.
"""


FERRY_LOOK_TTL_MS = 30000
"""How long a no-ferry scope-scout look answers "worth another pan?"
UNDER HUMAN PRESENCE.

Ferry positions are positional, not clocked ([[ferry-mechanics]]
no-drift law): a ferry moves only when someone rides it, so a pan
that showed a goal's boarding water WITHOUT a ferry is a fact that
holds until a foreign human — the only agent that rides ferries in —
is actually seen. This clock is the settled-knowledge law's fallback
arm for the scout's per-goal look memory
(``scope_scout_looks``, [[flag-triage-20260902]] row 8): with a
human about, a look decays on the scout's own half-minute rhythm;
settled, it is permanent. The memoryless scout re-panned the same
water forever — 31 pans, 1 acted on, in the flagged run."""


def ttl_floor_ms(now_ms: int, ttl_ms: int) -> int:
    """Return the scan-stamp validity floor under pure clock aging.

    The pre-settled-law semantics, kept for questions that must stay
    time-bounded regardless of world dynamism — the fleet report's
    publish horizon uses it so shared coverage stays a bounded
    payload.

    Args:
        now_ms: Current timestamp.
        ttl_ms: The question's staleness window.

    Returns:
        The instant at or after which a stamp still answers the
        question.
    """
    return now_ms - ttl_ms


def settled_knowledge_floor_ms(
    now_ms: int,
    ttl_ms: int,
    last_foreign_human_seen_ms: int,
) -> int:
    """Return the validity floor under the settled-knowledge law.

    A stamp is valid when it is recent (the clock arm) OR when it
    postdates the last moment the world could have changed unobserved
    (the fact arm) — and ``valid ⇔ stamp >= min(now - ttl,
    last_foreign_human_seen_ms)`` expresses both as one threshold.
    With no foreign human ever seen the floor is 0 and knowledge is
    permanent; with one present the floor is the plain TTL and
    behavior is exactly the pre-2026-09-02 clock; after one leaves,
    stamps written since their departure never age while older ones
    age out once ([[flag-triage-20260902]] rows 3-5).

    Args:
        now_ms: Current timestamp.
        ttl_ms: The question's staleness window while humans are
            about.
        last_foreign_human_seen_ms: Newest observation of any human
            tank that is not this bot and not a fleet sibling; 0 when
            none was ever seen.

    Returns:
        The instant at or after which a stamp still answers the
        question.
    """
    return min(now_ms - ttl_ms, last_foreign_human_seen_ms)


__all__ = [
    "FERRY_LOOK_TTL_MS",
    "FORAGE_COVERAGE_TTL_MS",
    "HARVEST_MEMORY_TTL_MS",
    "settled_knowledge_floor_ms",
    "ttl_floor_ms",
]
