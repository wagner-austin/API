"""Radar-spend economics: when a scan is worth its cost.

The shared reveal-floor rules ([[flag-triage-20260729]] s9-2/4/5)
consulted by the forage radar, the landing radar, the desync rescan
and the quad sweep — one price list, so no consumer can drift cheap.
Split from ``context`` 2026-09-03 at the file-size ceiling; the
predicates read the ``DecideCtx`` fields they always did.
"""

from __future__ import annotations

from tankpit_bot.bot.ai.context import DecideCtx
from tankpit_bot.state.scan_coverage import (
    free_radar_new_coverage,
    viewport_uncovered_count,
)
from tankpit_bot.state.viewport_geometry import viewport_visible_bounds

RADAR_SPEND_REVEAL_FLOOR_TILES = 32
"""Minimum uncovered viewport tiles that justify spending an extra radar.

The single radar-economics rule every discretionary radar site
consults ([[flag-triage-20260729]] s9-2/4/5, user 2026-07-30: "im
worried ... the viewport freshness handling is not properly wired to
the collecting system"). With extras stocked every scan CONSUMES an
item, and session 9 spent them on slivers: a displaced-landing rescan
of a fully-scanned viewport, a desync rescan of ground radared
seconds earlier, and a forage radar for a handful of tiles the tank
then hopped away from. 32 tiles is an eighth of the 256-tile
viewport — below that the reveal does not buy an item; the free
built-in radar (extras=0) stays gated only on "any uncovered tile"
because it costs nothing but the tick.
"""

RADAR_RESERVE_EXTRAS = 1
"""Extra-radar count treated as the reserve (user ruling 2026-07-31:
"if the bot runs out of radar ever ... its like dead in the water cuz
it takes so long to restock via free radar"). At or below this count
the spend bar escalates to :data:`RADAR_RESERVE_REVEAL_FLOOR_TILES`.
This is spend-gating inside the existing economics rule, NOT the
extras-toggle rationing rejected 2026-06-12 ([[radar-mechanics]]) --
the extras slot stays enabled and any scan that does fire uses the
extra."""

RADAR_RESERVE_REVEAL_FLOOR_TILES = 128
"""Uncovered-tile bar for spending the LAST extra radar: half the
256-tile viewport. The final paid sweep goes only to a near-full-value
reveal, never dribbles away on a sliver -- once it is gone, discovery
collapses to the built-in radius-2 scan and restock stalls
([[radar-mechanics]] "Death spiral at 0 extras")."""


def radar_spend_worthwhile(ctx: DecideCtx) -> bool:
    """Return True when a radar dispatch is worth its cost right now.

    Args:
        ctx: Decision context (coverage map + inventory).

    Returns:
        With extras above the reserve: True when the current viewport
        has at least :data:`RADAR_SPEND_REVEAL_FLOOR_TILES` uncovered
        tiles. At the reserve (the last extra): True only from
        :data:`RADAR_RESERVE_REVEAL_FLOOR_TILES` uncovered tiles.
        Without extras: True only when the built-in radar's own
        rank-scaled footprint around the tank holds an uncovered tile
        -- the press is free, but it reveals ``2 + rank // 3`` tiles
        around SELF, not the viewport. The old whole-viewport gate
        press-looped at zero extras: far uncovered corners kept
        answering "scan", the 5x5 revealed nothing new, and the
        scan-walk-scan doctrine (user ruling 2026-08-14) never got
        the tick (operator observation, live watch 2026-08-28).
    """
    left, top, right, bottom = viewport_visible_bounds(ctx.world["viewport"])
    extras = ctx.inventory["extra_radars"]["count"]
    if extras == 0:
        return (
            free_radar_new_coverage(
                ctx.world["scanned_tiles"],
                ctx.self_state["x"],
                ctx.self_state["y"],
                left,
                top,
                right,
                bottom,
                ctx.forage_floor_ms,
                ctx.self_state["rank"],
            )
            > 0
        )
    uncovered = viewport_uncovered_count(
        ctx.world["scanned_tiles"],
        left,
        top,
        right,
        bottom,
        ctx.forage_floor_ms,
    )
    if extras > RADAR_RESERVE_EXTRAS:
        return uncovered >= RADAR_SPEND_REVEAL_FLOOR_TILES
    return uncovered >= RADAR_RESERVE_REVEAL_FLOOR_TILES


__all__ = [
    "RADAR_RESERVE_EXTRAS",
    "RADAR_RESERVE_REVEAL_FLOOR_TILES",
    "RADAR_SPEND_REVEAL_FLOOR_TILES",
    "radar_spend_worthwhile",
]
