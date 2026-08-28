"""Human rendering for the per-run digest table.

Split from :mod:`tankpit_bot.diagnostics.run_digest` (2026-08-28, at
the file-size bar) when the wasted-tick census landed there.
"""

from __future__ import annotations

from tankpit_bot.diagnostics.run_digest_types import RunDigestDict


def render_run_digest(digest: RunDigestDict) -> str:
    """Render the digest as the aligned human table.

    Args:
        digest: Computed digest.

    Returns:
        Multi-line table text.
    """
    exit_line = (
        f"CLEAN {digest['exit_reason']}"
        if digest["clean_exit"]
        else "CRASHED (no teardown scorecard)"
    )
    lines = [
        "=== RUN DIGEST ===",
        f"source     {digest['source']}",
        f"window     {digest['started_at']} .. {digest['ended_at']}"
        f"  ({digest['duration_s'] // 60}m{digest['duration_s'] % 60:02d}s)",
        f"exit       {exit_line}",
        f"room       {digest['room_id']}   self tank id {digest['self_tank_id']}",
        f"combat     kills={digest['kills']} deaths={digest['deaths']} shots={digest['shots']}",
        *([f"rank       {'; '.join(digest['rank_changes'])}"] if digest["rank_changes"] else []),
        f"movement   teleports={digest['teleports']} displaced={digest['displacements']}"
        f" pickups={digest['pickups']}",
        f"activity   stalls={digest['liveness_stalls']}"
        f" superseded_plans={digest['superseded_undispatched']}"
        f" re_aims={digest['superseded_dispatched']}"
        f" max_wire_gap={digest['max_wire_gap_s']}s"
        f" gaps_over_30s={digest['wire_gaps_over_30s']}",
    ]
    if digest["rank_number"] != -1:
        lines.append(
            f"account    rank={digest['rank_name']} ({digest['rank_number']})"
            f" promo={digest['promotion_points']}"
        )
    if digest["inventory_first"]:
        lines.append(
            f"inventory  first={digest['inventory_first']} last={digest['inventory_last']}"
            " (armor,dual,missile,homing,radar)"
        )
    for row in digest["displacement_top"]:
        lines.append(f"displaced  ({row['requested_x']},{row['requested_y']}) x{row['count']}")
    for shot in digest["clearance_shots"]:
        outcome = "converted" if shot["pickup_followed"] else "no pickup followed"
        lines.append(f"clearance  {shot['timestamp']} ({shot['x']},{shot['y']}) {outcome}")
    for reason, count in sorted(digest["releases_by_reason"].items()):
        lines.append(f"release    {reason} x{count}")
    lines.append("timeline   min: kills/shots/teleports/pickups")
    for bucket in digest["timeline"]:
        lines.append(
            f"           {bucket['minute']:>4}: {bucket['kills']}/{bucket['shots']}"
            f"/{bucket['teleports']}/{bucket['pickups']}"
        )
    return "\n".join(lines)


__all__ = [
    "render_run_digest",
]
