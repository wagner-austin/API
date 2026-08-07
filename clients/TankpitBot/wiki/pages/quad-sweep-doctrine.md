---
title: "Quad-Sweep Doctrine — stationary recon, loop harvest"
tags: [doctrine, radar, viewport, collect]
related:
  - "[[viewport-shift-protocol]]"
  - "[[radar-mechanics]]"
  - "[[fuel-system]]"
  - "[[bot-behavior-contract]]"
source_paths:
  - "src/tankpit_bot/bot/ai/forage.py"
fact_checked: "2026-08-06"
confidence: high
hubs: [architecture]
---

# Quad-Sweep Doctrine — stationary recon, loop harvest

User-derived 2026-08-06, built entirely on measured laws. Design is
RECORDED, not yet implemented; one probe stands before code (§ Open
probe).

## The laws it stands on

- **The viewport is the radar** (user-confirmed): an extra radar
  scans the current 0x5A window, wherever the scope parked it.
- **ANCHOR law** ([[viewport-shift-protocol]]): every scope shift
  anchors to the TANK's current position — a diagonal shift pins the
  tank to the opposite corner; repeated pans never scroll; total
  reachable view from one standing position is the 31×31 centered on
  the tank.
- **Three shift triggers only**: explicit shift (incl. center),
  autoscroll edge-crossing, teleport landing. Walking NEVER moves
  the window.
- **Window-bound acceptance**: move and pickup targets must sit in
  the CURRENT stored window; outside answers 0x52.
- **Costs** (measured): walking ~1 fuel and ~0.22 s per Manhattan
  tile (diagonals two steps); teleport `floor(6 × euclid)` fuel plus
  ~2 ticks of overhead; scope shifts free, confirm 50 ms–1.5 s.

## The cycle

1. **Atomic quad sweep.** From the anchor, stationary: shift NW →
   extra radar, NE → radar, SE → radar, SW → radar. Four windows
   tile the 31×31 around the tank (~961 tiles, ~6% overlap on the
   shared cross-strips). ATOMIC means zero tank movement between the
   four scans — the ANCHOR law re-anchors every shift to the tank,
   so moving mid-sweep slides later windows off the grid, buying
   overlap and phantom gaps for the same four extras. Any
   interruption (threat, pickup) either waits the ~4 ticks or aborts
   the whole sweep for a later re-anchor; a half-sweep has no value
   worth preserving.
2. **Harvest loop.** Plan a walkable loop over the revealed clusters
   — ordered by composed-view WALKABLE distance, not compass or
   straight line (water-split clusters reorder the loop; items
   unreachable from every standing position are excluded from the
   plan, shoot-to-clear or abandoned). End the loop at the cluster
   nearest the next block. There is no "return to center": the
   anchor is just where the tank stood, and holds no status after
   the sweep.
3. **Per leg**: shift toward the destination FIRST (the walk target
   must be in-window), walk up to the framed limit, shift again as
   needed; on arrival, shift/recenter to frame the cluster, dispatch
   pickups. Teleport only when a cluster is farther than ~6× its
   walk cost justifies, or on fuel emergency.
4. **Exit hop**: map-open + teleport from the loop's end to the next
   block anchor (~31-tile stride ≈ 186 fuel — the cycle's single
   biggest cost, which is why the block is squeezed dry first). The
   landing recenters the window for free.

## Block layout knob

- **Grid-disciplined** next anchor (original anchor ± 31): map tiles
  cleanly, maximum unique tiles per extra.
- **Opportunistic** (anchor wherever the exit lands): shorter hops,
  some inter-block overlap.
Lean grid-disciplined while extras are plentiful, opportunistic when
low. User's call to tune.

## Open probe (blocks implementation)

A pickup has never been dispatched into a SHIFT-framed window (every
live pickup to date fired inside a teleport-centered frame). The
window-bound acceptance law says in-window is in-window, but one
live probe must confirm a pickup from a shifted frame transfers
normally before the harvest loop is built on it.

## Where it lands in the bot

Replaces the extras-stocked branch of forage (the branch whose
veto-then-walk deadlock produced the 2026-08-06 one-tile crawl,
fixed same day to yield nothing). Quad sweep becomes what
extras-stocked foraging IS; the free-radar reposition walk remains
the extras-empty strategy.
