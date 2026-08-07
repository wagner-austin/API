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

User-derived 2026-08-06, built entirely on measured laws. The one
probe that stood before code was resolved same day from the capture
archive (§ Shift-framed pickup: resolved). IMPLEMENTED 2026-08-07
(§ As built) and sim-verified end-to-end against the fake server's
wire-pinned window laws.

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

## Shift-framed pickup: resolved (archive-mined 2026-08-06)

The question was whether a pickup dispatched into a SHIFT-framed
window (not a teleport-centered one) transfers normally — every BOT
pickup to date fired inside a teleport frame. Answered from recorded
HUMAN play, no live probe needed: two archived sessions carry twelve
accepted in-shifted-window actions across seven scope shifts.

`runs/sniff/sniff-20260710-202821.capture_session.json` (the 421.8 s
human session behind the ANCHOR law): five shift→pickup windows, e.g.
shift dir=2 → `0x5A (144,157)` → `PICKUP_EQUIP (148,165)` → `0x47`
walk echo + duplicate `0x43` pickup records 610 ms later; likewise
windows (129,157), (160,159), (96,201), (110,19) — every accepted
target inside the shifted 16×16, standard choreography every time.
`runs/sniff/ghost_observe.capture_session.json` corroborates: shift →
`0x5A (142,195)` → equipment pickups at (149,207) and (155,200)
accepted, plus a 438-volume fuel container taken by move-onto at
(152,204). The only rejections in either corpus were `0x52` err=1
(no path — the retry succeeded) and code 7 (inventory full) — none
window-bound. **The shifted window IS the acceptance window; the
harvest loop can be built on it.** Miner: scratchpad
`mine_shift_pickup.py` (sent-frame classify `21 03 5a dir` scope /
`21 04 6a` pickup, 0x2E envelope unwrap on the receive side).

## Where it lands in the bot

Replaces the extras-stocked branch of forage (the branch whose
veto-then-walk deadlock produced the 2026-08-06 one-tile crawl,
fixed same day to yield nothing). Quad sweep becomes what
extras-stocked foraging IS; the free-radar reposition walk remains
the extras-empty strategy.

## As built (2026-08-07, `bot/ai/quad_sweep.py`)

Two COLLECT-cascade branches, both gated on fuel ABOVE the low
threshold (recon and framing are economy moves — at the break the
desperation ladder owns every tick):

- **`plan_quad_sweep`** sits between lock continuation and the
  pickups (atomicity: pickups wait the ~8 ticks). Stateless quadrant
  math — the four windows derive from the tank tile by the anchor
  law each tick — plus ONE latch pair, `sweep_anchor_x/y`: a sweep
  STARTS (latching the anchor) only when the 31x31 block holds >=480
  uncovered tiles (~half), and CONTINUES only while the tank stands
  exactly on its anchor, so any movement aborts the remainder and
  harvest walking can never re-trigger a sweep on the dragged block.
  Each tick: radar the framed pending quadrant; else radar the
  CURRENT window if it still clears the shared spend floors
  (self-correcting whatever the scope answered); else steer with the
  next quadrant's shift. Spend economics are the shared radar floors
  (32 tiles, 128 at the last-extra reserve).
- **`plan_block_harvest_leg`** sits after mine clearance, BEFORE the
  larder (a free shift beats any teleport while block stock
  remains). Greedy nearest-by-Manhattan over pursuable out-of-window
  containers within Chebyshev 31: shift-before-walk per leg — frame
  the target when the anchored shift changes the window, else hand
  the movement layer a plain walk toward it (edge-approach walk,
  teleport only when no walkable edge serves). Once framed, the
  ordinary in-window pickup branches take the container next tick.
  Kind gates: equipment refused at full inventory, fuel at cap and
  by the worth-the-walk rate; block-reachability via the bounded
  collection-path check (skipped when no terrain view exists — the
  assume-reachable stance the pickup search takes).
  The planned "end the loop nearest the next block" ordering is NOT
  in this cut — greedy nearest is the v1 approximation; the exit-hop
  bias is a later refinement alongside the anchor-layout knob.

**Landing-latch interaction** (found in the ferry sim soak): the
scan-on-landing gate keyed "origin changed" as "landed", so every
deliberate pan drew next tick's unconditional landing radar — taxing
the free ferry scout and mislabeling sweep scans. Scope-shift
decisions now pre-latch `last_landing_scan_viewport` with the
anchor-law origin (`latch_scope_shift_landing`, applied by the
arbitrator): a pan is a deliberate look, not a landing, and the
pan-er decides whether a radar follows.

**Sim verification**: the seam soak runs the doctrine cycle
end-to-end — atomic NW/NE/SE/SW tiling confirmed against the fake
server's wire-pinned window laws, in-window pickups, a harvest
framing shift, and equipment collection that ends extras ABOVE the
sweep's spend. The ferry scenario showed the sweep's pans SUBSUME
the dedicated scope scout when extras are stocked (the ferry arrives
in a quadrant pan's 0x5A patch and the larder boards it directly);
the scout remains the extras-empty fallback.
