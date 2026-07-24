---
title: Game Economy (Fuel, Damage, Costs)
tags: [game, combat, fuel, economy]
related:
  - "[[shoot-event-format]]"
  - "[[mine-mechanics]]"
  - "[[deactivation-format]]"
  - "[[bot-behavior-contract]]"
  - "[[fuel-system]]"
source_paths:
  - "runs/sniff/sniff-20260620-150155.capture_session.json"
  - "runs/sniff/sniff-20260620-155103.capture_session.json"
  - "runs/sniff/sniff-20260620-173727.capture_session.json"
  - "tpclient.js"
source_git_blobs:
  "tpclient.js": "cb253fe55b10221291a35382d2f4e2efcd02f2ff"
fact_checked: "2026-07-24"
confidence: high
verified: 2026-07-06 (capacity formula cross-checked client gauge math vs user deposits at ranks 1/3/6/7)
hubs: [combat]
---

# Game Economy

Empirically measured fuel costs, damage values, and capacity limits — derived from three controlled captures on 2026-06-20 where the user reported every action and I matched it against the wire bytes (0x2E TankStatusSync fuel-field deltas).[^1]

## Tank capacity

**Fuel capacity = 1000 + 100 × rank** (2026-07-06). Not sent on the wire — the client derives it from rank in the fuel-gauge draw (`Gc` in tpclient.js: fill width `7·fuel/100` px against a capacity region of `7·(10+rank)` px, equal iff `fuel = 100·(10+rank)`). Rank IS on the wire (`self_state["rank"]`), so the bot can compute capacity at tick 1 with no probe pickup.[^2]

| Rank | Capacity | Verified |
|---|---|---|
| recruit (0) | 1000 | formula only |
| **private (1)** | **1100** | wire 0x52 code-5 tank-full at exactly 1100 (2026-07-06 run); 7+ pickup caps at 1100 (2026-06-20); max deposit 1000 = 1100−100 |
| corporal (2) | 1200 | formula only |
| **sergeant (3)** | **1300** | max deposit 1200 = 1300−100 (user, 2026-07-06) |
| lieutenant (4) | 1400 | formula only |
| captain (5) | 1500 | formula only |
| **major (6)** | **1600** | max deposit 1500 = 1600−100 (user, 2026-07-06) |
| **colonel (7)** | **1700** | max deposit 1598 = 1700−100−~2 walk (user, 2026-07-06) |
| general (8) | 1800 | formula only |

The cap is enforced by the server on pickup — if the picker's tank can only hold `N` more fuel before hitting capacity, only `N` is transferred from the container and the rest remains. This is why `container_pickup.remaining_volume` is often non-zero.[^1]

The earlier "Max fuel cap = 1100" entry on this page was correct but rank-specific: all 2026-06-20 measurements were taken on a private. A 2026-06-11 learned-watermark of 2010 exceeds even a general's 1800 and was a polluted scrape read (the same run tank-full'd at 1100), not evidence against the formula.[^2]

## Cost per player action

| Action | Fuel cost | Notes |
|---|---|---|
| Walk one tile | **1** | Verified by 0x47 Movement Manhattan distance vs 0x2E fuel delta (4-tile walks → −4, 5-tile → −5, 8-tile → −8). Re-confirmed 2026-07-20: six clean SelfMovement segments, exact at 1/tile (7→7, 14→14, 5→5, 3→3, 2→2, 1→1). Billed IN FULL at the echo tick — server movement is instant, see [[walk-mechanics]] (2026-07-21: 200/200 archive episodes) |
| Single shot (`weapon=0`) | **6** | Systematic isolation 2026-07-20: 62 clean windows exactly −6 across the 204-capture archive (window = consecutive absolute fuel readings containing only our 0x53 echoes, no movement/radar/pickups/enemy fire). Consumes NO ammo |
| Dual shot (`weapon=1`) | **10** | Same isolation: **589 clean windows exactly −10**. Consumes 1 dual per LANDED shot (0x49 count snapshots: 49 windows of one dual fired → dual −1) |
| Homing shot (`weapon=3`) | **10** | Same isolation: 398 clean windows exactly −10, plus 124 at −5 — the homing debit sometimes lands in two −5 steps across sync boundaries; total per shot is 10. Consumes 1 homing per LANDED shot |
| Missile (`weapon=2`) | **10** | Manual capture sniff-20260720-213208 (user fired 10 missiles at enemies behind rock, stationary): 6 clean single-shot windows exactly −10 each; 0x49 counts 25→15 = 10 missiles consumed for 10 shots. Also live-confirms the obstruction trigger rule ([[weapon-selection]]) |
| Mine placement (per command) | **10 flat per press** | Manual capture sniff-20260720-214329 (place → walk 3 → place cadence): 8 consecutive presses each exactly −10, independent of how many of the 3×3 mines landed (terrain-blocked tiles and enemy-mine detonations don't change the price). The old "6 mines → −10" sample was one press with 3 blocked tiles — the cost was never per-mine |
| Radar scan | **10** | Archive-isolated 2026-07-24 via sent-command-keyed windows (exactly one radar dispatch, no other sent commands, no shots/pickups/detonations, 3 s charge-latency guard): **1,293/1,311 clean windows exactly −10** across the archive[^6] |
| Block pickup / drop | **0 (free)** | Manual captures 2026-07-20: stationary same-tile pickup/re-drop pairs produced zero fuel delta; towing movement costs the normal 1/tile ([[movable-blocks]]) |
| Teleport | **floor(6 × euclidean distance)** — measured from start to the **actual landing tile** | Systematically validated 2026-07-20: every `teleport(x,y)` dispatch in every run was paired with its wire fuel delta (pre-hop `Self:` fix → post-hop `Fuel: A -> B` line, contaminated windows excluded). Post-2026-06-24 era (after the fuel double-count fix): **248/248 pairs exact**, costs 6–654. All-era: 2,335/2,538 exact; the 203 residuals are all pre-fix runs with broken fuel tracking. When the server drifts the landing off the requested target, the charge matches distance to the LANDING, not the target (624 drift hops confirm this) — planner estimates on the target can be off by a few fuel on drifted hops |

## Damage taken

| Source | Fuel loss to victim | Notes |
|---|---|---|
| Single shot HIT (taken from enemy `weapon=0`) | **45** | Verified by 3 Yuppler hits in the multi-tank PvP capture, all 3 matched −45 each |
| Dual shot HIT (taken from enemy `weapon=1`) | **90** | Verified by 3 Yuppler dual hits in the same capture, all 3 matched −90 each; re-confirmed 2026-07-21 via armor (2 shields consumed per dual = 90/45) |
| Missile HIT (taken from enemy `weapon=2`) | **45** | Measured 2026-07-21: 5 isolated hits, each exactly −45 at the echo instant; container-drain cross-check 235 = 5×45 + radar 10 |
| Homing HIT (taken from enemy `weapon=3`) | **45** | Same session: 5 isolated hits, each exactly −45. Dual is the game's ONLY double-damage hit |
| Walking into an enemy mine | **45** | Verified at t+373.35s of the multi-pickup capture: 1-tile walk into mine at (92, 185) → −45 fuel |
| Mine cascade damage (your own mine detonating you) | not observed | Bot tests where player blew up their own mines didn't show a fuel delta on the placer |

Damage manifests as a fuel decrement — the game's "health" is essentially "fuel reserve" for combat purposes. When you take damage your fuel drops; when fuel hits zero the tank is deactivated. This is now a corpus-fitted law: the rendered damage tier IS the fuel quartile ([[deactivation-format]] §SOLVED, 19,658/19,658).

**Armor shields (measured 2026-07-21, 16 incoming hits, fuel untouched):** with shields enabled, damage is FULLY absorbed and shields are consumed at damage/45 per hit — singles/missiles/homings eat 1 shield, duals eat 2. Victim-side timing note from the same session: incoming-hit fuel debits land the SAME INSTANT as the 0x53 echo (no tick lag, unlike shooter-side charges).[^3]

## Container pickup mechanics

See [[fuel-system]] and the ``ContainerPickupDict`` decoder doc for the full wire shape. Key economy points:

- A radar scan reveals each container's **declared volume** (e.g. 300, 400, 1142).
- Walking over the container picks up **as much as the picker's tank can hold** (up to the 1100 cap).
- The wire reports the container's **remaining_volume** after pickup. Multiple pickers can drain one container in sequence.
- `remaining_volume = 0` means the container is exhausted (or it was equipment with no fuel attribute). Equipment pickups also fire a paired `0x67 EquipmentGain` carrying the items.
- **Multi-record bodies**: a single 0x43 message can carry 1, 2, 3 (or more) pickup records, fired when a tank's movement causes multiple container tiles to update in one server tick. Corpus distribution (156 sessions, 2026-06-20): 2653 single-record, 80 two-record, 2 three-record.
- **Duplicate broadcasts**: the server broadcasts each pickup twice within ~200 ms (one to the picker, one to the world view) -- empirical 43.9% duplicate rate. The dispatcher de-duplicates by `(x, y, remaining_volume)` signature within a 500 ms window so metrics and logs see each pickup once; the world-state mutation is idempotent regardless.

Verified breakdown from the multi-pickup capture:[^1]

| Container known volume | Picker tank before | Picker took | Container remaining (wire) |
|---:|---:|---:|---:|
| 1142 | 999 | 171 | 971 ✓ |
| 300 | 1083 | 17 | 283 ✓ |
| 400 | 1097 | 3 | 397 ✓ |
| 100 | 1096 | 4 | 96 ✓ |
| 571 (full bucket) | 456 | 571 | 0 ✓ |

Every row matches `remaining = declared − taken` exactly.[^1]

## Fuel deposit (your own tank donating fuel)

- Triggered by a deposit command: client code `'D'` (0x44), 6 bytes — x, y, **u16 LE amount** (tpclient.js `Wb` class, 2026-07-06). The amount accumulates during the mouse long-press ("DEPOSIT FUEL: N" HUD label) and is clamped client-side to current fuel.
- Client-gated by the fuel>100 check (`ce()`): a tank at ≤100 fuel cannot initiate a deposit.
- **Deposit floor = 100, server-enforced**: a max deposit always leaves exactly 100 fuel in the tank. Verified at four ranks 2026-07-06 — private 1000 (=1100−100), sergeant 1200 (=1300−100), major 1500 (=1600−100), colonel 1598 (=1700−100−~2 walked).
- Server places the fuel container on a tile **adjacent to the depositing tank's position** (verified: Yuppler at (131, 124) deposited → container at (132, 124)).
- The wire path: depositing player gets a 0x64 FuelDeposit; observers see the container via the next 0x4F RadarScan / viewport refresh.
- Containers are **invisible by default** to non-owning players. They appear via radar reveal.
- Bot use case: a tank at capacity can bank surplus fuel next to a defended position and reclaim it later.

## Container respawn dynamics (archive-mined 2026-07-22)

The world replenishes itself. Mined from 212 sessions holding 2+
0x4C map snapshots (the 0x4C fuel-dot atlas is GLOBAL, so
within-session diffs are true world dynamics — spawns and
consumption by anyone):[^4]

- **Steady-state population**: 569–656 fuel dots on the map
  (mean 619); per-session spawns ≈ consumption.
- **Spawn rate ~1.00 dots/minute below equilibrium** (605 spawns
  over 605.7 observed minutes). A 12-minute idle session at high
  population spawned ZERO — the rate is population-seeking, not a
  flat timer.
- **Spawns never reuse a consumed position**: 0/605 exact reuses;
  587/605 appeared at entirely fresh map locations (18 merely within
  2 tiles of a past consumption).
- **No wire message announces a spawn** — the client discovers new
  dots on the next map open or radar reveal.[^4]

Equipment containers are invisible to the 0x4C atlas, so their
respawn dynamics remain unmeasured. The sim ([[physics-module-roadmap]])
implements the fuel law deterministically (seeded population as the
target, one spawn per minute-beat at a tick-derived fresh passable
tile, constant 300 volume as a documented assumption) and mirrors it
for equipment on the offset beat as an assumption.

**First equipment measurements (radar-reveal mining, 2026-07-24):**
since equipment never appears on the map, the mining works from
partial visibility — every 0x4F radar entry and 0x5A patch tile is
tracked per-tile, and a tile observed empty then later observed
holding equipment is one witnessed spawn. Archive-wide: **45
witnessed spawns over 9,040 empty-tile-minutes of re-scan exposure**
(≈ 0.5% chance per empty tile per minute in the active play area);
5,440 first-reveals ≈ 22 distinct equipment tiles seen per session.
Consumption attribution is weak by construction (the re-scan lags
the pickup, so only 21/141 equipment→empty transitions land near an
own 0x67 within 5 s — a timing artifact, not evidence of another
consumer; bots demonstrably never collect, [[enemy-bot-behavior]]).
Placement distribution and true map-wide population remain open —
they need wider radar coverage than bot sessions produce.[^4]

## What's still open

Equipment-container respawn dynamics and spawn volumes (the 0x4C
atlas carries neither). Every player-action fuel cost is closed: walk=1/tile, single=6 (free ammo), dual/missile/homing=10 (+1 round per landed shot), radar=10, mine press=10 flat, teleport=floor(6×euclid to actual landing), deposit=free (clamped to fuel−100). Dual/homing/single came from the 204-capture archive; missile (sniff-20260720-213208) and mine press (sniff-20260720-214329) from dedicated manual captures the same day. The former mystery "paired −45/−10 per combat firing tick" decomposes as −10 = our firing cost + −45 = the incoming enemy single hit landing the same tick. The action-cost design is now visible: everything is 10 except walking (1/tile) and the free single (6).[^5]

## Machine-checked claims

This block binds each fact above to its code symbol in
`tankpit_bot.physics` ([[physics-module-roadmap]] Phase 1). The
`physics_claims` guard stage of `make check` imports every `code`
address and verifies it — constants by equality, formulas on the
probe grid — and fails the gate if any public physics symbol lacks a
claim here. Edit the fact → edit the claim → the gate forces the code
to follow (and vice versa).

```json claims
{
  "claims": [
    {
      "id": "walk-cost",
      "code": "tankpit_bot.physics.costs:WALK_COST_PER_TILE",
      "value": 1
    },
    {
      "id": "single-shot-cost",
      "code": "tankpit_bot.physics.costs:SINGLE_SHOT_COST",
      "value": 6
    },
    {
      "id": "dual-shot-cost",
      "code": "tankpit_bot.physics.costs:DUAL_SHOT_COST",
      "value": 10
    },
    {
      "id": "missile-shot-cost",
      "code": "tankpit_bot.physics.costs:MISSILE_SHOT_COST",
      "value": 10
    },
    {
      "id": "homing-shot-cost",
      "code": "tankpit_bot.physics.costs:HOMING_SHOT_COST",
      "value": 10
    },
    {
      "id": "radar-cost",
      "code": "tankpit_bot.physics.costs:RADAR_COST",
      "value": 10
    },
    {
      "id": "mine-press-cost",
      "code": "tankpit_bot.physics.costs:MINE_PRESS_COST",
      "value": 10
    },
    {
      "id": "block-op-cost",
      "code": "tankpit_bot.physics.costs:BLOCK_OP_COST",
      "value": 0
    },
    {
      "id": "teleport-cost",
      "code": "tankpit_bot.physics.costs:teleport_cost",
      "formula": "floor(6 * sqrt(dx^2 + dy^2)), charged to the actual landing tile",
      "probes": [
        {"args": [0, 0, 1, 0], "expect": 6},
        {"args": [0, 0, 3, 4], "expect": 30},
        {"args": [0, 0, 1, 1], "expect": 8},
        {"args": [10, 20, 10, 20], "expect": 0},
        {"args": [0, 0, 100, 100], "expect": 848}
      ]
    },
    {
      "id": "single-hit-victim-cost",
      "code": "tankpit_bot.physics.damage:SINGLE_HIT_VICTIM_COST",
      "value": 45
    },
    {
      "id": "dual-hit-victim-cost",
      "code": "tankpit_bot.physics.damage:DUAL_HIT_VICTIM_COST",
      "value": 90
    },
    {
      "id": "mine-detonation-cost",
      "code": "tankpit_bot.physics.damage:MINE_DETONATION_COST",
      "value": 45
    },
    {
      "id": "missile-hit-victim-cost",
      "code": "tankpit_bot.physics.damage:MISSILE_HIT_VICTIM_COST",
      "value": 45
    },
    {
      "id": "homing-hit-victim-cost",
      "code": "tankpit_bot.physics.damage:HOMING_HIT_VICTIM_COST",
      "value": 45
    },
    {
      "id": "armor-absorb-per-shield",
      "code": "tankpit_bot.physics.damage:ARMOR_ABSORB_PER_SHIELD",
      "value": 45
    },
    {
      "id": "fuel-capacity",
      "code": "tankpit_bot.physics.capacity:fuel_capacity",
      "formula": "1000 + 100 * rank",
      "probes": [
        {"args": [0], "expect": 1000},
        {"args": [1], "expect": 1100},
        {"args": [7], "expect": 1700},
        {"args": [8], "expect": 1800}
      ]
    },
    {
      "id": "inventory-capacity",
      "code": "tankpit_bot.physics.capacity:inventory_capacity",
      "formula": "20 + 5 * rank per slot",
      "probes": [
        {"args": [0], "expect": 20},
        {"args": [1], "expect": 25},
        {"args": [8], "expect": 60}
      ]
    },
    {
      "id": "deposit-floor",
      "code": "tankpit_bot.physics.capacity:DEPOSIT_FLOOR",
      "value": 100
    }
  ]
}
```

## How this was discovered

Three sequential captures on 2026-06-20:[^1]

1. **PvP capture** (Artax vs Yuppler combat) gave us: enemy-shot damage values, weapon byte semantics, mine cascade, mine-on-mine destruction (see [[mine-mechanics]]).
2. **Multi-pickup capture** (Yuppler deposits 300, 400, 100; Artax picks up each in isolation) gave us: container_pickup.remaining_volume semantic, max fuel cap, walking cost, radar cost.
3. **Ghost-observation capture** (Artax killed orange-5 twice) gave us: the corrected `0x58 ≠ death` finding that drove the [[bot-behavior-contract]] 2-state liveness rewrite.

Each value above is matched 1:1 between a user-declared action and a measurable wire delta. No inferences without direct evidence.[^1]

[^1]: the three frontmatter-pinned 2026-06-20 captures on disk: `runs/sniff/sniff-20260620-150155.capture_session.json` (PvP), `sniff-20260620-155103` (multi-pickup), `sniff-20260620-173727` (ghost-observation); every cost in the tables is also machine-checked against `tankpit_bot.physics` via the claim block (the `physics_claims` stage of `make check`) and re-derived from the archive by `make audit`.
[^2]: client gauge math: `Gc` in `tpclient.js` (blob-pinned in frontmatter); rank-table verifications 2026-07-06 (user deposits at ranks 1/3/6/7, recorded per-row in the table above and in the frontmatter `verified:` field); formula machine-checked by the `fuel-capacity` claim below.
[^3]: wiki-log entries "[2026-07-21] measurement | Victim costs closed (missile=45, homing=45), armor cracked, and the pathfinder is DETERMINISTIC" and "[2026-07-21] refactor | Victim-cost session folded through the whole pipeline — 11/11 claims, armor modeled live"; the shield-absorb constant is machine-checked by the `armor-absorb-per-shield` claim below.
[^4]: wiki-log entry "[2026-07-22] discovery+feature | The world replenishes and players return — spawn dynamics cracked from 0x4C atlas diffs"; the numbers are re-derivable by re-running the atlas-diff mining over the `runs/` corpus (212 sessions with 2+ 0x4C snapshots).
[^5]: dedicated manual captures on disk: `runs/sniff/sniff-20260720-213208.capture_session.json` (missiles) and `sniff-20260720-214329` (mine presses); the full cost set is machine-checked in the claim block below.
[^6]: radar isolation sweep 2026-07-24: `analysis_scripts/mine_bot_policy.py` (sent-command-keyed fuel windows) over the full archive; results snapshot `analysis_scripts/bot_policy_sweep_2026-07-24.json`; wiki-log entry "[2026-07-24] mining | Radar cost isolated, self-sync drift is activity-correlated". STANDING instrument since the same day: `validate_radar_cost` in `src/tankpit_bot/validate/archive.py` re-derives this claim on every `make audit` (`radar-cost` row; first run replicated the sweep digit-for-digit — 1,311 samples, 1,293 exact, PASS).
