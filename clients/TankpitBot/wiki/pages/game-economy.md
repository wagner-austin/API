---
title: Game Economy (Fuel, Damage, Costs)
tags: [game, combat, fuel, economy]
related: [[shoot-event-format]], [[mine-mechanics]], [[deactivation-format]], [[bot-behavior-contract]], [[fuel-system]]
sources: [runs/sniff/sniff-20260620-150155 (multi-tank PvP), runs/sniff/sniff-20260620-155103 (annotated multi-pickup), runs/sniff/sniff-20260620-173727 (ghost-observation 5 kill cycles), user narrative cross-references 2026-06-20, tpclient.js Gc/Wb/ce functions + user deposit measurements at 4 ranks 2026-07-06]
fact_checked: 2026-07-06
confidence: high
verified: 2026-07-06 (capacity formula cross-checked client gauge math vs user deposits at ranks 1/3/6/7)
---

# Game Economy

Empirically measured fuel costs, damage values, and capacity limits — derived from three controlled captures on 2026-06-20 where the user reported every action and I matched it against the wire bytes (0x2E TankStatusSync fuel-field deltas).

## Tank capacity

**Fuel capacity = 1000 + 100 × rank** (2026-07-06). Not sent on the wire — the client derives it from rank in the fuel-gauge draw (`Gc` in tpclient.js: fill width `7·fuel/100` px against a capacity region of `7·(10+rank)` px, equal iff `fuel = 100·(10+rank)`). Rank IS on the wire (`self_state["rank"]`), so the bot can compute capacity at tick 1 with no probe pickup.

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

The cap is enforced by the server on pickup — if the picker's tank can only hold `N` more fuel before hitting capacity, only `N` is transferred from the container and the rest remains. This is why `container_pickup.remaining_volume` is often non-zero.

The earlier "Max fuel cap = 1100" entry on this page was correct but rank-specific: all 2026-06-20 measurements were taken on a private. A 2026-06-11 learned-watermark of 2010 exceeds even a general's 1800 and was a polluted scrape read (the same run tank-full'd at 1100), not evidence against the formula.

## Cost per player action

| Action | Fuel cost | Notes |
|---|---|---|
| Walk one tile | **1** | Verified by 0x47 Movement Manhattan distance vs 0x2E fuel delta (4-tile walks → −4, 5-tile → −5, 8-tile → −8). Re-confirmed 2026-07-20: six clean SelfMovement segments, exact at 1/tile (7→7, 14→14, 5→5, 3→3, 2→2, 1→1) |
| Single shot (`weapon=0`) | **6** | Multiple isolated samples (shot at empty ground with no other wire activity in the 2 s 0x2E window) |
| Dual shot (`weapon=1`) | unknown precise value; presumed 10 | Higher than single. Tighter sample needed to nail the exact number. |
| Mine placement (per command) | **~1–2 per mine placed** | Noisy sample; 6 mines correlated with a −10 fuel delta over the placement window |
| Radar scan | **10** | Verified across 6 radar scans; reliable |
| Teleport | **floor(6 × euclidean distance)** — measured from start to the **actual landing tile** | Systematically validated 2026-07-20: every `teleport(x,y)` dispatch in every run was paired with its wire fuel delta (pre-hop `Self:` fix → post-hop `Fuel: A -> B` line, contaminated windows excluded). Post-2026-06-24 era (after the fuel double-count fix): **248/248 pairs exact**, costs 6–654. All-era: 2,335/2,538 exact; the 203 residuals are all pre-fix runs with broken fuel tracking. When the server drifts the landing off the requested target, the charge matches distance to the LANDING, not the target (624 drift hops confirm this) — planner estimates on the target can be off by a few fuel on drifted hops |

## Damage taken

| Source | Fuel loss to victim | Notes |
|---|---|---|
| Single shot HIT (taken from enemy `weapon=0`) | **45** | Verified by 3 Yuppler hits in the multi-tank PvP capture, all 3 matched −45 each |
| Dual shot HIT (taken from enemy `weapon=1`) | **90** | Verified by 3 Yuppler dual hits in the same capture, all 3 matched −90 each |
| Walking into an enemy mine | **45** | Verified at t+373.35s of the multi-pickup capture: 1-tile walk into mine at (92, 185) → −45 fuel |
| Mine cascade damage (your own mine detonating you) | not observed | Bot tests where player blew up their own mines didn't show a fuel delta on the placer |

Damage manifests as a fuel decrement — the game's "health" is essentially "fuel reserve" for combat purposes. When you take damage your fuel drops; when fuel hits zero the tank is deactivated.

## Container pickup mechanics

See [[fuel-system]] and the ``ContainerPickupDict`` decoder doc for the full wire shape. Key economy points:

- A radar scan reveals each container's **declared volume** (e.g. 300, 400, 1142).
- Walking over the container picks up **as much as the picker's tank can hold** (up to the 1100 cap).
- The wire reports the container's **remaining_volume** after pickup. Multiple pickers can drain one container in sequence.
- `remaining_volume = 0` means the container is exhausted (or it was equipment with no fuel attribute). Equipment pickups also fire a paired `0x67 EquipmentGain` carrying the items.
- **Multi-record bodies**: a single 0x43 message can carry 1, 2, 3 (or more) pickup records, fired when a tank's movement causes multiple container tiles to update in one server tick. Corpus distribution (156 sessions, 2026-06-20): 2653 single-record, 80 two-record, 2 three-record.
- **Duplicate broadcasts**: the server broadcasts each pickup twice within ~200 ms (one to the picker, one to the world view) -- empirical 43.9% duplicate rate. The dispatcher de-duplicates by `(x, y, remaining_volume)` signature within a 500 ms window so metrics and logs see each pickup once; the world-state mutation is idempotent regardless.

Verified breakdown from the multi-pickup capture:

| Container known volume | Picker tank before | Picker took | Container remaining (wire) |
|---:|---:|---:|---:|
| 1142 | 999 | 171 | 971 ✓ |
| 300 | 1083 | 17 | 283 ✓ |
| 400 | 1097 | 3 | 397 ✓ |
| 100 | 1096 | 4 | 96 ✓ |
| 571 (full bucket) | 456 | 571 | 0 ✓ |

Every row matches `remaining = declared − taken` exactly.

## Fuel deposit (your own tank donating fuel)

- Triggered by a deposit command: client code `'D'` (0x44), 6 bytes — x, y, **u16 LE amount** (tpclient.js `Wb` class, 2026-07-06). The amount accumulates during the mouse long-press ("DEPOSIT FUEL: N" HUD label) and is clamped client-side to current fuel.
- Client-gated by the fuel>100 check (`ce()`): a tank at ≤100 fuel cannot initiate a deposit.
- **Deposit floor = 100, server-enforced**: a max deposit always leaves exactly 100 fuel in the tank. Verified at four ranks 2026-07-06 — private 1000 (=1100−100), sergeant 1200 (=1300−100), major 1500 (=1600−100), colonel 1598 (=1700−100−~2 walked).
- Server places the fuel container on a tile **adjacent to the depositing tank's position** (verified: Yuppler at (131, 124) deposited → container at (132, 124)).
- The wire path: depositing player gets a 0x64 FuelDeposit; observers see the container via the next 0x4F RadarScan / viewport refresh.
- Containers are **invisible by default** to non-owning players. They appear via radar reveal.
- Bot use case: a tank at capacity can bank surplus fuel next to a defended position and reclaim it later.

## What's still open

Dual-shot exact fuel cost is still presumed-10, and homing-shot fuel cost has no row at all: in-combat fuel traces show paired −45/−10 deltas per firing tick (2026-07-19 soaks) that are not yet decomposed into per-weapon costs. Pinning them needs an isolated capture: fire one dual, then one homing, with no movement/radar in the same 0x2E windows.

## How this was discovered

Three sequential captures on 2026-06-20:

1. **PvP capture** (Artax vs Yuppler combat) gave us: enemy-shot damage values, weapon byte semantics, mine cascade, mine-on-mine destruction (see [[mine-mechanics]]).
2. **Multi-pickup capture** (Yuppler deposits 300, 400, 100; Artax picks up each in isolation) gave us: container_pickup.remaining_volume semantic, max fuel cap, walking cost, radar cost.
3. **Ghost-observation capture** (Artax killed orange-5 twice) gave us: the corrected `0x58 ≠ death` finding that drove the [[bot-behavior-contract]] 2-state liveness rewrite.

Each value above is matched 1:1 between a user-declared action and a measurable wire delta. No inferences without direct evidence.
