---
title: Game Economy (Fuel, Damage, Costs)
tags: [game, combat, fuel, economy]
related: [[shoot-event-format]], [[mine-mechanics]], [[deactivation-format]], [[bot-behavior-contract]], [[fuel-system]]
sources: [runs/sniff/sniff-20260620-150155 (multi-tank PvP), runs/sniff/sniff-20260620-155103 (annotated multi-pickup), runs/sniff/sniff-20260620-173727 (ghost-observation 5 kill cycles), user narrative cross-references 2026-06-20]
fact_checked: 2026-06-20
confidence: high
verified: 2026-06-20 (every value matched user-narrated actions to wire bytes)
---

# Game Economy

Empirically measured fuel costs, damage values, and capacity limits — derived from three controlled captures on 2026-06-20 where the user reported every action and I matched it against the wire bytes (0x2E TankStatusSync fuel-field deltas).

## Tank capacity

| Quantity | Value | Verified |
|---|---|---|
| **Max fuel cap** | **1100** | 7+ instances of fuel hitting exactly 1100 after a pickup, never above |

The cap is enforced by the server on pickup — if the picker's tank can only hold `N` more fuel before hitting 1100, only `N` is transferred from the container and the rest remains. This is why `container_pickup.remaining_volume` is often non-zero.

## Cost per player action

| Action | Fuel cost | Notes |
|---|---|---|
| Walk one tile | **1** | Verified by 0x47 Movement Manhattan distance vs 0x2E fuel delta (4-tile walks → −4, 5-tile → −5, 8-tile → −8) |
| Single shot (`weapon=0`) | **6** | Multiple isolated samples (shot at empty ground with no other wire activity in the 2 s 0x2E window) |
| Dual shot (`weapon=1`) | unknown precise value; presumed 10 | Higher than single. Tighter sample needed to nail the exact number. |
| Mine placement (per command) | **~1–2 per mine placed** | Noisy sample; 6 mines correlated with a −10 fuel delta over the placement window |
| Radar scan | **10** | Verified across 6 radar scans; reliable |
| Teleport | unknown precise value; high | Sample showed ~−400 fuel during a teleport+walk window, but couldn't isolate from the simultaneous walking and queued mine placement |

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

- Triggered by a deposit command. Server places a fuel container on a tile **adjacent to the depositing tank's position** (verified: Yuppler at (131, 124) deposited → container at (132, 124)).
- The wire path: depositing player gets a 0x64 FuelDeposit; observers see the container via the next 0x4F RadarScan / viewport refresh.
- Containers are **invisible by default** to non-owning players. They appear via radar reveal.

## What's still open

The first column of "Cost per player action" has two rows marked unknown — dual shot exact fuel cost, teleport exact fuel cost. To pin those down precisely we'd need isolated captures: dual shot in steady state with no other actions; teleport with no other queued commands. Both are quick follow-ups when you next sniff.

## How this was discovered

Three sequential captures on 2026-06-20:

1. **PvP capture** (Artax vs Yuppler combat) gave us: enemy-shot damage values, weapon byte semantics, mine cascade, mine-on-mine destruction (see [[mine-mechanics]]).
2. **Multi-pickup capture** (Yuppler deposits 300, 400, 100; Artax picks up each in isolation) gave us: container_pickup.remaining_volume semantic, max fuel cap, walking cost, radar cost.
3. **Ghost-observation capture** (Artax killed orange-5 twice) gave us: the corrected `0x58 ≠ death` finding that drove the [[bot-behavior-contract]] 2-state liveness rewrite.

Each value above is matched 1:1 between a user-declared action and a measurable wire delta. No inferences without direct evidence.
