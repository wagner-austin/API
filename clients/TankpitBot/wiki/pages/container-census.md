---
title: Container Census (where containers stand, fitted on one field and scored on another)
tags: [game-mechanics, economy, sim, measurement]
related:
  - "[[game-economy]]"
  - "[[equipment-system]]"
  - "[[sim-world-parameterization]]"
  - "[[capture-conformance]]"
  - "[[game-rules]]"
source_paths:
  - "src/tankpit_bot/validate/container_census.py"
  - "src/tankpit_bot/sim/actions.py"
  - "tests/validate/test_container_census.py"
  - "wiki/sources/container_census_2026-10-05.json"
  - "src/tankpit_bot/sim/field_choice.py"
  - "tests/sim/test_field_choice.py"
source_git_blobs:
  "src/tankpit_bot/validate/container_census.py": "697c797392eda9336a15f3e17842e62c2d17ff44"
  "src/tankpit_bot/sim/actions.py": "b1df4395545335965eb4a2d7b38290468a0128b4"
  "tests/validate/test_container_census.py": "931c51e109c8278365daf90c93145d975e158fa1"
  "wiki/sources/container_census_2026-10-05.json": "4212f91529ef466b754af6e69769dd5cdea685b2"
  "src/tankpit_bot/sim/field_choice.py": "cbfcc15d038057798795875d393e6149a0d23f2a"
  "tests/sim/test_field_choice.py": "89b0659f82080fd2f290f10f4372803101323c55"
provenance:
  - "Board task b008ab91 (the multiplayer track), Phase 4, 2026-10-05: tankpit-container-census over runs/bot and runs/sniff, the 586-capture archive copied from diphtheria:/mnt/archive-a/austinpc/tankpitbot-runs/runs; field01 captures span 2026-04-01 to 2026-09-02, field05 (World, Desert) captures 2026-08-26 to 2026-09-01"
fact_checked: "2026-10-05"
confidence: medium
hubs: [game-mechanics]
---

# Container census: fitted on field01, scored on field05

*Phase 4 of the multiplayer track (board task `b008ab91`), 2026-10-05.*

The track asked whether the server's container placement can be
recovered, and warned that one field cannot tell an algorithm from one
sample of its output. A second field now exists in the archive: 19 bot
captures on field05 (World, Desert). This page fits a distribution on
one field and scores it on the other. It is a **fitted distribution,
not a recovered algorithm**.

## How a scan becomes an observation

`tankpit-container-census` reads every tick in which the client sent a
lone radar. It applies the sim's own footprint law, `radar_covers`,
which is now public.[^1] An extra radar covers the whole 16x16 window,
recognised by the 0x49 ahead of the 0x4F. A free radar covers the
rank-scaled square around the tank. The window, tile and rank come
from the archive's last statements.

Every revealed tile is an observation, and every container the 0x4F
lists inside it is a sighting. **The footprint law holds:** only 51
listed containers (field01) and 60 (field05) fell outside the computed
footprint, over 13,249 and 2,963 scans.[^2]

## What the two fields show

| | field01 | field05 |
|---|---:|---:|
| captures with scans | 373 | 19 |
| scans | 13,249 | 2,963 |
| tiles observed | 65,505 | 65,006 |
| tiles ever holding a container | 15,282 | 4,478 |
| ground tiles ever holding one | 27.81% | 8.16% |
| water tiles ever holding one | 22.08% | 6.53% |
| rock tiles ever holding one | 0 of 8,505 | 0 of 9,307 |
| containers per revealed tile, at scan time | 0.66% | 0.80% |
| 16x16 block dispersion | 1.717 | 1.474 |

- **Rock never holds a container.** None did on either field.
- **Water holds containers at about 0.8 of ground's rate, on both
  fields.** The factor is 0.794 fitted on field01 and 0.801 fitted on
  field05.
- **Density at scan time transfers; the cumulative count does not.**
  Containers per revealed tile differ by a fifth between the fields,
  and the share of tiles *ever* seen holding one differs threefold.
  The radar is not undercounting: two extra-radar scans of the same
  window within two minutes listed identical tiles in 194 of 215 pairs,
  and an extra radar lists 1.77 containers per viewport on average.[^3]
  So containers keep appearing on new tiles. The cumulative share
  measures how long a field was watched (five months against six days),
  not how dense it is.
- **Mild clustering at the viewport scale.** Sites spread over 16x16
  blocks with a chi-square per degree of freedom of 1.7 and 1.5.
  Independent placement would read near 0.73 and 0.92.

## The held-out score

The fitted shape has two parts. Rock holds nothing, and water holds
containers at the train field's water factor times ground's rate. Only
the level is fitted on the held-out field, so the comparison is one
parameter against one: the held-out field's own constant rate.[^4]

| fitted on, scored on | shape | constant |
|---|---:|---:|
| field01 -> field05 | 0.23958 | 0.25075 |
| field05 -> field01 | 0.50480 | 0.54323 |

Mean log loss per observed tile; lower is better. The shape wins in
both directions. Finer neighbourhood features added nothing. Ground
tiles bucketed by water and rock within two tiles sat between 27% and
30% on field01.

## Playing another field

`tankpit-sim-run --field field05` (also with `--clients N`) plays any
of the 44 shipped minimaps.[^5] Seeds the field01 scenarios placed
on ground the chosen field lacks are moved to the nearest open tile
within one viewport. The container population is laid by the existing
walk over that field's own passable tiles. The ferry, larder, atlas and
ghost scenarios are written on field01 or replay its records, and they
refuse another field with `SIM_FIELD_SCENARIO`.

The first live practice-room session on field05's real terrain ran 120
of 120 ticks; layout `bot-20260706-223721` needed no seed moved. The
bot foraged as on field01: 24 container pickups and 22 radar scans,
answered with ordinary full-tank and empty-container receipts.[^6] The
population still uses field01's uniform counts on dry ground. Seeding
water at the fitted 0.8 and matching scan-time density are the
measured next step, not taken here.

## What this does not establish

The census sees where containers *were*, through one bot's scanning.
How fast sites turn over, and why, is not measured here. The
cumulative counts suggest more turnover than the persistence figures in
[[game-economy]], which compare volumes at tiles seen twice. Placement
within non-rock ground is consistent with uniform at the tile scale and
mildly clustered at the viewport scale. That is a distribution, which
two fields support. It is not a generator.

[^1]: `src/tankpit_bot/sim/actions.py`, `radar_covers`; `process_radar` now calls it, and `src/tankpit_bot/validate/container_census.py`, `tally_capture`, reads every lone-radar tick through it.
[^2]: `wiki/sources/container_census_2026-10-05.json`, the `fields` records (`footprint_misses`, `scans`, `observed`, `sites`, `tile_reads`, `container_reads`, `block_dispersion`), written by `make container-census` at this page's commit.
[^3]: a one-off scratch pass over the first 120 `runs/bot` captures on 2026-10-05, reading consecutive extra-radar scans of one window less than 120 s apart through `ArchiveMirror`: 215 pairs, 194 listing the same tiles, 14 where the second listed nothing; 7,399 extra scans listing 1.77 containers on average, 1,750 free scans listing 0.12.
[^4]: `src/tankpit_bot/validate/container_census.py`, `transfer_score`; the pinned case is `tests/validate/test_container_census.py`, `test_the_shape_is_fitted_on_one_field_and_scored_on_another`; the two rows are the `transfers` records of `wiki/sources/container_census_2026-10-05.json`.
[^5]: `src/tankpit_bot/sim/field_choice.py`, `resolve_field`, `require_field_scenario` and `settle_on_field`; `tests/sim/test_field_choice.py` plays a one-bot session, a two-bot field and both CLI forms on field05.
[^6]: `tankpit-sim-run --field field05 --practice --rounds 120 --stamp desert-live --layout bot-20260706-223721 --population-seed 7`, 2026-10-05, into a scratch archive; its events log counted 24 `container_pickup_dispatched`, 22 `radar_dispatch` and 18 `command_error` (17 code 5, 1 code 4).
