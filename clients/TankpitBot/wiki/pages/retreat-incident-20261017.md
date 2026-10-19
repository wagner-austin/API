---
title: Retreat failures in the 2026-10-17 World run
tags: [incident, combat, retreat, scheduling, fuel]
related: ["[[committed-intent]]", "[[bot-behavior-contract]]", "[[fuel-system]]", "[[fleet-coordination]]", "[[teleport-mechanics]]"]
fact_checked: "2026-10-19"
confidence: medium
hubs: [architecture]
verified: Preserved captures and release source inspected; corrective regression tests executed offline without a live restart.
provenance:
  - "Corvis task d947f299; originating task 6275ecaa, handoff abd5bb45-eb63-4c67-88fe-470f1d411629."
  - "/home/corvis/tankpit-evidence-20261017/manifest.json: 20 files, 104395465 bytes; every size and SHA-256 reverified."
  - "Runtime API release 3888f06b22f5eea63d8e8cb5fe9934143ee7f392; tv-1 run bot-20261017-093852, tank 740 / Yuppler."
source_paths:
  - "src/tankpit_bot/state/fuel_confidence.py"
  - "src/tankpit_bot/state/tank_mutations.py"
  - "src/tankpit_bot/sniffer/world_state_tiles.py"
  - "src/tankpit_bot/bot/ai/equipment.py"
  - "src/tankpit_bot/bot/ai/equipment_search.py"
  - "tests/world_state/test_fuel_confidence.py"
  - "tests/bot/ai/test_suspect_fuel.py"
  - "src/tankpit_bot/sniffer/world_state_tanks.py"
  - "src/tankpit_bot/sniffer/world_state_containers.py"
  - "src/tankpit_bot/analysis/scan.py"
  - "src/tankpit_bot/container/decoders/events.py"
  - "src/tankpit_bot/browser/cdp_utils.py"
  - "src/tankpit_bot/browser/cdp_service.py"
  - "src/tankpit_bot/physics/costs.py"
  - "tpclient.js"
  - "src/tankpit_bot/bot/ai/escape_policy.py"
  - "src/tankpit_bot/bot/ai/collect_locks.py"
  - "src/tankpit_bot/bot/ai/resource_search.py"
  - "tests/bot/ai/test_retreat_regressions.py"
  - "tests/bot/test_tick_loop_coverage.py"
  - "src/tankpit_bot/bot/tick_loop.py"
  - "src/tankpit_bot/bot/tick_body.py"
  - "src/tankpit_bot/bot/tick_loop_actions.py"
  - "src/tankpit_bot/bot/world_sync.py"
  - "src/tankpit_bot/bot/ai/combat_break.py"
  - "src/tankpit_bot/bot/ai/combat_prep.py"
  - "src/tankpit_bot/bot/ai/collect_mode.py"
  - "src/tankpit_bot/bot/ai/collect_mode_outcomes.py"
  - "src/tankpit_bot/bot/ai/collect_pickups.py"
  - "src/tankpit_bot/bot/ai/movement.py"
  - "src/tankpit_bot/bot/ai/larder.py"
  - "src/tankpit_bot/bot/ai/collect_hops.py"
  - "src/tankpit_bot/bot/executor.py"
  - "src/tankpit_bot/state/container_mutations.py"
  - "src/tankpit_bot/fleetshare/report.py"
  - "src/tankpit_bot/replay/engine.py"
source_git_blobs:
  "src/tankpit_bot/state/fuel_confidence.py": "2e3b1301bd85c720fe70d94260b312fdf87e7f8a"
  "src/tankpit_bot/state/tank_mutations.py": "d3ab81a2cc867b75322a379a785c9cc00da810b9"
  "src/tankpit_bot/sniffer/world_state_tiles.py": "d636e4d712145a24e56fff37ce70b2c066487558"
  "src/tankpit_bot/bot/ai/equipment.py": "3a0aa9d05f106db46f77d9332e305c721a4f460f"
  "src/tankpit_bot/bot/ai/equipment_search.py": "b23f9cd58a5fb80f878e1f9a897dc348f760a647"
  "tests/world_state/test_fuel_confidence.py": "d6f8d98f1f3b92769f3edb1107aaa668d1ea7c80"
  "tests/bot/ai/test_suspect_fuel.py": "e50b3d842171809bc6525f35df63dc79dc51759f"
  "src/tankpit_bot/sniffer/world_state_tanks.py": "52d730f73c82657055e931d2bf29d1e0648edbdf"
  "src/tankpit_bot/sniffer/world_state_containers.py": "5db49ebf63ccd417658d6c29a39c01f6ee662a58"
  "src/tankpit_bot/analysis/scan.py": "808b553d59c0a4e80e0907c8fef7c28bf8843cbc"
  "src/tankpit_bot/container/decoders/events.py": "c2a81cd5ffe3b478915f5a5f2b85550bcf210577"
  "src/tankpit_bot/browser/cdp_utils.py": "810b4d1eb9fbba0f598b3bf6c0ac2da4b55463a3"
  "src/tankpit_bot/browser/cdp_service.py": "fb0dd48c72ebafba5a165c28c90bba3a2bb1f06c"
  "src/tankpit_bot/physics/costs.py": "cfd4ecc3b7dca858aebb7fdc7e4b5c9f93d77f62"
  "tpclient.js": "cb253fe55b10221291a35382d2f4e2efcd02f2ff"
  "src/tankpit_bot/bot/ai/escape_policy.py": "b82ae280e9b10292de53f265312e350e7efca962"
  "src/tankpit_bot/bot/ai/collect_locks.py": "24362310ed1be0cbdf169ff31efd4e1478e30a92"
  "src/tankpit_bot/bot/ai/resource_search.py": "423bfcd2cc3f00b37ffc8817ab857b1750335d33"
  "tests/bot/ai/test_retreat_regressions.py": "47ffecbc76df1f5cafa1ed87a4d350445fc31fef"
  "tests/bot/test_tick_loop_coverage.py": "16847e20227f3d5eaf2448fd3faff2fb7b531582"
  "src/tankpit_bot/bot/tick_loop.py": "66e1a47c4fed13cbcfe7ec05a9a4a09f131557b9"
  "src/tankpit_bot/bot/tick_body.py": "f89288ff6a4b37a57ad41520167638635f2a3d17"
  "src/tankpit_bot/bot/tick_loop_actions.py": "48ea79bf8ae71fcca61b91ffdf552f496fa116e1"
  "src/tankpit_bot/bot/world_sync.py": "902858477bc22a632a38b8a5b2b6618b08be02be"
  "src/tankpit_bot/bot/ai/combat_break.py": "295d58704a009d1f30fa2f09f1f6d0984c2bb6a7"
  "src/tankpit_bot/bot/ai/combat_prep.py": "b527aa154060e6a4dd4f1bce107a470d83ee9f69"
  "src/tankpit_bot/bot/ai/collect_mode.py": "9a2b8bb090dd5039effce7eb67405a6d7636254f"
  "src/tankpit_bot/bot/ai/collect_mode_outcomes.py": "85152d9a6f3a0e32ca1f471a7d28b7d79f71bc43"
  "src/tankpit_bot/bot/ai/collect_pickups.py": "652a96bec44ae109046be8e59d361b842a1d6a31"
  "src/tankpit_bot/bot/ai/movement.py": "c4df95be78254ee18e2a1d670f4b509a0551ddac"
  "src/tankpit_bot/bot/ai/larder.py": "da1a5f3dcecff2513da65afb4c0516a93be0bd4e"
  "src/tankpit_bot/bot/ai/collect_hops.py": "3d4b8ff983ff8cbe802cc7e7110e8160f0d9fe59"
  "src/tankpit_bot/bot/executor.py": "d2430383d3e214d08ce64a5c88d54f649962a620"
  "src/tankpit_bot/state/container_mutations.py": "e6be8dedfa1b403716d50312f691c35be64a96b5"
  "src/tankpit_bot/fleetshare/report.py": "cfc0608da47dc6e2b559ecd46d6248a054ccdce9"
  "src/tankpit_bot/replay/engine.py": "7a1333337901d19972a617c3d59e8e20d4efcc03"
---

# Retreat failures in the 2026-10-17 World run

The engagement break activated. Its delegated action sequence failed to secure fuel or leave both attackers, and the control loop added measurable delay after responses. The incident reconstruction describes release `3888f06b2`; the later corrections below are offline-verified source changes, not proof that a particular alternative would have survived. Frontmatter pins describe the corrected tree; the provenance release identifies the historical source. All incident timestamps below are UTC on October 17; event line numbers are one-based, capture message indexes zero-based.[^1]

## What happened

| Event window | Verified sequence |
|---|---|
| 09:47:02–08 | Break at fuel 698 with **two** recorded hits. The locked fuel plan was not executable, so ordinary collection dispatched equipment. After that pickup, fuel was 651 and the bot chose a cost-565 fuel hop to (103,56), opened the map, and chose it again at fuel 606. The server refused code 8. The next fuel update read 561. The handler reset the action and replanned in tick 205. |
| 09:49:10–16 | Break at 687, five hits, projected fuel −483, escape floor 390; human-fight resume latch set to 800. A walk toward (148,249), believed volume 580, hit a mine and got code 1. The next decision converted the same pickup to a teleport. It opened the map, sent the teleport at fuel 416, and landed exactly at (148,249) with fuel 256. No fuel gain followed. |
| 09:49:16–20 | Landing radar took priority over continued retreat; fuel fell to 201. A walk to (153,251) **did gain 70**, then code 4 closed the drained container. After another hit, fuel was 175. Calling this pickup a failed transfer would be wrong. |
| 09:49:20–27 | Map open for a new hop; fuel fell 175→130→85 while waiting. Replanning chose a cost-81 desperation hop to (148,238). Code 8 arrived; a later fuel update read 40. The handler replanned and sent a walk in the same tick 261. Tank 740 died at 09:49:27; the killer was 506 / red-7, not the held target 814 / 28824528. |

The table is grounded in events L3212–3282 and L3979–4142 plus the capture's death message. No `stall_timeout` outcome was found in the five saved runs.[^1]

## Corrections to the initial handoff

**Both fuel refusals were fast on the wire.** They arrived 62 ms and 61 ms after the corresponding sends, before the next −45 fuel update. The event ledger's 1082/1416 ms durations include local handling; they are not server execution times. Fuel updates arrived 311/1045 ms **after** the respective refusal. Thus the earlier inference “damage while a long teleport was executing caused the refusal” is not established. Unreported server fuel changes, the server's affordability rule, and resolved landing cost remain alternatives. Decoding the actual outbound commands confirmed the intended coordinates, excluding a mislabeled destination in these two sends.[^2]

**The advertised 580 fuel was not credited on landing.** Yuppler saw 580 via radar (capture message 1592, 09:48:57.711). Its later belief used tv-2's radar-processing timestamp 09:48:58.672; tv-2 had received the same radar payload at 09:48:57.719 without sending its own scan. These are shared observations, not two independent scans. The import preserved that timestamp rather than refreshing it every tick. The tv-2 viewport patch advertised 580 at 09:49:13.726, and Yuppler's own landing patch advertised 580 at 09:49:15.762. Using viewport origin plus `(col−1,row−1)` places both entries at (148,249). Yuppler then received fuel=256, a landing confirmation, fuel=256 again, and two remaining-volume-zero receipts. The sibling also received a removal receipt. These facts establish zero credited gain and a cleared target belief, not that 580 collectible fuel still existed at landing. A freshly received viewport patch does not independently establish the freshness of its underlying container value. Stale-container correction remains a viable explanation; neither lost fuel nor a competing consumer is proved.[^3][^15]

## Further fuel investigation on October 18

The unchanged-fuel/cache-clear signature repeats outside the fatal engagement. Across the five preserved captures, there are 140 outbound teleports and 138 landing confirmations. Restricting each landing to the interval before its next non-keepalive, recognized gameplay command gives 42 cases with both a container update within Manhattan distance one of the actual landing and a subsequent own-tank fuel reading. Of these, 31 first readings increase and 11 stay unchanged. These are conditional counts in this five-run sample, not 11 confirmed lost transfers or a measured server failure rate. Every unchanged case has a positive container volume in the landing viewport and two subsequent zero-volume updates for that tile.[^11]

| Run | Teleport send index | Landing index | Next fuel index | Fuel at landing / next | Container tile | Advertised volume | Zero-volume indexes |
|---|---:|---:|---:|---:|---|---:|---|
| tv-1 | 618 | 625 | 626 | 734 / 734 | (85,20) | 706 | 627,628 |
| tv-1 | 1656 | 1670 | 1671 | 256 / 256 | (148,249) | 580 | 1672,1673 |
| tv-1 | 1892 | 1899 | 1900 | 789 / 789 | (164,236) | 501 | 1901,1902 |
| tv-2 | 1080 | 1085 | 1086 | 1186 / 1186 | (41,120) | 1098 | 1087,1088 |
| tv-2 | 1370 | 1378 | 1379 | 867 / 867 | (138,244) | 321 | 1380,1381 |
| tv-2 | 1689 | 1695 | 1696 | 915 / 915 | (164,238) | 263 | 1697,1698 |
| tv-3 | 285 | 290 | 291 | 752 / 752 | (123,152) | 616 | 292,293 |
| tv-3 | 1284 | 1289 | 1290 | 962 / 962 | (40,148) | 318 | 1291,1292 |
| tv-3 | 1518 | 1523 | 1524 | 765 / 765 | (19,144) | 923 | 1525,1526 |
| tv-3 | 1717 | 1722 | 1723 | 724 / 724 | (105,188) | 687 | 1724,1725 |
| tv-4 | 336 | 342 | 343 | 762 / 762 | (60,163) | 540 | 344,345 |

Extending each inspection 6.5 seconds past its landing avoids treating the first reading as the end of the evidence. The fatal tv-1 case reads 256→211→201 after a radar command, then 156→150→220→175 around the separate pickup at (153,251). There is no delayed 580 credit in that interval. One important control is tv-3 landing 1523: after radar, fuel becomes 755, then 1000, but the accompanying container update is at **(20,144)** with 603 remaining, whereas the zero-credit landing removed **(19,144)**. This later adjacent pickup must not be attributed to the original 923-volume container. Other later increases in these windows follow new pickup commands; an indefinitely delayed credit cannot be excluded by a finite window.[^11]

The wire also rules out a local arithmetic-only explanation for the 580 case: decoded gameplay bodies at tv-1 indexes 1662, 1669 and 1671 are byte-identical and all report fuel 256. The archived browser client treats `0x43` as a tile-cache update; it does not infer a fuel gain or name a recipient. Our decoder's `container_pickup` name therefore must not be read as confirmation that the observing tank received the removed contents. None of the other four recorded bots shows a corresponding 580 gain in the 09:49:12–18 window. This excludes an observed sibling receipt in that window, not every possible competing consumer.[^12]

**Shared radar and stale corrections were checked explicitly.** Radar is a delta update: explicit zeros clear container beliefs, and omitted entries need not be empty. The fleet-coordination contract also records team-shared reveals. In this capture, tv-1 sends radar at 09:48:56.885 and both tv-1/1592 and tv-2/1422 receive the identical 580-container/two-mine payload. Again, tv-1 sends radar at 09:49:16.954 and both tv-1/1681 and tv-2/1480 receive the identical 70-container/six-mine payload. Tv-2 sends no radar between 09:48:45.309 and 09:49:19. The second shared scan occurs after the target's tv-1 zero updates at 09:49:15.789/15.791, so it cannot explain their arrival. No other recorded bot scanned a viewport covering (148,249) around that clearing; tv-3's contemporaneous scan was far away, and tv-4's capture had ended. This rules out a matching scan by the recorded siblings, not stale correction during landing or an unrecorded teammate's action.[^15]

The same bounded screen was applied to all 11 cases: other bots' radar sends within −4 to +1 seconds of landing, with the target inside their stored 16×16 viewport. Ten have no candidate. The remaining case, tv-3 landing 1289, overlaps demo-1 radar send 553; its response 556 arrives about two seconds before landing and contains no entry for (40,148), so it supplies no explicit zero for that target. This screen uses viewport containment as a permissive upper bound, not proof that a built-in radar reached every tile in the viewport. It is not an exhaustive exclusion of delayed commands or unrecorded players.[^15]

**Enemy collection has direct supporting evidence.** Both tv-1/1612 and tv-2/1439 place human enemy 814 on the exact target (148,249) at 09:49:03.726. The following status messages, tv-1/1613 and tv-2/1440 at 09:49:03.727, raise its damage tier from 2 to 3 while rank stays 6. That tier is a fuel-quartile indicator, so this is evidence of an increased fuel band, not additional damage. The enemy moves to (153,252) by tv-1/1627 at 09:49:07.722; our landing follows at 09:49:15.765, about 12 seconds after the enemy occupied the fuel tile. Enemy refueling there, followed by correction of our stale container value when we attempt collection, is now the leading explanation for this case. The observer messages carry no exact enemy fuel total or actor id on the container update, so they do not prove that enemy 814 took all 580. This also does not establish the cause of the other ten cases.[^16]

Successful controls also eliminate two simple proposed explanations for code 8:[^13]

| Control | Observed fuel accounting | Implication |
|---|---|---|
| demo-1 send 09:45:00.139, (117,99)→(120,28) | 433−426=7 before pickup; Manhattan cost would be 444 | Neither a universal 100-fuel landing reserve nor Manhattan affordability explains all teleports. |
| tv-1 send 09:47:10.659, (197,49)→(136,10) | Send fuel 516; incoming hit leaves 471; teleport costs 434 and leaves 37 before pickup | The same tank can land with less than the earlier refused hop's apparent 41-fuel margin. |
| tv-2 send 09:49:38.784 | Send fuel 343; incoming hits leave 208; teleport costs 207 and leaves 1 before pickup | A positive one-fuel landing is accepted even in a sequence with incoming damage. |

These controls do not prove what fuel or resolved destination the server used for either rejection. Unreported state changes and destination-dependent acceptance remain open. The client-side Euclidean charge calculation should not be replaced with an invented minimum reserve or Manhattan formula on this evidence.[^13]

For reproduction, use the existing decode pipeline described below, retaining each original message index and sorting by `timestamp_ms`. Resolve the player's id from a self status carrying fuel; track its absolute fuel and position, open a window at each outbound teleport, and identify the matching landing confirmation. In each window compare the last own fuel at confirmation with the first subsequent own fuel, and join nearby container updates to the landing viewport using `(left+col−1, top+row−1)`. Close the window at the next recognized gameplay command; then inspect the following 6.5 seconds separately for deferred gains and other pickups. This separates the census criterion from the later-credit check.[^11]

Within-session send/receive deltas use CDP timestamps mapped through one anchored clock offset. Separate browser sessions anchor independently, so millisecond ordering across tv-1 and tv-2 is not a global server chronology. The repeated signature narrows the investigation to server-reported container/fuel behavior, but does not distinguish stale server map state, competing consumption, and other server collection rules. No new live run or guessed protocol correction was used to produce these findings.[^14]

## What the enemy-consumption evidence means for the bot

The existing accounting rule remains correct: `pickup_container` changes container beliefs, never our fuel. Absolute fuel messages remain authoritative. A zero remaining-volume receipt removes the target and records a tombstone so older fleet sightings cannot reintroduce it. The corrected retreat policy also prices transit and incoming-fire exposure without adding anticipated pickup fuel. Enemy consumption therefore does not invalidate the retreat, clearance or response-wake fixes, and it supplies no reason to change teleport cost arithmetic.[^17]

The October 19 correction implements a conservative local inference in `containers_after_enemy_refuel`: a live enemy's movement observation followed within 2000 ms by a wire status showing a higher fuel quartile at unchanged rank marks known positive fuel at that position with confidence at most 0.5. This is a ranking signal, not a calibrated depletion probability. It retains the advertised volume and resource observation time, cites the movement/status evidence, and leaves our fuel unchanged. Unknown tanks, our tank, teammates, rank changes, deactivation/respawn transitions, stale positions, and map/DOM-only position evidence do not establish this inference. The rule intentionally covers the observed movement/status pattern; it is not a general enemy-consumption detector.[^17]

Walk candidate ranking, fuel-lock replacement, larder scoring and desperation-hop scoring prefer higher-confidence eligible fuel before their existing value/distance criteria. A sole suspect target remains eligible under the existing safety gates: partial enemy consumption is possible. Repeated viewport advertisements, fleet reports and synthetic known-resource radar refreshes retain suspicion. An explicit radar quantity or 0x43 remaining-volume correction restores confidence; an explicit zero removes the target. This does not change retreat exposure budgeting or assign an exact enemy transfer.[^17]

Container receipt diagnostics now say “Container cleared” or “Container updated … fuel remains”, because the receipt identifies no collector. Regression tests cover the observed 580-fuel pattern, exclusions, repeated status, refresh authority, partial consumption, target ranking, held-lock preference and sole-target recovery. The correction is verified offline; it does not prove a particular alternate move would have survived, and no live restart is part of it.[^17]

## Confirmed control-flow gaps in release 3888f06b2

1. **A break does not consistently select escape actions.** `assess_engagement_break` counts one hit; `escape_under_fire_decision` requires three. `refuel_for_hunt` delegates to ordinary `decide_collect_mode`. At two hits the escape gate declines and the equipment-before-fuel order remains eligible. This explains the earlier equipment pickup without requiring a missing break or slow computation.[^4]
2. **A pickup helper can bypass escape clearance.** The under-fire branch returns `select_and_pickup_fuel` directly. That helper calls `walk_or_teleport`, which implements the prior user instruction to teleport after a mine hit. Its returned teleport never reaches `_hop_escapes_attacker`. The fatal run logs this exact mine-flip branch twice. The 09:49:15.765 landing was only Manhattan 2 from red-7 at (149,248), and 8 from 28824528 at (153,252). The later clearance helper considers only the held combat target and, even there, ultimately permits a trapped candidate; it is not an all-attacker safety guarantee.[^5]
3. **Landing sensing outranks retreat.** `_sense_and_safety_gates` returns a landing radar before it asks for an under-fire escape. A mine-flipped pickup does not carry the larder hop's suppression semantics. The observed radar therefore follows current ordering, including the historical “radar right on landing” rule; blindly reversing every landing scan would regress mine/displacement protection.[^6]
4. **Affordability is not an exposure budget.** The larder accepts `cost < current fuel` and checks reserve after adding the expected pickup. Desperation permits `cost <= fuel`; the executor checks current belief against a displacement-adjusted lower cost bound. None of these checks budgets incoming damage over map opening, local processing, and transit. The 606−565=41 and 85−81=4 margins were below one observed 45-fuel hit. This is a confirmed risk-policy gap, not proof of the server's reason for these immediate refusals.[^7]
5. **An already-buffered reply can lose the early wake.** `_wait_between_ticks` records the current buffer length and wakes only if it grows. With a pending teleport, one already-buffered reply and no additional traffic, an offline call requested eight 250 ms sleeps: **2000 ms despite pending input**. The existing three early-wake tests pass but cover an empty buffer at entry. Shot actions also take the full 2000 ms sleep because `has_in_flight_action` does not treat shooting as a movement-style pending action.[^8]

## Measured responsiveness and its limits

| Command sent | Matching response received | Send→response | Response→next command |
|---|---|---:|---:|
| Teleport 09:47:07.184 | Code 8 09:47:07.246 | 62 ms | 1875 ms, next map open |
| Map 09:49:13.129 | Map data 09:49:13.717 | 588 ms | 1255 ms, next teleport |
| Teleport 09:49:14.972 | Landed 09:49:15.778 | 806 ms | 1176 ms, next radar |
| Radar 09:49:16.954 | Radar data 09:49:17.733 | 779 ms | 1128 ms, next pickup |
| Map 09:49:20.925 | Map data 09:49:23.720 | 2795 ms | 899 ms, next teleport |
| Teleport 09:49:24.619 | Code 8 09:49:24.680 | 61 ms | 1824 ms, next walk |

These are matched command/response types, not “first subsequent packet” latency. Capture list order is not always chronological: rejection entries 1308 and 1709 precede their send entries 1309 and 1710 in the list, but their timestamps are later. Sort by `timestamp_ms`. The table measures client-visible elapsed time, not CPU time; it cannot divide the local tail into planner computation, synchronous browser calls, HUD/caption work, fleet file exchange, and host scheduling.[^2]

The tick drains wire at its start, then polls the game log and performs other synchronous browser reads, plans, emits alignment diagnostics, dispatches, updates captions/HUD, exchanges fleet knowledge, and publishes status before sleeping. It does not drain again at every phase. Pending action and shot-feedback gates precede new strategy evaluation; incoming damage has no general preemption path there. The buffer drained by `world_sync` contains payload strings, so the world mutations do not retain each message's capture-arrival timestamp. The 953 ms radar receipt-to-belief stamp difference above is one measured consequence. None of this licenses indiscriminate repeated commands: replacing an in-flight action can reset server work.[^9]

## Corrective work and reproducible validation

The following table records the investigation's corrective cases. The subsequent implementation is described below; server interpretation and phase profiling remain open. Preserve never-drop combat locks, post-mine movement doctrine, the separate confirmed map-open phase, and displaced-landing sensing while fixing their composition.[^4][^5][^6][^7]

| Work | Offline validation case |
|---|---|
| Consume buffered input before waiting; separate response processing from command pacing | Pending teleport/map/scan with a reply already queued must drain without an extra 2 s sleep; traffic arriving during sleep still wakes; empty buffer remains bounded; processing damage must not duplicate or reset a pending server command. |
| Carry a committed retreat obligation into action selection | One- and two-hit losing projections, and a latched break whose hit window falls below three, must not silently resume ordinary equipment priority. Retain target lock and the existing return-to-fight policy. |
| Validate the actual returned escape command | A mine-triggered teleport from the pickup helper must be assessed as a teleport against all fresh incoming attackers. Include safe in-viewport walking, unsafe nearby teleport, and no safe route without a replan loop. |
| Price the complete escape sequence | Replay a 45-fuel hit around map open and dispatch with margins 41 and 4; distinguish latest observed fuel, observation age, potential landing displacement, and uncertain pickup credit. Do not turn a conservative budget into an unaffordable-action retry loop. |
| Resolve the no-gain landing and sensing exception | Reproduce the exact 580 patch→256 self sync→landed→256 sync→two zero-removal messages. Assert zero credited gain and removed belief. Compare successful historical auto-picks before changing protocol interpretation; keep mining protection when choosing whether retreat can skip a landing scan. |
| Measure the local critical path | Record capture arrival, drain, plan start/end and dispatch with a monotonic clock, plus durations of browser reads, captions/HUD and fleet I/O. The saved logs cannot recover these missing phase durations. |

Offline reproduction uses the manifest-preserved files, not `latest.*` on the live host. Verify each manifest entry's byte count and SHA-256 first. Event extraction: parse JSONL and select the line ranges cited above; capture extraction: load the named JSON, build its session table with `capture.xor.build_session_xor_table`, split each selected payload with `capture.frames.split_payload_frames`, then decode received `0x2e` bodies using `protocol.decode_message(body[0], xor_decode_body(body, table, offset=1))`. Decode the selected outbound teleport bodies with `sim.commands.decode_client_command(xor_decode_body(body, table, offset=1))`. Keep keys and raw authentication frames private. The selected coordinates, timestamps and numeric fuel readings are the reproducible outputs.[^1][^2][^3]

Scheduling reproduction: construct `Bot` without calling `run`, assign a pending `make_in_flight_action(ActionKind.TELEPORT,148,238,get_current_time_ms())`, preload `_cdp_message_buffer` with one string, and call `_wait_between_ticks` with the existing test `_FakePage` subclass whose `wait_for_timeout` only records requests. Assert return `2000` and waits `[250.0]*8`. The companion baseline command `python -m pytest tests/bot/test_tick_loop_coverage.py::TestEarlyWakeSleep -o addopts='' -q` returned **3 passed**. This demonstrates current behavior through the real wait function; it does not simulate the live server or prove that this branch caused all observed latency.[^8]

The standard `replay_session` deliberately omits executor/in-flight state and fleet exchanges; its planner traces are not a faithful reproduction of this incident's scheduling or shared beliefs. Use the captures for protocol facts and the original event trace for dispatched decisions. No live game was entered during this investigation; the shutdown requirement from the originating handoff remains in force.[^1][^10]

## Corrections after the investigation

Task `afdc4b26` implements the operator's follow-up request to correct and double-check these issues. `escape_policy.py` owns the shared rules: a fuel-resume latch keeps retreat active even at zero, one or two recent hits; without a latch, isolated hits retain the ordinary collection policy. Fuel recovery outranks equipment travel and equipment-plan continuation during that commitment. Combat target ids and the resume latch survive each resulting fuel decision.

Hop candidates are checked **before ranking**, including fuel pickups that movement converts into teleports, locked pickup continuations, larder, equipment, desperation and fuel-dot hops. The actual resolved landing must clear the held target and every fuel-confirmed, wire-fresh attacker with a known position. Clearance uses a 16-tile square envelope: the old Manhattan test wrongly admitted a diagonal eight tiles away on each axis. A rejected top candidate therefore cannot hide a safer second candidate, and the old trapped-hop branch no longer reinstates a rejected landing.

Transit reserves no pickup credit. During measured fire the budget is twice the larger of the observed average hit and the window's per-tick damage rate, plus nine fuel for ring-1 landing displacement. Cost plus that exposure must stay below current fuel. The same selectors re-evaluate after map opening; the captured 41- and 4-fuel margins fail this policy. This is conservative risk pricing, not a guarantee against arbitrary future damage or a new assertion about why the server sent code 8.

A validated committed escape can teleport or open its map before scanning ground it is leaving. A proposed walk still requires the landing scan and any pending mine reveal. If no escape candidate exists, sensing gets its turn before the exhausted-collection outcome. Fuel pickup teleports, including locked continuations, now carry the harvest suppression flag; displaced landings still use the existing scan exception.

The between-tick wait reads pending state without consuming action outcomes. A queued reply returns immediately; fresh traffic wakes pending actions and shots within the 250 ms slice. Idle startup retains its normal wait. The tick's existing movement and shot-feedback gates remain responsible for preventing duplicate commands. Session budgeting counts elapsed iteration time as well as requested waiting, so immediate replies do not freeze the bounded-session budget.

Executable regressions are `tests/bot/ai/test_retreat_regressions.py` and `tests/bot/test_tick_loop_coverage.py::TestEarlyWakeSleep`. They assert fuel-first behavior through hit-window decay, mine-flipped and locked commands, second-attacker and diagonal clearance, safe alternatives, both low-margin refusal shapes, re-evaluation after map fuel loss, pre-walk sensing, queued and arriving shot replies, and preservation of an unconsumed rejection. The original 2000 ms reproduction above documents the **old release**; the corrected queued-reply expectation is zero. The complete package check remains the release gate. Neither the zero-gain 580 landing nor the exact server affordability discrepancy is claimed solved, and no live bot run is part of this correction.

[^1]: Private evidence root `/home/corvis/tankpit-evidence-20261017/`; `tv-1/latest.events.jsonl` SHA-256 `75ae12caad498dab8ec098c9e7149b4e30d40c7fb123e6a51e1844547c4273d0`; original host `sedona-tailnet:C:/fleet/tankpit/runs/bot`. JSONL L3212–3282, L3979–4142; capture message 1719 at 1792230567733 has victim 740, killer 506. Handoff `abd5bb45` records the graceful shutdown and no-restart instruction. All five event files were searched for `outcome=stall_timeout`, zero matches.
[^2]: `tv-1/latest.capture_session.json`, message indexes 1303–1313, 1648–1699, 1705–1716; exact timestamps shown in the timing table. Refusal messages 1308/1709; outbound 1309/1710; later fuel updates 1311/1712. Event outcome durations: JSONL L3268/L4126. Incoming messages are decoded through the release's protocol; outbound commands decode to (103,56), (148,249), (148,238).
[^3]: `tv-1` capture messages 1592, 1662–1673; `tv-2` messages 1422, 1463, 1478. `tv-1` JSONL L3932, L3958, L3987, L4025, L4051; `tv-2` L3574, L3632, L3645. `src/tankpit_bot/state/container_mutations.py`, `merge_container_sighting`, preserves observed_ms; `src/tankpit_bot/fleetshare/report.py`, `_container_rows`, exports that stamp with a 60000 ms publication horizon. Removal receipts contain no consumer identity.
[^4]: `src/tankpit_bot/bot/ai/combat_break.py`, `assess_engagement_break` and `_MIN_SUSTAIN_HITS`; `src/tankpit_bot/bot/ai/combat_prep.py`, `refuel_for_hunt`; `src/tankpit_bot/bot/ai/collect_mode_outcomes.py`, `escape_under_fire_decision`; `src/tankpit_bot/bot/ai/collect_mode.py`, `decide_collect_mode`. Incident JSONL L3212–3221.
[^5]: `src/tankpit_bot/bot/ai/collect_mode_outcomes.py`, `escape_under_fire_decision` and `_hop_escapes_attacker`; `src/tankpit_bot/bot/ai/collect_pickups.py`, `_first_walkworthy_fuel`; `src/tankpit_bot/bot/ai/movement.py`, `walk_or_teleport` and `_mine_flip_teleport`. JSONL L4004–4013/L4020–4028; capture messages 1664–1668 give the landed tank and both attackers' coordinates.
[^6]: `src/tankpit_bot/bot/ai/collect_mode.py`, `_sense_and_safety_gates`; `src/tankpit_bot/bot/ai/collect_mode_outcomes.py`, `scan_on_landing_decision`; `src/tankpit_bot/bot/ai/collect_hops.py`, `_hop_toward_fuel_larder`. JSONL L4048–4063.
[^7]: `src/tankpit_bot/bot/ai/larder.py`, `select_fuel_larder_hop`; `src/tankpit_bot/bot/ai/collect_hops.py`, `desperation_fuel_hop`; `src/tankpit_bot/bot/executor.py`, `_dispatch_teleport_command`. Incident JSONL L3255/L4112. The executor docstring records the failed same-tick map/teleport burst experiment and why the confirmed-open precondition remains.
[^8]: `src/tankpit_bot/bot/tick_loop.py`, `_wait_between_ticks`; `src/tankpit_bot/bot/tick_loop_actions.py`, `has_in_flight_action`. Offline reproduction and three existing early-wake cases executed against release-equivalent source on 2026-10-17 Pacific time, without a browser session.
[^9]: `src/tankpit_bot/bot/tick_body.py`, `_tick_once` and `_accept_dispatched_decision`; `src/tankpit_bot/bot/tick_loop.py`, `run_tick_loop`; `src/tankpit_bot/bot/world_sync.py`, `drain_messages`. No per-phase CPU profiling was preserved; the capture timestamps and second-resolution event processing times are different clocks of observation.
[^10]: `src/tankpit_bot/replay/engine.py`, module contract and `replay_session`: received frames are grouped into nominal tick batches, planner is evaluated without execution masking, and each replay owns a fresh world service.
[^11]: The five manifest-pinned `latest.capture_session.json` files under the private evidence root; zero-based indexes in the table. Follow-up windows are measured from landing timestamps through +6500 ms. In particular, tv-1 messages 1670–1702 and tv-3 messages 1523–1533 distinguish later actions and adjacent gains. `src/tankpit_bot/analysis/scan.py`, `load_capture_session` and `decode_session_frames`, define the existing load/split/XOR pipeline used for the offline comparison.
[^12]: Private tv-1 messages 1662, 1669, 1671–1673; all five captures' gameplay commands and own fuel readings from 09:49:12 through 09:49:18. `src/tankpit_bot/container/decoders/events.py` decodes cache coordinates and remaining volume without a consumer id. Archived `tpclient.js`, `$g.prototype.h` and the `V.C=$g` registration, apply the remaining cache value to a tile.
[^13]: Manifest-pinned demo-1, tv-1 and tv-2 captures at the outbound timestamps in the controls table, subsequent own position, absolute fuel and landing messages. `src/tankpit_bot/physics/costs.py`, `teleport_cost`, computes integer Euclidean cost; the controls compare its result with the observed charge before pickup.
[^14]: `src/tankpit_bot/browser/cdp_service.py`, `_record_frame`, calls `CDPClock.to_unix_ms`; `src/tankpit_bot/browser/cdp_utils.py`, `CDPClock`, anchors the first CDP timestamp to local wall time and reuses that offset within the session.
[^17]: `src/tankpit_bot/state/fuel_confidence.py`, `containers_after_enemy_refuel`, called by `state/tank_mutations.py`, `apply_tank_observation`; `state/container_mutations.py`, resource refresh and pickup functions; `sniffer/world_state_tiles.py`, `update_cache_tiles`; `sniffer/world_state_containers.py`, `update_world_state_from_container_pickup` and `update_world_state_from_fuel_total`; `bot/ai/equipment_search.py`, fuel selectors; `bot/ai/equipment.py`, `is_fuel_lock_release_warranted`; `bot/ai/larder.py`, `select_fuel_larder_hop`; `bot/ai/collect_hops.py`, `desperation_fuel_hop`; `bot/ai/escape_policy.py`, `escape_hop_permitted`. Executable cases: `tests/world_state/test_fuel_confidence.py` and `tests/bot/ai/test_suspect_fuel.py`.
[^16]: Private tv-1 capture messages 1612/1613/1627/1664 and tv-2 messages 1439/1440; all referenced observations are decoded gameplay records. The enemy's short status has `fuel=None`, `rank=6`, and `damage_state=3` immediately after its position record has `damage_state=2`. `src/tankpit_bot/bot/ai/combat_break.py`, `assess_engagement_break`, documents and uses the fuel-quartile interpretation; [[deactivation-format]] § SOLVED (2026-07-23): there is no healing — the damage tier IS the fuel quartile records the historical 19,658-pair fit. The consumption attribution is an inference from position and improving fuel tier, not an actor-bearing pickup receipt.
[^15]: Manifest-pinned captures: tv-1 commands 1587/1676 and radar messages 1592/1681; tv-2 command 1387, all outbound messages through 1482, and radar messages 1422/1480; demo-1 command 553 and response 556; tv-3 landing 1289 and zero updates 1291/1292. All five streams were checked over 09:48:40–09:49:20 for the fatal target and over the stated per-case windows. Archived `tpclient.js`, radar handler `ch` and cache handler `$g.prototype.h`, apply tile values without awarding fuel. The existing wiki contracts are [[radar-mechanics]] § The radar response is a delta sync (reveals + corrections, unchanged omitted) and [[fleet-coordination]] § The exchange, whose shared-scan paragraph records the August 25 team-reveal ruling; capture locators here provide the new primary evidence.
