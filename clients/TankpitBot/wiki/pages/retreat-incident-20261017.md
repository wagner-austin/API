---
title: Retreat failures in the 2026-10-17 World run
tags: [incident, combat, retreat, scheduling, fuel]
related: ["[[committed-intent]]", "[[bot-behavior-contract]]", "[[fuel-system]]", "[[fleet-coordination]]", "[[teleport-mechanics]]"]
fact_checked: "2026-10-17"
confidence: medium
hubs: [architecture]
verified: Offline inspection of preserved captures and release source; no live restart or behavior fix.
provenance:
  - "Corvis task d947f299; originating task 6275ecaa, handoff abd5bb45-eb63-4c67-88fe-470f1d411629."
  - "/home/corvis/tankpit-evidence-20261017/manifest.json: 20 files, 104395465 bytes; every size and SHA-256 reverified."
  - "Runtime API release 3888f06b22f5eea63d8e8cb5fe9934143ee7f392; tv-1 run bot-20261017-093852, tank 740 / Yuppler."
source_paths:
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
  "src/tankpit_bot/bot/tick_loop.py": "507c551e925ac7001eda6d4903964054ee451bb0"
  "src/tankpit_bot/bot/tick_body.py": "f89288ff6a4b37a57ad41520167638635f2a3d17"
  "src/tankpit_bot/bot/tick_loop_actions.py": "48ea79bf8ae71fcca61b91ffdf552f496fa116e1"
  "src/tankpit_bot/bot/world_sync.py": "902858477bc22a632a38b8a5b2b6618b08be02be"
  "src/tankpit_bot/bot/ai/combat_break.py": "295d58704a009d1f30fa2f09f1f6d0984c2bb6a7"
  "src/tankpit_bot/bot/ai/combat_prep.py": "b527aa154060e6a4dd4f1bce107a470d83ee9f69"
  "src/tankpit_bot/bot/ai/collect_mode.py": "fce9a51df3f7165c76933a4ecc04e64613e47b9b"
  "src/tankpit_bot/bot/ai/collect_mode_outcomes.py": "c8d40965837f5e76fa5c62b58ce1663f6df8c57d"
  "src/tankpit_bot/bot/ai/collect_pickups.py": "25c0f283d658c0de2e45fa7178fab21c9bfc7387"
  "src/tankpit_bot/bot/ai/movement.py": "c4df95be78254ee18e2a1d670f4b509a0551ddac"
  "src/tankpit_bot/bot/ai/larder.py": "cd075da13ac7792a058008d094df9d48b17c5e2c"
  "src/tankpit_bot/bot/ai/collect_hops.py": "854b07cd406df844615f84a542987305fd417153"
  "src/tankpit_bot/bot/executor.py": "d2430383d3e214d08ce64a5c88d54f649962a620"
  "src/tankpit_bot/state/container_mutations.py": "a6e0c51208b0c438bbe56ef1139d1ac0917bfaba"
  "src/tankpit_bot/fleetshare/report.py": "cfc0608da47dc6e2b559ecd46d6248a054ccdce9"
  "src/tankpit_bot/replay/engine.py": "7a1333337901d19972a617c3d59e8e20d4efcc03"
---

# Retreat failures in the 2026-10-17 World run

The engagement break activated. Its delegated action sequence failed to secure fuel or leave both attackers, and the control loop added measurable delay after responses. This is an offline incident diagnosis, not a deployed correction or proof that a particular alternative would have survived. All timestamps below are UTC on October 17; event line numbers are one-based, capture message indexes zero-based.[^1]

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

**The missing 580 fuel is not explained by an old fleet report alone.** Yuppler saw 580 via radar (capture message 1592, 09:48:57.711). Its later belief used tv-2's radar-processing timestamp 09:48:58.672; tv-2 had received that radar at 09:48:57.719. The import preserved that timestamp rather than refreshing it every tick. More decisively, tv-2's viewport patch still advertised 580 at 09:49:13.726, and Yuppler's own landing patch advertised 580 at 09:49:15.762. Using viewport origin plus `(col−1,row−1)` places both entries at (148,249). Yuppler then received fuel=256, a landing confirmation, fuel=256 again, and two remaining-volume-zero receipts. The sibling also received a removal receipt. These facts establish zero credited gain and a removed target, but identify neither a competing consumer nor the server-side reason. Do not invent a gain from the removal or blame a sibling without evidence.[^3]

## Confirmed control-flow gaps

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

The following are proposed corrections, not changes made by this investigation. Preserve never-drop combat locks, post-mine movement doctrine, the separate confirmed map-open phase, and displaced-landing sensing while fixing their composition.[^4][^5][^6][^7]

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
