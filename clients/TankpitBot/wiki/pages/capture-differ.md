---
title: Capture Differ (sim-fidelity pipeline)
tags: [sim, protocol, tooling, verification]
related:
  - "[[game-economy]]"
  - "[[physics-module-roadmap]]"
  - "[[viewport-shift-protocol]]"
  - "[[ferry-mechanics]]"
source_paths:
  - "analysis_scripts/mine_container_atlas.py"
  - "analysis_scripts/diff_server_laws.py"
  - "runs/analysis"
fact_checked: "2026-08-01"
confidence: high
hubs: [protocol]
---

# Capture Differ — the sim-fidelity pipeline

The systematic answer to "how close is the sim to the real server, and
where exactly does it diverge?" (user direction, 2026-08-01:
"structurally parse the run log ... and then we can systematically
identify divergences"). Three stages, all built and run 2026-08-01
against the full archive (318 real captures, 120 days).

## Stage 1 — the longitudinal container atlas

`analysis_scripts/mine_container_atlas.py` +
`analyze_container_atlas.py` + `mine_deposit_attribution.py`: every
per-tile container statement from every capture, cross-session
ordered. Findings in [[game-economy]] (persistence 94.9%+ over a
month; refills are discrete agent deposits, corr(Δv,Δt)=−0.13;
stocked population ~5,000+ vs the model's ~670). Artifacts:
`runs/analysis/container_atlas.json`, `container_observations.jsonl`,
`container_refills.json`.

**Mining discipline learned the hard way:** (1) intra-payload wire
order must be preserved — one tick's batch shares a timestamp, and
value-sorted ties manufactured 123 phantom "refills" out of a
pickup's (pre-read, remaining) pair; (2) a visible-layer 0 (0x5A/0x43
cache) means "no VISIBLE container", never "empty" — treating it as
empty misreads the exposure law as regeneration.

## Stage 2 — atlas world seeding

`sim/atlas_seed.py` + `tankpit-sim-run --from-atlas [PATH]`
(composes with `--practice`): the sim seeds the REAL mined room
instead of the statistical density model — 1,969 stocked fuel
containers, 6,675 drained dots, 498 equipment on field01, water tiles
included (the validator now accepts floating containers: the mined
population proves they are real state, and ferries drift — only rock
is a typo). First roster-on-real-field soak: 400/400 rounds, 8 kills,
0 deaths.

## Stage 3 — the response-shape differ

`analysis_scripts/diff_server_laws.py`: pairs every SENT command in
every capture with the self-caused messages inside its window (window
ends at the next sent command), reduces them to an ordered shape over
the self-caused alphabet (0x53/0x47/0x3D self, landed, 0x52 codes,
0x4F/0x46, 0x5A, pickup records, 0x49/0x44/0x67/0x4C/0x4D), and
diffs the live distribution against a fresh sim baseline. Numeric
laws ride the pass: teleport cost (6,941/6,960 live windows within
displacement tolerance of floor(6×euclid)) and the window-bound
acceptance law (77 outside-window sends, 0 accepted — perfect
archive-wide).

### Law gaps found AND closed by the first two cycles (2026-08-01)

| # | Divergence (live vs sim) | Fix |
|---|---|---|
| 1 | Landed teleport order: live ``5A -> 3D -> landed`` (69% of 7,176); sim emitted the recentered 0x5A LAST | server splices the recenter 0x5A ahead of the landing messages |
| 2 | Every client 0x5A pairs with a self 0x3D (the corpus 22:22); sim scope answered a bare 0x5A | scope emits ``5A + 3Dself`` |
| 3 | Radar extra-consumption snapshot LEADS: live 84% ``49+4F+46``; sim trailed it at tick end | inline 0x49 before 0x4F |
| 4 | Firing costs never snapshot: live 92.4% of 11,051 shots answer a bare 0x53; sim sent a 0x49 per shot | shot ammo changes no longer snapshot; counts re-sync on the next 0x49-bearing event |
| 5 | Equipment pickups close with a container-pickup record: live ``47+67+49+pickup``; sim omitted the record | grant emits 0x67, 0x49, then the remaining-0 pickup record |
| 6 | The fuel-pickup choreography (~1,600 windows: ``47+pickup+pickup+44+pickup+52c5`` and variants) — the sim emitted one bare record | byte-mined 2026-08-01 ([[fuel-system]]): four measured branches in `emit_fuel_pickup_close`, duplicate-record law on all auto-picks, the two 0x44 forms, walks executing for known-drained containers |

(Plus the two the ferry soak caught the same morning: the
0x3D-before-landed order and the 0x5A skip-walk entity sort.)

### Open divergence triage (remaining rows, bucketed)

| Shape (live-only) | n | Bucket |
|---|---|---|
| move ``47self+5A+3Dself`` | 392 | KNOWN OPEN — autoscroll-ON edge recenter (the queued riding build, [[viewport-shift-protocol]]) |
| move ``52c6`` (ALREADY_THERE) | 91 | MODEL LIMIT — the sim's instant single-tick movement has no in-transit state to re-click |
| equipment pickup ``67+49+pickup`` (no 0x47) | 894 | OPPORTUNITY? — own-tile pickups draw no movement echo; verify the sim's own-tile path emits none |
| assorted ``(silent)`` windows | — | ARTIFACT — response latency past the next command send truncates the window |
| radar ``...+pickup+pickup`` | 2,006 | ARTIFACT — landing-scan windows absorb the teleport's own pickup records |

Sim-only rows (shapes the LIVE archive never produces — post-lift
baseline ``sim-lift*``, 2026-08-03):

| Shape (sim-only) | n | Bucket |
|---|---|---|
| teleport ``5A+3Dself+landed+67+49+pickup`` | 13 | SUSPECTED INVENTED LAW — the sim grants equipment on teleport landing, but live's dominant explicit-pickup shape is own-tile ``67+49+pickup``, suggesting the real server auto-picks only FUEL on landing and equipment needs the explicit command. Needs one byte-mined live equipment-hop landing window before changing the sim. |
| shoot ``53self+landed`` | 8 | ARTIFACT — sim queue compression resolves a queued teleport inside the shot's window |

## Stage 4 — ghost replay (2026-08-01)

`sim/ghost.py` + `tankpit-sim-run --ghost CAPTURE [--from-atlas]`:
one capture compiles into a replayable spec — the recorded client's
opening state (spawn, fuel, counts, rank, team), every sighted
opponent with its recorded name, a tick-indexed timeline of their
positions (0x3D + walk-corrected 0x47), shots (0x53 aim tiles), and
chats (0x4D — the consent signals replay), the session's own first
0x4C dot atlas as the exposed set, and its first-observed container
reads. The PRODUCTION bot then plays live against the recording:
ghosts relocate by recorded authority (`SimServer.relocate_tank` —
already-visible in-window moves re-state 0x3D; entries/exits ride
the membership diff), their shots and chats queue as real commands,
and damage resolves by sim law. ``--from-atlas`` underlays the mined
room beneath the capture's reads (per-tile capture truth wins; the
recording's dot atlas bounds the exposed set) so long replays don't
starve on observation-sparse worlds.

The ``ghost_summary`` diagnostic is the measurement: how many rounds
the live bot stayed within 4 tiles of the recorded client, the first
divergence tick, and the final drift. Receipts: the self-replay
validation (current code vs its own sim recording) tracked 19/21
rounds; replaying the 2026-07-29 live session against TODAY'S bot
diverges at round 2 — the recorded bot wandered off collecting while
today's immediately acquired and killed orange-2, a truthful measure
of three days of behavior change. A 400-round live-capture replay
runs in ~120 s.

Standing baseline (2026-08-02, run bot-20260802-205105 — the 20-kill
soak): replaying the run's own capture under the same code tracked
10/150 rounds, first divergence round 2, final drift 53. A LIVE
capture self-replay diverges far earlier than a sim-capture
self-replay (19/21 above) because the seeded world is the atlas
approximation, not the live field: the first radar answers
differently and the hunt forks from there. Compare future ghost
numbers against THIS baseline, not against the sim-recording one.

Honest limits (v1, documented in the module): off-viewport ghosts
freeze between sightings; ghost fuel is unobservable (they seed at
rank capacity); a ghost the live bot kills early stops consuming its
timeline (though a BOT-named one now reactivates by the corpse law).

## Stage 5 — the bot-policy differ + reactive ghosts (2026-08-03)

The roster policy was already archive-mined (2026-07-24,
``sim/bot_policy.py``); stage 5 re-judged it against the GROWN
archive (310 sessions, 29 bot-hours) and closed the loop:

* **Policy re-confirmed at scale**: 5,975/5,975 bot shots are
  weapon-0 singles, 95.7% inside the 3 s reflex window; rank
  teleport-off thresholds re-pinned (mode 7 at recruit / 87 samples,
  8 at private / 117); reactivation-gap mode exactly the 22 s corpse
  window. ``make shadow`` is green across all seven laws.
* **Two validators were judging our line of sight, not the law** —
  the differ's first catches were instrument bugs: ``sync-cadence``
  now medians CLEAN gaps only (absence holes excluded — 74/266 false
  mismatches were 2 s cores wrapped in 18-943 s viewport-exit holes)
  and ``bot-reactivation`` skips re-sights far past the corpse
  window (34/35 "failures" were bots damaged by third parties before
  drifting back into view). Residual true anomalies: one early
  (16 s) re-sync, 10 cadence outliers.
* **Reactive ghosts**: a bot-named ghost carries the certified
  policy UNDER its recorded timeline (``PracticeRoomDriver`` over
  ghost ids — the same driver, states over existing tanks). The live
  bot's shots draw the mined shot-for-shot return and team aggro
  even where the recording holds no answer; recorded events keep
  per-tick authority (held policy commands fire on the next quiet
  tick, pinned by ``tests/sim/test_ghost_reactive.py``); killed bot
  ghosts reactivate by the corpse-window law. Human-named ghosts
  stay pure recordings. Remaining: per-player human style models.

## How to re-run

```
poetry run python analysis_scripts/mine_container_atlas.py
poetry run python analysis_scripts/analyze_container_atlas.py
poetry run python analysis_scripts/mine_deposit_attribution.py
# fresh sim baseline (any stamps sharing a prefix), then:
poetry run python analysis_scripts/diff_server_laws.py "" sim-<prefix>
```

Every future live session extends the atlas and the shape corpus for
free; every sim law change re-verifies against 38,000+ recorded
command windows.

## Tick-paced sim sessions (2026-08-01)

`run_sim_session` now paces the decision clock at the measured 2 s
server tick (`TickPacedClock`) instead of wall time. Wall-clock
sessions ran TTL dynamics ~1000x fast solo and load-dependently under
parallel test runs (the same 600-round session exited at round 54
solo and never under xdist); tick pacing makes every sim session
deterministic AND live-realistic — a 300-round soak ages its forage
coverage, harvest memory, and belief freshness exactly like a
10-minute live session, and the captures it writes carry live-shaped
2 s timestamps the differ can window.
