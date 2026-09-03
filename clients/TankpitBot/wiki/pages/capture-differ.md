---
title: Capture Differ (sim-fidelity pipeline)
tags: [sim, protocol, tooling, verification]
related:
  - "[[game-economy]]"
  - "[[physics-module-roadmap]]"
  - "[[viewport-shift-protocol]]"
  - "[[ferry-mechanics]]"
source_paths:
  - "scripts/build_sim_baseline.py"
  - "scripts/analyze_response_shapes.py"
  - "analysis_scripts/mine_container_atlas.py"
  - "analysis_scripts/diff_server_laws.py"
  - "runs/analysis"
source_git_blobs:
  "analysis_scripts/mine_container_atlas.py": "b94d5f5fd8ee4369d02a9a8d50143ed059688067"
  "analysis_scripts/diff_server_laws.py": "7bb83c8dd8aee438f9ad0e3da1f00dc01ef2829b"
fact_checked: "2026-09-02"
confidence: high
hubs: [protocol]
---

# Capture Differ — the sim-fidelity pipeline

The systematic answer to "how close is the sim to the real server, and
where exactly does it diverge?" (user direction, 2026-08-01:
"structurally parse the run log ... and then we can systematically
identify divergences"). Three stages, all built and run 2026-08-01
against the full archive (318 real captures, 120 days).[^1]

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
empty misreads the exposure law as regeneration.[^2]

## Stage 2 — atlas world seeding

`sim/atlas_seed.py` + `tankpit-sim-run --from-atlas [PATH]`
(composes with `--practice`): the sim seeds the REAL mined room
instead of the statistical density model — 1,969 stocked fuel
containers, 6,675 drained dots, 498 equipment on field01, water tiles
included (the validator now accepts floating containers: the mined
population proves they are real state, and ferries drift — only rock
is a typo). First roster-on-real-field soak: 400/400 rounds, 8 kills,
0 deaths.[^3]

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
archive-wide).[^4]

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
0x3D-before-landed order and the 0x5A skip-walk entity sort — see
[[ferry-mechanics]].)

### Open divergence triage (remaining rows, bucketed)

| Shape (live-only) | n | Bucket |
|---|---|---|
| move ``47self+5A+3Dself`` | 392 | KNOWN OPEN — autoscroll-ON edge recenter (the queued riding build, [[viewport-shift-protocol]]) |
| move ``52c6`` (ALREADY_THERE) | 91 | MODEL LIMIT — the sim's instant single-tick movement has no in-transit state to re-click |
| equipment pickup ``67+49+pickup`` (no 0x47) | 894 | OPPORTUNITY? — own-tile pickups draw no movement echo; verify the sim's own-tile path emits none |
| assorted ``(silent)`` windows | — | ARTIFACT — response latency past the next command send truncates the window |
| radar ``...+pickup+pickup`` | 2,006 | ARTIFACT — landing-scan windows absorb the teleport's own pickup records |

Sim-only rows (shapes the LIVE archive never produces — post-lift
baseline ``sim-lift*``, 2026-08-03):[^4]

| Shape (sim-only) | n | Bucket |
|---|---|---|
| teleport ``5A+3Dself+landed+67+49+pickup`` | 13 | **INVENTED LAW — CONFIRMED AND REMOVED 2026-09-01.** The row asked for one byte-mined live equipment-hop landing window; the archive supplied 10,619. See below. |
| shoot ``53self+landed`` | 8 | ARTIFACT — sim queue compression resolves a queued teleport inside the shot's window |

### The teleport equipment grant: an invented law, settled 2026-09-01

The suspected-invented-law row above is closed, against it. Two
archive-wide window sweeps, method as Stage 3 (pair each SENT command
with the received messages before the next SENT command):

| Question | Answer |
|---|---|
| Teleport windows carrying a 0x67 EquipmentGain | **0 of 10,619** |
| Total 0x67 gains in the archive | 5,409 |
| Gains whose most recent sent command was `pickup_equipment` | **5,205 (96.2%)** |
| Gains whose most recent sent command was `teleport` | 1 |

**The law: a teleport landing auto-picks FUEL ONLY. Equipment requires
the explicit `pickup_equipment` command.** The single teleport-adjacent
gain is attribution noise of the "most recent sent command" heuristic,
which also credits 11 gains to `shoot` and 2 to `map_open`; against
10,619 teleports drawing zero it carries no weight.

This is not "the bot rarely lands on equipment". Landings demonstrably
DO auto-pick fuel — the duplicate-record law measures it in 31% of
7,176 live teleports — so an equipment auto-pick would have surfaced.

The sim granted it anyway, and `test_teleport_landing_collects_equipment`
PINNED the invention. That is how it survived a divergence-zero soak:
the sim and its test agreed with each other about a server neither had
asked. Removed from `sim/server_move.py`; the test now pins the
measured pairing (fuel yes, equipment no) instead.

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
starve on observation-sparse worlds.[^5]

The ``ghost_summary`` diagnostic is the measurement: how many rounds
the live bot stayed within 4 tiles of the recorded client, the first
divergence tick, and the final drift. Receipts: the self-replay
validation (current code vs its own sim recording) tracked 19/21
rounds; replaying the 2026-07-29 live session against TODAY'S bot
diverges at round 2 — the recorded bot wandered off collecting while
today's immediately acquired and killed orange-2, a truthful measure
of three days of behavior change. A 400-round live-capture replay
runs in ~120 s.[^5]

Standing baseline (2026-08-02, run bot-20260802-205105 — the 20-kill
soak): replaying the run's own capture under the same code tracked
10/150 rounds, first divergence round 2, final drift 53. A LIVE
capture self-replay diverges far earlier than a sim-capture
self-replay (19/21 above) because the seeded world is the atlas
approximation, not the live field: the first radar answers
differently and the hunt forks from there. Compare future ghost
numbers against THIS baseline, not against the sim-recording one.[^6]

Honest limits (v1, documented in the module): off-viewport ghosts
freeze between sightings; ghost fuel is unobservable (they seed at
rank capacity); a ghost the live bot kills early stops consuming its
timeline (though a BOT-named one now reactivates by the corpse law).[^5]

## Stage 5 — the bot-policy differ + reactive ghosts (2026-08-03)

The roster policy was already archive-mined (2026-07-24,
``sim/bot_policy.py``); stage 5 re-judged it against the GROWN
archive (310 sessions, 29 bot-hours) and closed the loop:[^7]

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

## The baseline is the measurement (2026-09-02)

A response-shape verdict is only as good as the sim corpus behind it,
and until 2026-09-02 that corpus was `runs/sim` — which is not a
corpus. It is a **graveyard**: `tankpit-sim-run` has always archived
there, so it accumulates every generation of the sim ever run. On
2026-09-02 it held 91 sessions of which **76 predated that morning's
fixes**, and the differ dutifully reported **36 invented laws** most of
which existed in no code path any more (`shoot 53self 49` n=709,
`radar 4F 46 49` n=601, `teleport landed 3Dself 5A` n=302 — all fixed
weeks earlier, all still being reported). A verdict taken over that
directory describes the union of every sim ever written.

### The sim is byte-deterministic, so repetition buys nothing

Three sessions of the same scenario run minutes apart produced
**identical wire** — 168 messages, the same payloads in the same
order, the same file size to the byte. That kills the obvious way to
enlarge a baseline: running one scenario N times adds N copies of one
sample and no information at all. The 91-session archive was never 91
samples either.

A baseline therefore widens by **scenario** and deepens by **rounds**,
never by repetition. `scripts/build_sim_baseline.py`
(`make sim-baseline`) sweeps five scenarios that drive genuinely
different command vocabularies out of the production bot — `duel`,
`solo`, `practice`, `ferry`, `human` — into a stamped directory that
has never held anything else, then diffs exactly those sessions.[^10]

### First one-generation verdict

| | mixed `runs/sim` (91 sessions) | fresh baseline (5 scenarios) |
|---|---|---|
| sim windows | 11,805 | 527 |
| invented laws | **36** | **4** |
| largest invented row | n=709 | **n=1** |

Every invented row above n=1 disappeared, which is the measurement
that the 36 were historical. The four survivors are all single-sample
queue-compression artifacts (a chat window that swallowed the radar
answers; a fuel-pickup window that swallowed a trailing 0x5A).

**Across 527 command windows from the current sim, every response
shape it produces more than once also occurs in the 73,053-window live
archive.**

### What the fresh baseline immediately caught

The first run reported `teleport  3Dself landed` at **n=31** — a
teleport confirm with no leading 0x5A. The live archive does not
contain that shape once in 10,683 teleport windows: every live confirm
reads `5A 3Dself landed` (56.7%) or `5A 3Dself landed pickup pickup`
(38.5%).

Attributing it by scenario put all 31 in `practice`, the only scenario
whose bots teleport off — and the cause was a recipient-policy hole:
`narrate_teleport` gated its refusal and blocked branches on the actor
but **narrated every successful landing to every observer**. On the
real wire TeleportLanded is per-recipient (10,541 arrivals against
10,683 own commands, **zero** zero-trigger — see [[recipient-policy]]),
and a foreign tank's new tile reaches a client from the end-of-tick
membership diff, so stating it here doubled it as well. Gating the
landed branch on the actor took the invented rows from 7 to 4 and the
row itself to zero. The auto-pick records still broadcast.

The tests had refusals and blocked hops pinned for both observers and
the successful landing pinned only for the actor — so nothing failed.
That is the same shape as every other miss recorded here: **a check
that cannot observe its own failure.**

### The missing side WAS the differ's fault, partly (2026-09-02)

The first reading of the fresh baseline reported 208 missing rows and
concluded the differ was blameless — the corpus was just too small.
That was half wrong, and the half that was wrong is the interesting
half: **a flat missing list reports three unrelated phenomena as one
number.** Splitting 208 rows by why each is one-sided:

| cause | rows | live windows | what it implies |
|---|---:|---:|---|
| live window drew nothing | 9 | 14,118 | **not a gap.** Timing |
| command never sent by the sim corpus | 77 | 4,647 | fix the scenarios |
| shape holds a token the sim never emits here | 92 | 3,555 | check: gap or window overlap |
| the sim emits every token, never this combination | **30** | 2,798 | **read these first** |

The first bucket is a **property of the two servers' clocks, not of
the sim's laws.** The real server is asynchronous: a command whose
answer arrives after the next command has opened its own window
records against that one and leaves the first silent. A
tick-synchronous sim answers inside the same batch and so cannot
produce a silent window for a command it handles. Those 9 rows carry
56% of the missing mass and can never be closed by any amount of sim
work.

The second bucket is real but is a CORPUS defect, not a server one —
the baseline never sent those commands at all. Notably it never sent
a plain `move`: the five scenarios drive the bot entirely through
pickup clicks and teleports.

Only the fourth bucket is readable as "the real server assembles
something the sim does not", and it is 30 rows, not 208. The report
now prints these groups with those headings, and leads with the
INVENTED side, which is the half that means something at any sample
size.

### The 30 readable rows are mostly one coverage hole

| n | command | shape |
|---:|---|---|
| 1,324 | `pickup_equipment` | `67 49 pickup` |
| 616 | `radar` | `4F 46` |
| 366 | `pickup_fuel` | `44 pickup 52c5` |

All three are the **no-walk / no-consumption variant** of a command
the corpus only ever exercised in its walking, consuming form: an
equipment pickup with no `47self` walk echo, a radar that consumed no
extra (`narrate_radar` gates the leading `0x49` on `consumed_extra`),
and the own-tile fuel-pickup branch `narrate_fuel_pickup` implements
exactly. The sim has all three branches; the baseline bot never
stands on what it collects. Those three rows are 2,306 of the
bucket's 2,798 live windows.

So the answer to "why can the missing side not be read yet" is three
concrete things, in descending value: **bucket the rows by cause**
(done), **make a scenario where the bot collects from its own tile**
(the single highest-value corpus fix), and **depth on the one
scenario that sustains**.

### The 92 token-never-emitted rows, read (2026-09-02)

The bucket was built and then not read. Reading it: for every novel
token, what had the client sent IMMEDIATELY BEFORE the window that
carried it. The real server is asynchronous, so a slow answer to
command N lands inside command N+1's window; if the novel token
belongs to the previous command, that is what happened.

| command | novel token | n | preceded by |
|---|---|---:|---|
| radar | `pickup` | 2,690 | **teleport 75%** |
| teleport | `4C` | 195 | **map_open 100%** |
| map_open | `53self` | 46 | **shoot 95%** |
| shoot | `52c3` | 45 | map_open 88% |
| shoot | `52c0` | 47 | shoot 74% |
| pickup_equipment | `52c4` | 87 | pickup_equipment 67% |

**Roughly 3,000 of the bucket's 3,555 windows are spill**, not gaps:
the bot teleports and immediately scans on landing, so the landing's
auto-pick records arrive after the radar's answers; it opens the map
and immediately hops, so the 0x4C lands in the teleport's window.
Those rows can never be closed and should not be chased.

The `52c*` rows are different, and two of them turned out to be real
laws.

### Three shot refusals the sim could not produce

Decoding the codes ([[decode-coverage]]): `52c0` is CANT_DO and
`52c3` is FRIENDLY_FIRE — both are the server REFUSING a shot, and
the sim's entire shoot vocabulary was `{53self}`. It had no refusal
path at all.

Both are already load-bearing on the production side:
`bot-behavior-contract.md` records that a shot-rejecting 0x52 ends
the feedback wait immediately, and `_disprove_target_by_friendly_fire`
treats code 3 as the one unfakeable proof that an id no longer
resolves to an enemy — written after a ghost drew 43 consecutive
rejected shots. **That path had never been driven end to end**,
because no sim session could produce the message it consumes.

Field values, swept 2026-09-02 across every 0x52 in a shoot window
and INVARIANT in each:

| code | meaning | n | `(reset_action, close_map)` |
|---|---|---:|---|
| 0 | aim outside the viewport | 47 | **(0, 1)**, 47/47 |
| 3 | shot at a teammate | 45 | **(1, 0)**, 45/45 |

Implemented as `physics.supervisor.shot_refusal`, beside the fuel,
equipment and teleport refusal laws. The sweep also showed code 0 is
`(0, 1)` for EVERY command that draws it (83 of 86 archive-wide) —
the sim's move-family code-0 used `(1, 0)`, a field-level divergence
the differ is structurally blind to because it tokenizes a 0x52 to
`52c<code>` and discards the rest.

NOT implemented: code 8 on a shot. The contract lists it among the
shot-rejecting codes, but the archive holds ZERO shot windows
carrying it.

### The larder scenario, and the law it flushed out (2026-09-02)

The three biggest `shape_never_assembled` rows were all the no-walk
or no-consumption variant of a command the corpus only ever
exercised walking and consuming. `make_larder_sim_world` targets
them: the client spawns ON equipment, ringed by more, with all five
slots at ZERO.

First run: `radar 4F 46` **HIT**, `pickup_equipment 67 49 pickup`
**MISS** — every grant still carried a `47self`. The sim resolves an
own-tile click as a "moved" outcome with an EMPTY path and echoed it
anyway.

Measured rather than assumed: tracking each live capture's own 0x3D
position and finding every command clicked at exactly that tile —
**1,042 of 1,044 own-tile `pickup_equipment` clicks drew NO echo.**
(`pickup_fuel` reads 23 silent to 9 and plain `move` 22 to 21; those
residuals are consistent with a stale tracked position, but only the
equipment ratio stands alone.) **No movement, no echo** is now the
sim's law, and both target shapes appear.

It also settled an older disagreement: two fuel-choreography tests
had docstrings naming the measured shape `44+pickup+52c5` and
assertions demanding a leading `0x47` that shape does not contain.
The prose carried the law, the expectation carried the invention, and
nothing compared them.

Not reachable, and correctly so: `pickup_fuel 44 pickup 52c5` (366
live windows) is a full-tank own-tile fuel click. The bot has
predicted that refusal before dispatch since 2026-08-03, so those
windows were drawn by a SUPERSEDED bot. **The live archive is a
mixture of bot generations too** — the same graveyard property as
`runs/sim` — and the differ cannot tell "the sim lacks this law" from
"the current bot never asks for it."

### Depth is not a lever: the sim's vocabulary SATURATES

**The first version of this section was measured wrong and its
numbers are retracted.** It varied session depth and the run STAMP
together, and `select_practice_layout` picks the practice world from
`crc32(stamp)` — so every row played a different world and none of it
said anything about depth. The retracted series reported a
non-monotonic window count (a 3,200-round run producing 684 windows
against a 1,600-round run's 1,598) and a shape count still climbing at
6,400. Both were the confound.

Re-measured with the stamp HELD CONSTANT, so depth is the only thing
that varies:

| rounds | windows | distinct shapes |
|---:|---:|---:|
| 200 | 202 | 11 |
| 400 | 396 | **15** |
| 800 | 782 | **15** |
| 1,600 | 1,542 | **15** |
| 3,200 | 3,106 | **15** |
| 6,400 | 6,286 | **15** |

**Sixteen times the windows, zero new shapes.** Two controls ran in
the same pass and both hold: replaying 800 rounds at the same stamp
reproduced the session identically, and **3,106 of 3,106** windows of
the 3,200-round run match the 6,400-round run's prefix exactly — a
longer run IS the shorter run plus more, with nothing carried between
sessions in the process.

So the sim's response-shape vocabulary is COMPLETE at ~400 rounds of
the sustaining scenario, and "the sim never produces this shape" is a
safe claim for any command the corpus sends. What is missing is not
volume: it is BRANCH COVERAGE. The bot never stands on what it
collects, never runs a radar with no extra to spend, never sends a
plain `move`.

**Retracted with the old numbers:** the claim that "windows per round
collapses past 1,600 rounds, so the bot goes quiet late in a long
session". At a fixed stamp the window count tracks the round count all
the way to 6,400. There was no such effect.

### The stamp is an INPUT to the practice world, not a label

`select_practice_layout(stamp)` returns
`PRACTICE_LAYOUTS[crc32(stamp) % len(PRACTICE_LAYOUTS)]`. The stamp is
also the artifact filename, so **naming a run changes what it plays** —
and only for the practice scenario, which is why three default-scenario
captures with different stamps compared byte-identical earlier the same
day and made "the sim is deterministic" look unconditional.

It is deterministic given identical inputs INCLUDING the stamp. The
variety is deliberate and useful, but it is undeclared at the API
level and it welds two knobs together: there is no way to ask for the
same world at a different depth, or a different world at the same
depth, without also moving the other. Any sweep — especially a
cluster array where each task stamps itself — is choosing worlds by
accident unless it states the layout explicitly.

## How to re-run

```
poetry run python analysis_scripts/mine_container_atlas.py
poetry run python analysis_scripts/analyze_container_atlas.py
poetry run python analysis_scripts/mine_deposit_attribution.py
# the fidelity verdict: generate a one-generation corpus and diff it
make sim-baseline
# re-read an existing baseline without regenerating it
poetry run python -m scripts.analyze_response_shapes runs/sim-baseline/<stamp>
```

Never diff against bare `runs/sim` for a verdict — see "The baseline
is the measurement" above.

Every future live session extends the atlas and the shape corpus for
free; every sim law change re-verifies against 38,000+ recorded
command windows.[^4]

Both mining scripts were re-plumbed 2026-08-06 onto
`tankpit_bot.analysis.scan` (`scan_session`, the typed capture-scan
owner with direction-tagged frames); each script's private
load/XOR/frame-walk pipeline is deleted. The commands above are
unchanged; the migration note in `analysis_scripts/mine_container_atlas.py:41-45`
records that results reproduce exactly, and sent-frame room SELECTs
and text ROOM_LIST rows now read `frame["raw"]` (never ciphered), as
the production receive path does. The numbers on this page date from
before that migration and have not been re-derived under it.[^8]

## Tick-paced sim sessions (2026-08-01)

`run_sim_session` now paces the decision clock at the measured 2 s
server tick (`TickPacedClock`) instead of wall time. Wall-clock
sessions ran TTL dynamics ~1000x fast solo and load-dependently under
parallel test runs (the same 600-round session exited at round 54
solo and never under xdist); tick pacing makes every sim session
deterministic AND live-realistic — a 300-round soak ages its forage
coverage, harvest memory, and belief freshness exactly like a
10-minute live session, and the captures it writes carry live-shaped
2 s timestamps the differ can window.[^9]

[^1]: Pipeline entry points on disk, all three blob-pinned or path-pinned in this page's frontmatter: `analysis_scripts/mine_container_atlas.py` (stage 1), `analysis_scripts/diff_server_laws.py` (stage 3), and the `runs/analysis/` artifact directory the stages write. Verified present 2026-08-07.
[^2]: `analysis_scripts/mine_container_atlas.py` — the ordering and visible-layer-zero rules are properties of the mining pass itself; the migration note at `:41-45` records that the 2026-08-06 re-plumb onto `tankpit_bot.analysis.scan` reproduces results exactly, which is the check that would have caught a reordering regression.
[^3]: `src/tankpit_bot/sim/atlas_seed.py` — the atlas-seeded world builder invoked by `tankpit-sim-run --from-atlas`. Verified present 2026-08-07.
[^4]: `analysis_scripts/diff_server_laws.py` — the window-pairing differ that produces both the live-only and sim-only shape tables above. Blob-pinned in frontmatter; verified present 2026-08-07.
[^5]: `src/tankpit_bot/sim/ghost.py` — the capture-to-spec compiler behind `tankpit-sim-run --ghost`. Ghost relocation by recorded authority is `SimServer.relocate_tank` at `src/tankpit_bot/sim/server.py:185`. Verified present 2026-08-07.
[^6]: **Point-in-time measurement, no committed artifact.** The 10/150-round / first-divergence-2 / drift-53 baseline was read from a ghost replay of `runs/bot/bot-20260802-205105` on 2026-08-02. The capture is on disk; the replay summary is not, so re-run rather than carry the numbers forward.
[^7]: `src/tankpit_bot/sim/bot_policy.py` — the archive-mined roster policy. The reactive-ghost driver is `PracticeRoomDriver` at `src/tankpit_bot/sim/practice_room.py:84`, pinned by `tests/sim/test_ghost_reactive.py`. Verified present 2026-08-07.
[^8]: `analysis_scripts/mine_container_atlas.py:41-45` — the migration note recording that both mining scripts now read through `scan_session` (`src/tankpit_bot/analysis/scan.py:121`) and that results reproduce exactly.
[^10]: `scripts/build_sim_baseline.py` — the one-generation baseline builder behind `make sim-baseline`; its scenario matrix is `SCENARIOS` and the per-run archive directory reaches `run_sim_session` through the `--out` flag (`src/tankpit_bot/sim/run.py:80`). The determinism measurement compared `runs/sim/sim-20260902-155056`, `-155100` and `-155104` payload-by-payload. Verified present 2026-09-02.
[^9]: `run_sim_session` at `src/tankpit_bot/sim/run.py:80`, paced by `TickPacedClock` at `src/tankpit_bot/sim/run_boot.py:57`. Verified present 2026-08-07.
