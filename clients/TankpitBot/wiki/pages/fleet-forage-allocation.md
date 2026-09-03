---
title: Fleet Forage Allocation
tags: [fleet, coordination, foraging, architecture]
related:
  - "[[fleet-coordination]]"
  - "[[fleet-lifecycle]]"
  - "[[bot-behavior-contract]]"
source_paths:
  - "src/tankpit_bot/bot/tick_body.py"
  - "src/tankpit_bot/bot/tick_claims.py"
  - "src/tankpit_bot/fleetshare/claims.py"
  - "src/tankpit_bot/fleetshare/merge.py"
  - "src/tankpit_bot/bot/ai/context.py"
  - "src/tankpit_bot/bot/ai/tactics.py"
  - "src/tankpit_bot/fleetshare/types.py"
source_git_blobs:
  "src/tankpit_bot/bot/tick_body.py": "45addc1e7e7a2e82116c7bd4f4d177941a394f3d"
  "src/tankpit_bot/bot/tick_claims.py": "d29caeef6f472f7b45dcdc9658057aae94e54764"
  "src/tankpit_bot/fleetshare/claims.py": "167c63cd04e3e799177cdbe6c9d29386002dbbff"
  "src/tankpit_bot/fleetshare/merge.py": "bce1c259f0e79d5709f3634d30ec3686344d565d"
  "src/tankpit_bot/bot/ai/context.py": "7ab60480ed3b3d5e1faf2f0f8a1438b80edb4702"
  "src/tankpit_bot/bot/ai/tactics.py": "3e2e7bc64815c7687469a8afe3eaaadecf87a429"
  "src/tankpit_bot/fleetshare/types.py": "7bc23b7b1177ba508258d777be91da1cc93704de"
fact_checked: "2026-09-02"
confidence: high
hubs: [architecture]
---

# Fleet forage allocation: the fleet coordinates fighting, not foraging

*Measured 2026-09-02 off a five-bot World run, after the operator
reported it twice: "multiple bots go for the same fuel container. i
thougt we fixed this but i guess not."*

Both halves of that are right, and the second half is the interesting
one. The container layer WAS built and it is not broken. It solves a
different problem than the one being observed.

## The asymmetry

Laying the layers side by side is what makes the gap visible; each is
well documented on its own ([[fleet-coordination]]) and the pattern
across them is not.

| Layer | Knowledge sharing | Actual coordination |
|---|---|---|
| Enemies | sightings, freshness law, own-wire-outranks-remote | **yes** — focus fire via `engaged_target_id`[^1] |
| Combat timing | `war_ready` rows | **yes** — the swarm muster's quorum[^2] |
| Containers | atlas, tombstones, shared coverage | **yes since 2026-09-02** — advisory claim to steer planning, exclusive-create file mutex to arbitrate commitment[^3][^5] |

So the fleet already performs positive, peer-to-peer coordination —
for fighting. The swarm muster is a genuine dispatch-like behaviour
reached with no broker: bots sense a quorum through published
`war_ready` rows and strike together. Focus fire is the same shape.

Foraging is the odd one out. Its layer is excellent at **knowledge** —
tombstones stop the fleet chasing consumed containers, shared coverage
stops it re-scanning cleared ground — and holds no opinion about
**whose** a live container is. Five bots converging on one container is
therefore the layer working exactly as built: it told all five,
correctly and promptly, that the container was there.

## Why the claim cannot close it

A claim exists and is wired end to end: `tick_body` publishes
`collect_claim_x/y`, the merge collects them, and
`filter_fleet_claimed_containers` removes claimed tiles from the world
the planner sees.[^3] Nothing is unwired.

The defect is WHEN. The claim is gated on `resource_target_kind != ""`
— a bot claims only AFTER it has selected. The race window is
select → write report → sibling reads → sibling replans, and two
siblings planning in the same tick both see the tile unclaimed.

Measured across the five-bot run: **1,499 pickup dispatches, 273 tiles
where two different bots dispatched within 30 s, 1,160 such pairs, and
a MEDIAN GAP OF ZERO SECONDS.**[^4] A median of zero says this is not a
narrow race lost occasionally — siblings commit to the same tile inside
the same second, continuously. A protocol whose message arrives after
the decision cannot arbitrate a collision that happens before it.

The claim is not the wrong idea; it is an **advisory,
eventually-consistent** claim being asked to do an authoritative one's
job.

## What closed it (2026-09-02)

**An authoritative claim needs no broker**, because the filesystem is
already an arbiter: exclusive create (`O_CREAT|O_EXCL`) is atomic, so
first-writer-wins on one file per container tile is a true mutex. It
survives divergent knowledge (a bot claims only what it can see),
reaps a crashed bot's claims by a staleness horizon, and preserves the
single-tank rule — a solo bot creates claims nobody contends and
behaves identically.

Built exactly so:[^5] `fleetshare/claims.py` owns the protocol —
claim files at `runs/bot/_claims/<room>/<x>_<y>.claim` (the leading
underscore cannot collide with an instance directory by the instance
name grammar), a typed `ContainerClaimDict` body, and three laws:
existence is the lock while content is metadata (creation is atomic,
the content write is not, so an unreadable claim denies for one beat
and never a journey); the holder refreshes every full tick and a
stamp older than `CLAIM_TTL_MS` (30 s — sized to ride out the
measured 8 s receipt stalls that skip refresh ticks) is reaped by any
contender, the reap race itself re-arbitrated by the retry create;
and only the owner deletes. `bot/tick_claims.py` reconciles the claim
with the committed plan BETWEEN decide and execute: a plan that just
latched pays one exclusive create, and when a sibling won it the same
tick, the plan dies right there — `plan_released` reason
`claim_lost`, a hold command instead of the doomed dispatch, and the
tile remembered in the session's OWN denial memory
(`ws.claim_denied_tiles`, unioned into the planner's filter, expiring
at the claim horizon). The denial deliberately does NOT ride the
advisory claimed set: every merge pass replaces that set wholesale,
and a winner that crashes right after claiming never publishes an
advisory row at all — the double-check found that a stamped denial
died within the tick, leaving the loser to re-pick the dead-claimed
tile for one denied beat per tick for up to the 30 s horizon. The
local memory caps it at one beat. A session with no selected room passes
through unclaimed (the same scope law as `build_fleet_report`'s
pre-join return — no room, no fleet channel, nobody to contend
with), which is also what keeps the sim seam byte-identical.

**Deterministic assignment is the weaker option here**, recorded so it
is not re-proposed: it requires every bot to compute from the same
inputs, and they cannot. Each container set is one bot's viewport plus
merged sightings under a share horizon, so two bots with different
views compute different "identical" answers, and it fails silently and
asymmetrically.

Claiming solves collision avoidance — negative coordination. It can
never produce positive allocation (territory splits, "three push while
two forage"); that needs a plan the bots agree on, which is the one
thing peer-to-peer sensing has not been asked to carry.

[^1]: `engaged_target_id` in `FleetReportDict`; the merge collects
      siblings' locks into `ws.fleet_engaged_target_ids`
      (`fleetshare/merge.py`).
[^2]: `SWARM_MUSTER_QUORUM = 2` in `bot/ai/tactics.py`, fed by the
      `war_ready` report row. A sibling already fighting is joined
      with no quorum at all — reinforcement beats book-keeping.
[^3]: Published at `bot/tick_body.py` (gated on
      `resource_target_kind != ""`), collected in
      `fleetshare/merge.py` into `ws.fleet_claimed_containers`,
      consumed by `filter_fleet_claimed_containers`
      (`bot/ai/context.py`).
[^4]: Counted 2026-09-02 over `container_pickup_dispatched` records in
      `runs/bot/{artax,arterial,despair,malignant,yuppler}/latest.events.jsonl`
      (release v0.1.0-293f1ad7, World, 4x swarm + 1x passive).
      **Correction, recorded because the mistake is easy to repeat:**
      an earlier pass read `remaining_volume: 0` as "arrived at an
      empty container" and produced a 55%-wasted-trips figure. The
      field is documented at `container/decoders/events.py:85` as the
      fuel REMAINING AFTER the pickup, so zero means the pickup drained
      it — a success rate, not a waste rate. This page therefore
      measures CONVERGENCE only; wasted travel is not measured by any
      diagnostic currently emitted, and would need the collect plan's
      origin rather than its dispatch.
[^5]: `fleetshare/claims.py` (protocol: `acquire_container_claim` /
      `release_container_claim`, `CLAIM_TTL_MS`), `bot/tick_claims.py`
      (`_arbitrate_collect_claim`, wired in `bot/tick_body.py` step
      6b), `_test_hooks/fs.py` (`create_text_exclusive`, the
      O_CREAT|O_EXCL primitive). Events: `container_claim_acquired`
      on transition, `container_claim_denied` on loss, `plan_released`
      reason `claim_lost`. Every protocol arm is pinned in
      `tests/fleetshare/test_claims.py` and
      `tests/bot/test_tick_claims.py`; the real first-writer-wins
      primitive in `tests/test_test_hooks.py`.
