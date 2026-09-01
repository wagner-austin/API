---
title: "Silent Placement Refusal — Detection, Not Prediction"
tags: [engine, building, agent, wire, policy, refusal]
related:
  - "[[building-structures]]"
  - "[[wire-contract-ndjson]]"
  - "[[policy-economy]]"
  - "[[runtime-split-java-agent-python-brain]]"
source_paths:
  - "wiki/sources/m31-refusal/construction-attempt-blocked.txt:1"
  - "wiki/sources/m31-refusal/cant-afford-retry.txt:5"
  - "wiki/sources/m31-refusal/waypoint-removed-silently.txt:30"
  - "wiki/sources/m31-refusal/waypoint-resume-check.txt:4"
  - "agent/src/rwbot/agent/BuildWatch.java"
  - "src/rw_bot/policy/workforce.py"
  - "src/rw_bot/policy/runner.py"
source_git_blobs:
  "wiki/sources/m31-refusal/construction-attempt-blocked.txt": "eb9a904a74d2a88d8370f12fdf201520bf756d15"
  "wiki/sources/m31-refusal/cant-afford-retry.txt": "d902172470e3163005453f94b1aace347f9aa850"
  "wiki/sources/m31-refusal/waypoint-removed-silently.txt": "17b5f882b8aa7e1fc2b97a5af7ffdca3fc2ca13f"
  "wiki/sources/m31-refusal/waypoint-resume-check.txt": "60393dfcc0ea3bf3eaa0940f33cc3668b7b22273"
  "agent/src/rwbot/agent/BuildWatch.java": "7c234857a53641b306daf022d606bbfc26a43f48"
  "src/rw_bot/policy/workforce.py": "3ced6c8025c33d3d1406d679650d43086c740c7d"
  "src/rw_bot/policy/runner.py": "857d1e24c2cde142054e40c7bcb50b3a1d94b009"
game_version: "1.15 (code 176, build #28)"
fact_checked: 2026-09-01
confidence: high
hubs: [engine-internals, bot-architecture]
---

# Silent Placement Refusal — Detection, Not Prediction

A build order can pass every logged check and still build nothing. The validator refuses with a named reason ([[building-structures]]); the *construction attempt* refuses silently, and it is the stage that judges terrain. This is what emptied duel_lake's Hard panel — `expansions 64 (0 factories)` — before the ledger existed.

## Where the engine drops the order

When the builder reaches its waypoint, the tick calls the construction attempt `y.a(au, as, int, float, float)`.[^1] The attempt creates a **ghost instance** of the target type at the site, runs the blocked-pair overlap test against it, and — when blocked — destroys the ghost and returns a result whose only cargo is the blocking unit. No log line is written on this arm.[^2]

The caller then removes the waypoint whenever the result's retry flag `c` is false: `if (!z2.c) this.ay();`.[^3] The flag is set true in exactly one arm — can't-afford, which also plants a marker at the site — so *affordability is the only refusal the engine itself retries*. A blocked site is simply dropped.[^4]

## Detection, not prediction

Mirroring the engine's terrain rules to pre-check a site would mean reimplementing an obfuscated judgement that moves every release. Instead the agent watches what actually happened: `BuildWatch` records every dispatched build order and sweeps each sample. A structure of the ordered type standing at the site resolves the order; the waypoint still queued keeps it pending; the waypoint **gone with no structure appearing** is a refusal, reported on the wire.

The waypoint-identity rule is the engine's own resume check, not an invented one: kind `av.c`, build type by reference, coordinates within 10 world units.[^5] Orders the watch has not yet seen queued get a 90-frame grace before silence alone convicts them, so a slow first sweep does not misread an accepted order.

## One report, three effects

The refusal record travels in the world sample ([[wire-contract-ndjson]]) and does three things the tick it lands, all reading one shared ledger in the workforce:

1. **Enters the ledger** — deduplicated against the presumed-lost clock's entries, the slow second writer that stays as fallback for whatever the watch cannot see.
2. **Reopens the plan's slot** — judged against the site that was *ordered*, because every fresh decision has already moved past a ledgered site.
3. **Frees the worker** standing on the dead job — otherwise the retry waits out the 45-sample presumed-lost window the report exists to beat.

The site chooser reads the same ledger, so the retry goes to the next ring site. End to end: order, silent drop, report, retry elsewhere — three samples.

## Verified live, deterministically

`scripts/refusal_probe.py` fires the chain on demand: it orders a landFactory onto the player's own command centre — the one site the blocked-pair test can never pass, whatever the map or seed — and waits for the report. Against the real engine (duel_lake, seed 12345, lockstep 75, pinned+ff10) the record arrived two samples after the order, with the engine's own log carrying **zero** messages about it: `frame 150: engine refused landFactory at (990.0, 2010.0) for unit 24`.[^6] Run it as the play harness's module: `python -m rw_bot.harness.play_match_cli ... --module scripts.refusal_probe --play-args " "`.

[^6]: `runs/refusal-probe.log` (2026-09-01 run, regenerable by the invocation above) — the agent's dispatch line `channel: build landFactory by 24 at (990.0, 2010.0)` is followed by no validator or engine message; the probe's stdout carried the refusal record the same run.

[^1]: `wiki/sources/m31-refusal/construction-attempt-blocked.txt:1` — `public z a(au au2, as as2, int n2, float f2, float f3) {`. Excerpted from decompiled `runs/decompiled/com/corrodinggames/rts/game/units/y.java:4855` (gitignored; regenerate with `make decompile`).
[^2]: `wiki/sources/m31-refusal/construction-attempt-blocked.txt:15-30` — the pair `a((am)object, null)` / `a(true, null)` sets `bl2`, then `am3.a();` destroys the ghost and `z2.b = y2;` carries only the blocking unit; no logging call appears in the arm (decompiled y.java:4909-4934).
[^3]: `wiki/sources/m31-refusal/waypoint-removed-silently.txt:30-32` — `if (!z2.c) { this.ay(); }` after `z2 = this.a(au2, au2.b, au2.d, au2.e, au2.f)` at `:2` (decompiled y.java:1630-1632).
[^4]: `wiki/sources/m31-refusal/cant-afford-retry.txt:5-12` — the failed charge `!(...).c(this)` arm is the only one setting `z3.c = true`, guarded by `this.V < 1000.0f`, alongside the site marker (decompiled y.java:4940-4949).
[^5]: `wiki/sources/m31-refusal/waypoint-resume-check.txt:4` — `if (au2.a != com.corrodinggames.rts.game.units.av.c || au2.b != as2 || !(...c(au2.e - f2) < 10.0f) || !(...c(au2.f - f3) < 10.0f)) continue;` (decompiled y.java:3411).
