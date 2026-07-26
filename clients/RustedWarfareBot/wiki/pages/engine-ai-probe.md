---
title: "The AI Zone Probe, and Why It Is Not Perception"
tags: [ai, research, legitimacy, agent, invariants, tooling]
related:
  - "[[engine-ai-zones]]"
  - "[[engine-ai-triggers]]"
  - "[[multiplayer-portability-invariants]]"
  - "[[perception-visibility]]"
  - "[[wire-contract-ndjson]]"
  - "[[engine-name-oracle]]"
source_paths:
  - "agent/src/rwbot/agent/AiZones.java"
  - "agent/src/rwbot/agent/AgentOptions.java"
  - "wiki/sources/m15-ai-zones/zone-dump.txt"
  - "wiki/sources/m15-ai-zones/zone-dump-330s.txt"
  - "runs/decompiled/com/corrodinggames/rts/game/i.java:547"
game_version: "1.15 (code 176, build #28)"
fact_checked: "2026-07-26"
confidence: high
hubs: [bot-architecture, engine-internals]
---

# The AI Zone Probe, and Why It Is Not Perception

Reading the shipped AI's zones settles questions the source alone cannot ([[engine-ai-zones]], [[engine-ai-triggers]]). It is also the single most tempting thing in this codebase to cheat with, so the tool exists under constraints, and the constraints are the interesting part.

## Why the planner must never see this

**A zone is intent, not observation.** It carries where the AI plans to expand, how large a group must be before it commits, and how long it has been staging. A human player cannot see any of that. They infer it from units they can observe, and the inference is most of the skill. There is no observable counterpart to launder a zone read through, so no amount of care makes a planner that consumes one honest.

**Zones have no visibility model at all.** Every entity carries the engine's own per-player fog test, which is what keeps perception legitimate ([[perception-visibility]]). The zone base class has containment tests, distance helpers and a placement sampler, and nothing resembling a visibility check. Reading zones does not stretch the fog rule; it steps around it.

**And it would not work anyway.** Zones exist only on AI players. The local human player is constructed as a different player subclass entirely — slot 0 is built as one class named `"Player"`, slots 1 through 7 as the AI class.[^1] A human opponent therefore has no zone list to read. A policy resting on this would beat the shipped AI and silently do nothing against a person, which is precisely the failure the portability invariant exists to prevent ([[multiplayer-portability-invariants]]).

That last point is the one worth keeping. The objection is not only that zone-reading is unsporting; it is that it produces a bot that cannot be evaluated against anything but itself.

## Two guarantees, both structural

Discipline is not a mechanism, so the separation is enforced by shape rather than by intention.

**It never reaches the wire.** The output goes to the agent log. The planner reads the NDJSON stream and nothing else ([[wire-contract-ndjson]]), so the Python side *cannot* consume this — not "does not", cannot. Wiring it in would mean adding a record kind, a decoder and a validator, which is a deliberate act with a diff, not a slip.

**It is off unless asked for.** A boolean agent option, defaulting false, with a self-check asserting the default.[^2] A run that did not request it cannot produce it, so an archived capture cannot quietly contain it and a later reader cannot mistake one for perception.

## What it does

Enumerates the master entity list, collects the distinct owners that are AI instances, reads each one's zone collection and renders every declared field of every zone. It runs on the discovery schedule rather than inventing its own, so several offsets in one run give a time series — which is the whole point when the open questions are cooldowns.[^3]

Three details matter more than they look:

**Players are reached through their units** rather than through the engine's player table. That keeps the probe to three new pinned names — the AI class, the zone base class and the zone field — instead of also pinning the player array and its accessor. An AI owning nothing goes unreported, which costs nothing: a player with no units has no strategy to measure.

**Zones are matched against the pinned base class**, not trusted from the collection's element type, which is erased. Reading an unrelated object's fields as a zone would produce a plausible table of numbers, and this project has already paid twice for plausible wrong answers.

**Fields are rendered generically, not by pinned name.** For a probe whose purpose is to confirm what the obfuscated letters mean, a dump that applied the current reading could only ever agree with it. Reporting every declared field lets the reading be checked against the numbers — and on first use it caught a wrong one.

## What the first run settled

Four AI players sampled at 40, 90 and 150 seconds.[^4] Radii confirmed at 420 and 360; expansion zones confirmed arriving over time on the cooldown path rather than at bootstrap; the capacity ratio confirmed bounded in [0, 1]; the unit budget found to start at −1.0 and range negative, which the source reading had missed.

It also appeared to contradict the source on attack-group size, which took a second run to resolve — see below, because how that resolved is the better argument for how the probe is built.

## The generic dump earning its keep twice

Both times the probe changed a conclusion, it was because it reported a field nobody had asked for.

**The sea group.** A first run showed every attack-flagged group targeting five, never the three the source predicted, and the triggers page briefly recorded that reading as unsupported. The dump also carried `B`, a flag no page mentioned: every one of those groups had it set, and it marks a **sea** group, whose target is five by a different branch. A longer run then caught the genuine first land attack group at three.[^5] A probe rendering only the fields the pages cared about would have shown two indistinguishable fives and left a correct claim retracted.

**The negative budget.** The unit-production accumulator was read from the source as filling toward a cap. The dump showed it initialising at −1.0 and ranging negative, which makes it a debt counter rather than a stock — a distinction with real behavioural weight that the source reading had passed over.

## Enums render by name

The first version printed enum fields as their class, which lost exactly the values worth reading. The renderer now calls `Enum.name()`, which is safe under the rule it lives beside: `name()` is final on `java.lang.Enum` and returns a stored string, so no engine code runs on the probe thread.[^6]

That turned out to matter beyond this probe. The engine's enums are obfuscated in their *field* names only — the constant name strings survive in the bytecode — so the zone state and kind read as `Pre`/`Prepare`/`Active` and `Main`/`ResourceOutpost`/`ForwardOutpost` rather than as `a`, `b`, `c`. Following that thread out of the probe produced a general naming oracle for the whole jar ([[engine-name-oracle]]).

[^1]: `runs/decompiled/com/corrodinggames/rts/game/i.java:547` — `this.bs = new e(0); this.bs.v = "Player";` followed by `for (int i2 = 1; i2 < 8; ++i2) { new com.corrodinggames.rts.game.a.a(i2); }`. Only the second class carries a zone list.
[^2]: `agent/src/rwbot/agent/AgentOptions.java` — the `aiZones` accessor and its rationale; `OptionChecks` asserts that it defaults off, parses, and rejects a non-boolean.
[^3]: `agent/src/rwbot/agent/AiZones.java` — the whole probe, including the ownership walk and the base-class match.
[^4]: `wiki/sources/m15-ai-zones/zone-dump.txt` — the distilled dump, and the header recording the exact agent options that produced it.
[^5]: `wiki/sources/m15-ai-zones/zone-dump-330s.txt` — `Q=37 A=3 h=true seaGroup=false` at 330 seconds, against the `A=5 seaGroup=true` groups present from the first sample.
[^6]: `agent/src/rwbot/agent/ObjectView.java` — the enum branch of `summarise`, and the comment recording why it does not breach the no-engine-`toString` rule. `DiscoveryChecks` asserts a constant renders by name.
