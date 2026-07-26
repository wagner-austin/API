---
title: "How the Built-in AI Plays"
tags: [ai, strategy, opponent, combat, production, reverse-engineering]
related:
  - "[[policy-loop]]"
  - "[[mechanics-build-tree]]"
  - "[[issuing-orders]]"
  - "[[engine-name-oracle]]"
  - "[[multiplayer-portability-invariants]]"
source_paths:
  - "wiki/sources/m14-ai/ai-state-dump.txt:11"
  - "wiki/sources/m14-ai/ai-state-dump.txt:25"
  - "wiki/sources/m14-ai/unit-mix.txt:7"
  - "wiki/sources/m14-ai/unit-mix.txt:18"
  - "wiki/sources/m14-ai/unit-mix.txt:67"
  - "wiki/sources/m14-ai/attack-group-staging.txt:30"
  - "wiki/sources/m14-ai/attack-group-staging.txt:39"
  - "wiki/sources/m14-ai/attack-group-staging.txt:42"
  - "wiki/sources/m14-ai/attack-group-staging.txt:49"
  - "wiki/sources/m14-ai/attack-group-staging.txt:73"
  - "wiki/sources/m13-expand/idle-after-plan.txt"
game_version: "1.15 (code 176, build #28)"
fact_checked: "2026-07-26"
confidence: medium
hubs: [engine-internals, bot-architecture]
---

# How the Built-in AI Plays

The bot loses to these opponents, so how they play is the most useful strategy document available — and it ships inside the jar. This reads it rather than reasoning about RTS play in the abstract.

`com.corrodinggames.rts.game.a` is the AI package. It identifies itself: two of its error strings begin `"AI: "`, and it draws its own state over the map with labels naming the fields.[^1]

**Read this as a map, not as a measurement.** Every claim below is decompiled source — strong for what the code *can* do, weaker than a run for what it *does*. Decompiled Java is a reconstruction and can be subtly wrong on obfuscated input, which is why this page is `medium` and not `high`. The behavioural counts that *are* observed are marked as such.

## Production is a weighted mix, not a build order

There is no scripted opening. The AI keeps a **unit mix**: it scans every registered type, keeps the ones a predicate admits, and gives each a weight — 10.0 unless a subclass overrides it.[^2][^3] Picking is then weighted-random over that set, filtered by movement class and capped at a tech level.[^4]

That is a different shape of decision from ours entirely. Our planner executes an ordered list and stops ([[policy-loop]]); this samples continuously, so it never finishes and never needs to know what to build next.

## Attacking is delayed, then staged, then committed

Units are pooled into attack groups, and a group moves through three states.

**Delay.** A group carries an attack-delay counter that ticks down before it will do anything at all. It starts at 1000.[^5]

**Staging.** When the delay expires the group starts massing, and it will not attack while any member is still more than **170 world units** from the rally point — the code compares squared distance against 28900.[^6]

**Commit.** Staging ends on any of three conditions, and the two that are not "everyone arrived" are the interesting ones:

- Everyone has gathered.
- **Any member takes damage within the last second** — the group stops massing and attacks immediately, logging *"Not staging due to damage"*.[^7]
- **17 seconds elapse**, logging *"attacking target"*.[^8] Massing is bounded so a group cannot wait forever for a unit that will never arrive.

## Two dispatch details worth copying

**Orders refresh on a timer, not on a change.** Once committed, the group re-issues every 800 ms.[^9] Our loop re-orders only when a unit's target changes, which is cheaper but leaves a unit with a stale waypoint when the world moves under it.

**It attacks the ground, not the unit — 80% of the time.** On each refresh the AI rolls 0–99 and, below 80, targets the *position* the enemy occupies rather than the enemy itself.[^10] Only the remaining fifth is a unit-targeted attack of the kind [[issuing-orders]] describes.

That ratio is not obviously a compromise. A position attack does not chase, so the group stays together and engages whatever it meets on arrival; a unit attack follows a target that may be running, which pulls a formation apart. The engine's own AI prefers the first four times in five.

## Measured against what the bot does

Our bot builds a fixed list of four tanks and attacks with them the moment the plan completes. Against that, the observed opponents went from 54 to 126 visible units in five minutes while we banked credits and built nothing further.[^11]

Three differences, in the order they probably matter:

1. **They never stop producing.** We stop at the end of a list.
2. **They mass before committing, up to a 17-second bound.** We attack with whatever exists.
3. **They break off massing when hit.** We have no reaction to damage at all.

None of those is a *strategy* in the sense of a build order to copy. They are mechanisms we lack, which is why reading this was worth more than reading advice.

## What is not established here

Which types each mix admits, and their weights, live in the mix subclasses and are not read yet — so "weighted-random over admitted types" is the shape without the contents. The difficulty settings are not traced either, so whether delay, staging radius and the 17-second bound vary by difficulty is unknown. Both are code-reading tasks, not run tasks.

Nothing on this page has been confirmed by watching the AI do it. The strongest available check is cheap and not yet done: a run that logs an opponent's unit count and engagement timing against these constants.

[^1]: `wiki/sources/m14-ai/ai-state-dump.txt:11` — the base overlay printing `"attackingCount: "`, with `"Turtling: true"` at `:16`; the per-group dump prints `StagingForAttack` at `:25`, `AttackDelay` at `:26` and `StagingTimer` at `:28`, naming the fields this page cites. The `"AI: "` error strings are at `com/corrodinggames/rts/game/a/d.java:58` and `:86` in the decompiled tree.
[^2]: `wiki/sources/m14-ai/unit-mix.txt:18` — `for (as as2 : ar.ae)` inside the mix rebuild, filtered by the abstract predicate `a(as)`.
[^3]: `wiki/sources/m14-ai/unit-mix.txt:7` — `return 10.0f;` as the default weight, overridable per mix.
[^4]: `wiki/sources/m14-ai/unit-mix.txt:67` — `float f3 = f.c(0.0f, f2);` picking a point in the summed weight, with the movement-class and tech-level filters applied on both passes.
[^5]: `com/corrodinggames/rts/game/a/g.java:37` [synthesis] — `float l = 1000.0f;`, the field the state dump labels `AttackDelay`. Field declarations sit outside the archived excerpt's range; the excerpt begins at the method that consumes it.
[^6]: `wiki/sources/m14-ai/attack-group-staging.txt:30` — `if (!(this.c(y2) > 28900.0f)) continue;`, and 28900 is 170 squared.
[^7]: `wiki/sources/m14-ai/attack-group-staging.txt:39` — `this.a("Not staging due to damage");`, reached when a member's last-damage clock is within 1000 ms of the engine clock at `:37`.
[^8]: `wiki/sources/m14-ai/attack-group-staging.txt:42` — `if (this.u > 17000.0f)`, followed by `this.a("attacking target");` at `:44`.
[^9]: `wiki/sources/m14-ai/attack-group-staging.txt:49` — `this.z = 800.0f;` reset each time the dispatch branch runs.
[^10]: `wiki/sources/m14-ai/attack-group-staging.txt:73` — `if (this.w != null && com.corrodinggames.rts.gameFramework.f.a(0, 100) < 80)` choosing the position setter over the unit setter.
[^11]: `wiki/sources/m13-expand/idle-after-plan.txt` — 800 samples past a completed plan: owned units unchanged at 9, credits 8,539 → 21,164, visible enemy units 54 → 126.
