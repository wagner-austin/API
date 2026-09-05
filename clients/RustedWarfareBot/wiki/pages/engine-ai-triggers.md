---
title: "The Shipped AI's Build and Attack Triggers"
tags: [ai, strategy, engine, economy, combat, planner]
related:
  - "[[engine-ai-zones]]"
  - "[[policy-loop]]"
  - "[[mechanics-build-actions]]"
  - "[[mechanics-unit-catalogue]]"
  - "[[policy-threat]]"
  - "[[engine-ai-probe]]"
source_paths:
  - "runs/decompiled/com/corrodinggames/rts/game/a/i.java:1029"
  - "runs/decompiled/com/corrodinggames/rts/game/a/i.java:1045"
  - "runs/decompiled/com/corrodinggames/rts/game/a/i.java:1098"
  - "runs/decompiled/com/corrodinggames/rts/game/a/g.java:190"
  - "runs/decompiled/com/corrodinggames/rts/game/a/g.java:448"
  - "runs/decompiled/com/corrodinggames/rts/game/a/g.java:474"
  - "runs/decompiled/com/corrodinggames/rts/game/a/a.java:1611"
  - "runs/decompiled/com/corrodinggames/rts/game/a/a.java:1620"
  - "runs/decompiled/com/corrodinggames/rts/game/n.java:1088"
  - "wiki/sources/m15-ai-zones/zone-dump.txt"
  - "wiki/sources/m32-imp-ladder/ladder-timeline.txt"
source_git_blobs:
  "wiki/sources/m15-ai-zones/zone-dump.txt": "dbfff06d71e2e199976a8bd8727e163aa7d451f9"
  "wiki/sources/m32-imp-ladder/ladder-timeline.txt": "2d5b68a6b1ff3f6e18554bb15ebc33e7870d4ee5"
game_version: "1.15 (code 176, build #28)"
fact_checked: 2026-09-05
confidence: medium
hubs: [engine-internals, game-mechanics]
---

# The Shipped AI's Build and Attack Triggers

Zones say *where* ([[engine-ai-zones]]). This is *when*: what makes the AI put down a building, make a unit, and commit a force. All three are timers with gates on them, and none of the numbers are ones the bot could have guessed.

Two conventions throughout. Timers count **down** to zero and are then reset — a large reset value means a rare action. And credit tests go through the player's own `credits >= n` predicate, reading the same field the bot reads for its own balance.[^1]

## Difficulty is one boolean in most places

Almost every constant below has a harder variant, selected by a single predicate: difficulty is at maximum, **or** the player's handicap exceeds 300.[^2] A second predicate for "difficulty at least medium" exists and is used more sparingly. Where a figure below has two values, the second is the hard one.

## Putting down a building

The base zone carries a `buildBuildingDelay`. When it reaches zero the zone considers one building, then resets the delay to **270 + (zone id mod 15)**, or **190 + (zone id mod 15)** on hard.[^3]

That modulus is worth stealing outright. The zone's own id is used as a phase offset, so an AI with several bases does not have all of them attempt to build on the same tick. It is a one-token fix for a thundering herd, and the bot will need the equivalent the moment it runs more than one builder.

Two penalties stretch the delay before anything else is checked: a ratio the overlay does not name — call it *capacity* — adds **+180** below 0.2 and another **+180** below 0.08. A low-capacity base therefore considers building roughly three times less often.

Then the gate, and it is a **credit ladder** rather than a threshold:[^4]

| capacity above | …or credits at least |
|---|---|
| 0.8 | 1,300 |
| 0.4 | 1,700 |
| 0.2 | 2,100 |
| 0.1 | 2,800 |
| 0.05 | 3,100 |
| 0.01 | 4,800 |

Every row must pass. The effect is a sliding bar: a base near capacity builds on 1,300 credits, one nearly out of room needs 4,800. The AI does not stop expanding a saturated base — it prices it.

On a failed attempt the delay is **cut by 120** rather than left alone, so a refusal retries sooner than a success does, and a failure counter is incremented alongside the attempt counter. Those are the `lastAttemptedBuilding*` fields the overlay prints.

## Making units

Unit production is not a timer but a **budget**. A float the overlay calls `allowedUnits` accumulates every tick and is spent by the production routine.[^5]

It fills from three sources: a flat **+0.015 per tick while the base is unsafe** — larger than every credit-driven increment except the one at 30,000, so a threatened base outproduces a merely rich one — plus a stack of credit-gated increments that switch on at 1,600, 2,200, 2,600, 5,000, 8,000, 9,000, 10,100 and 30,000 credits, plus a baseline that is **cut by a factor of seven once two or more defensive groups already exist**.

Two clamps: hard ceiling **3.5**, and a much lower ceiling of **1.2** whenever the AI holds under 800 credits and is not under threat. So a poor, safe AI is held to roughly one pending unit; a rich or frightened one banks up to three and a half.

The budget also **starts and goes negative** — the field initialises to −1.0 and the live dump shows values from −1.0 to 1.79 across zones.[^11] So it is a debt-then-credit counter rather than a stock: a new zone owes a unit's worth of accumulation before it may make anything.

Spending is a burst: the production routine is called up to **12 times in one tick**, breaking out as soon as the budget drops below 3.0.[^6] Normally that is one unit per tick. An AI sitting on a full budget empties it in a single frame.

This is the mechanism our bot most obviously lacks. It produces when the plan's next entry is affordable, with no notion of a reserve, no reaction to being attacked, and no burst.

## Committing an attack

A combat group is created **empty with a target size** and recruits until it is full — it only accepts members while the size exceeds the current count, and reports itself full on the same comparison.[^7] The sizes are fixed constants, and they escalate:[^8]

- **Defensive groups: 8**, or 10 on hard. One to three of them, the cap rising with a counter at thresholds of 6 and 11.
- **Attack groups: 3** for the first wave, **5** for the next few, **7** thereafter — and on hard, **14**, rising to **18** after 25 waves. The source was read as allowing exactly one attack group at a time, and that reading **did not survive live observation**: at Impossible, single snapshots carry up to nine attack-flagged groups at once — land and sea, in every fill state, including two 14-target land groups both labelled `attacking main target` in one snapshot (7 members each, consistent with post-commit attrition).[^13] Whether the one-at-a-time reading was ever right for a narrower scope (one *forming* group per creation branch) is unresolved; as a statement about the live game it is wrong.
- **Sea groups: 5**, or 10 on hard, one at a time, and only when the map's water area exceeds a threshold.
- **Transport groups**: up to three, each wanting one unit.

So the AI will not attack until a group is full, and the first land attack group targets **three**.

That last claim went wrong once and is worth keeping the trace of. A first dump showed every `h=true` group targeting 5, never 3, and this page briefly recorded the escalation ladder as unsupported. It was the reading of the dump that was wrong, not the reading of the source: those groups all carried `B=true`, the **sea-group** flag, and a sea group's target is 5 by a different branch entirely.[^11] A longer run caught the real thing — a fourth group appearing at 330 seconds with `A=3`, `h=true`, `B=false`.[^12] The generic field dump is what made that visible; a probe rendering only the fields this page cared about would have shown two indistinguishable fives.

The same runs confirm fill-then-commit as a mechanism. Membership climbs while the target stays fixed, staging stays false on every group that has not reached its size, and the attack delay sits at a flat 1,000 until a group fills — the one defensive group observed reaching 8 of 8 is also the only one whose delay had reached 0.[^12]

Once a group is full, the cycle is:[^9]

1. **Wait** for `AttackDelay` to reach zero, then flip to *staging*.
2. **Acquire a target** from the AI's chooser, revalidated whenever it dies or completes.
3. **Stage** — gather at the zone. Staging ends when **no member is more than 170 world units from the centre**, which is a physical rendezvous test rather than a timer.
4. **Attack.**

Two early exits from staging, and both are more interesting than the happy path. If **any member has taken damage within the last 1,000 game frames**, staging is abandoned immediately and the group attacks — the engine's own log line is `"Not staging due to damage"`. Being shot while forming up commits you. And there is a hard timeout: after **17,000** accumulated staging time it attacks regardless, logged as `"attacking target"`.

While attacking, orders are re-issued every **800**. Each re-issue filters the group to the members that can actually reach the target, and aborts the attack outright when none can — `"cannot reach main target"`. The surviving members get one command, and **80% of the time it is an attack-move to the target's current position rather than an order against the unit itself**.[^10] The AI mostly attacks *ground*, not units.

## What transfers

The bot has attack orders and nothing that decides when to use them. Three pieces here are directly portable and cost no strategy commitment:

1. **Fill-then-commit.** A group with a target size that recruits until full is a complete answer to "when do I attack", and the first size is 3.
2. **Damage cancels staging.** A one-line rule that turns a passive gather into a reactive one.
3. **Attack-move to a position, mostly.** Cheaper than chasing, and it is what the engine's own AI does four times out of five.

The build side transfers less directly, because the capacity ratio driving the credit ladder is a quantity the bot does not compute. What does transfer is the shape: **price the decision instead of thresholding it**, and use the zone id as a phase offset so parallel builders do not collide.

## What is not established

The capacity ratio is read but not named — the overlay does not print it, so its definition is inferred from use rather than from the engine saying so. Everything that depends on it is therefore shape-correct and scale-uncertain.

The group sizes and the budget range have now been watched ([[engine-ai-probe]]), and one of them contradicted the reading, which is recorded above rather than quietly fixed. The **timer constants still have not been**: they are in the engine's own tick accumulator rather than seconds, and converting them to wall-clock needs a measurement against the frame rate established for this build ([[engine-tick-and-clock]]).

**The lifecycle has now been observed firing, at Impossible, over whole matches** (2026-09-05, five instrumented champion matches on the impopen96 control seeds, 53 zone snapshots).[^13] What the sandbox runs never reached: full groups labelled `attacking main target`; the escalation ladder live at difficulty 3 — the first land group at target **3** (full by ~160 game-seconds), then **5**, then **14**, so Impossible uses the hard sizes; the source-predicted abort path live (`cannot reach main target` on a sea group whose label persisted across three snapshots); and behaviour labels the overlay never showed in sandbox — `fighting attacker`, `flight from attacker`, `random move: bad target`. The 18-after-25-waves rung was not reached in any match (the champion dies at ~10-14 waves). What is still unobserved: the staging flag transitioning (column `c` read false in all 308 extracted rows — either the flag is briefer than the 15-second sampling or the letter is misassigned), the damage abort's log line, and the 17,000 timeout.

The capacity ratio is confirmed to be bounded in [0, 1] by the same dump, and the build-delay values are consistent with the two penalties, but the ladder thresholds themselves remain unobserved.

The target chooser itself is deliberately out of scope here — this page covers when a force commits, not what it picks.

[^1]: `runs/decompiled/com/corrodinggames/rts/game/n.java:1088` — `public boolean a(double d2) { return this.o >= d2 || d2 == 0.0; }`. `o` is the credit field the agent already reads as `EngineNames.CREDITS`, and the AI class extends the player class, so `this.R.a(1300.0)` is "the AI holds at least 1,300".
[^2]: `runs/decompiled/com/corrodinggames/rts/game/a/a.java:129` — `ac()` returns `ag() == 3 || ag() > 300` and `ad()` returns `ag() >= 2`, where `ag()` returns a single stored int.
[^3]: `runs/decompiled/com/corrodinggames/rts/game/a/i.java:1029`–`1039` — the `e <= 0` branch, its two resets, and the two capacity penalties.
[^4]: `runs/decompiled/com/corrodinggames/rts/game/a/i.java:1045` — one conjunction of six `(capacity > x || credits >= y)` clauses; `:1048` is the `-120` penalty on a failed attempt.
[^5]: `runs/decompiled/com/corrodinggames/rts/game/a/i.java:1057`–`1088` — the accumulator, including the unsafe bonus at `:1059`, the sevenfold reduction when the defending-group count reaches 2, and the credit-gated increments.
[^6]: `runs/decompiled/com/corrodinggames/rts/game/a/i.java:1092`–`1101` — the 1.2 and 3.5 clamps, then `for (int i2 = 0; i2 < 12; ++i2) { this.v(); if (!(this.d >= 3.0f)) break; }`.
[^7]: `runs/decompiled/com/corrodinggames/rts/game/a/g.java:190` — recruitment skips a candidate when `this.A <= this.F.size()`; `:199` returns the same comparison as the group's full test.
[^8]: `runs/decompiled/com/corrodinggames/rts/game/a/a.java:1611`–`1646` — four creation branches setting `A` to 8/10, 3/5/7/14/18, 5/10 and a transport group wanting 1, each with its own concurrency cap.
[^9]: `runs/decompiled/com/corrodinggames/rts/game/a/g.java:448` — `e(float)`, the group cycle. The rendezvous test at `:474` is `c(y2) > 28900.0f`, a squared distance, so 170 world units. The damage abort is at `:481`, comparing the member's last-hit frame against the current frame minus 1,000. The timeout is at `:486`.
[^10]: `runs/decompiled/com/corrodinggames/rts/game/a/g.java:490`–`520` — the 800 re-issue interval, the reachability filter, the `"cannot reach main target"` abort, and `f.a(0, 100) < 80` selecting the position-targeted command over the unit-targeted one.
[^11]: `wiki/sources/m15-ai-zones/zone-dump.txt` — the first run, four AI players at 40, 90 and 150 seconds, where every `A=5` group is also `B=true`. The sea-group branch is at `runs/decompiled/com/corrodinggames/rts/game/a/a.java:1636`, gated on the map's water area.
[^12]: `wiki/sources/m15-ai-zones/zone-dump-330s.txt` — a longer run at 60, 180 and 330 seconds. At 330s one player holds `Q=37 A=3 h=true B=false`, the first land attack group, and a defensive group at 8 of 8 with `l=0.0`.
[^13]: `wiki/sources/m32-imp-ladder/ladder-timeline.txt` — 308 combat-group rows from five instrumented Impossible matches (header records the exact launch and extraction). Nine concurrent attack-flagged groups and the twin `attacking main target` 14-groups at 7 members each: seed 8711417 rows at t=170s. First land group full at target 3: seed 8711001, t=35s (`null target=3 members=3 attack=true sea=false`). The persistent `cannot reach main target` sea group: seed 8711417, t=140/155/170s, same accumulator column climbing 36210 -> 42451.
