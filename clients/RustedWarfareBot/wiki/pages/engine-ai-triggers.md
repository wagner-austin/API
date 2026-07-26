---
title: "The Shipped AI's Build and Attack Triggers"
tags: [ai, strategy, engine, economy, combat, planner]
related:
  - "[[engine-ai-zones]]"
  - "[[policy-loop]]"
  - "[[mechanics-build-actions]]"
  - "[[mechanics-unit-catalogue]]"
  - "[[policy-threat]]"
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
game_version: "1.15 (code 176, build #28)"
fact_checked: "2026-07-26"
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

Spending is a burst: the production routine is called up to **12 times in one tick**, breaking out as soon as the budget drops below 3.0.[^6] Normally that is one unit per tick. An AI sitting on a full budget empties it in a single frame.

This is the mechanism our bot most obviously lacks. It produces when the plan's next entry is affordable, with no notion of a reserve, no reaction to being attacked, and no burst.

## Committing an attack

A combat group is created **empty with a target size** and recruits until it is full — it only accepts members while the size exceeds the current count, and reports itself full on the same comparison.[^7] The sizes are fixed constants, and they escalate:[^8]

- **Defensive groups: 8**, or 10 on hard. One to three of them, the cap rising with a counter at thresholds of 6 and 11.
- **Attack groups: 3** for the first wave, **5** for the next few, **7** thereafter — and on hard, **14**, rising to **18** after 25 waves. Exactly one attack group exists at a time.
- **Sea groups: 5**, or 10 on hard, one at a time, and only when the map's water area exceeds a threshold.
- **Transport groups**: up to three, each wanting one unit.

So the first thing a shipped AI ever attacks with is three units, and it will not send them until it has three.

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

None of the timer constants have been observed. They are read from the source and the units are the engine's own tick accumulator, not seconds; converting them to wall-clock needs a live measurement against the frame rate already established for this build ([[engine-tick-and-clock]]). That is the main reason this page is `medium`, and it is the same gap as [[engine-ai-zones]]: confirming any of it means watching an opponent's AI object over a live game, which the agent cannot do today.

The target chooser itself is deliberately out of scope here — this page covers when a force commits, not what it picks.

[^1]: `runs/decompiled/com/corrodinggames/rts/game/n.java:1088` — `public boolean a(double d2) { return this.o >= d2 || d2 == 0.0; }`. `o` is the credit field the agent already reads as `EngineNames.CREDITS`, and the AI class extends the player class, so `this.R.a(1300.0)` is "the AI holds at least 1,300".
[^2]: `runs/decompiled/com/corrodinggames/rts/game/a/a.java` — `ac()` returns `ag() == 3 || ag() > 300` and `ad()` returns `ag() >= 2`, where `ag()` returns a single stored int.
[^3]: `runs/decompiled/com/corrodinggames/rts/game/a/i.java:1029`–`1039` — the `e <= 0` branch, its two resets, and the two capacity penalties.
[^4]: `runs/decompiled/com/corrodinggames/rts/game/a/i.java:1045` — one conjunction of six `(capacity > x || credits >= y)` clauses; `:1048` is the `-120` penalty on a failed attempt.
[^5]: `runs/decompiled/com/corrodinggames/rts/game/a/i.java:1057`–`1088` — the accumulator, including the unsafe bonus at `:1059`, the sevenfold reduction when the defending-group count reaches 2, and the credit-gated increments.
[^6]: `runs/decompiled/com/corrodinggames/rts/game/a/i.java:1092`–`1101` — the 1.2 and 3.5 clamps, then `for (int i2 = 0; i2 < 12; ++i2) { this.v(); if (!(this.d >= 3.0f)) break; }`.
[^7]: `runs/decompiled/com/corrodinggames/rts/game/a/g.java:190` — recruitment skips a candidate when `this.A <= this.F.size()`; `:199` returns the same comparison as the group's full test.
[^8]: `runs/decompiled/com/corrodinggames/rts/game/a/a.java:1611`–`1646` — four creation branches setting `A` to 8/10, 3/5/7/14/18, 5/10 and a transport group wanting 1, each with its own concurrency cap.
[^9]: `runs/decompiled/com/corrodinggames/rts/game/a/g.java:448` — `e(float)`, the group cycle. The rendezvous test at `:474` is `c(y2) > 28900.0f`, a squared distance, so 170 world units. The damage abort is at `:481`, comparing the member's last-hit frame against the current frame minus 1,000. The timeout is at `:486`.
[^10]: `runs/decompiled/com/corrodinggames/rts/game/a/g.java:490`–`520` — the 800 re-issue interval, the reachability filter, the `"cannot reach main target"` abort, and `f.a(0, 100) < 80` selecting the position-targeted command over the unit-targeted one.
