---
title: "The Verdict — Asking the Engine Who Won"
tags: [policy, scoring, perception]
related:
  - "[[policy-loop]]"
  - "[[perception-visibility]]"
  - "[[policy-combat]]"
source_paths:
  - "src/rw_bot/policy/verdict.py"
  - "src/rw_bot/policy/campaign.py"
  - "wiki/sources/m6-wire/world-sample.ndjson"
source_git_blobs:
  "src/rw_bot/policy/verdict.py": "095d8c64f2e8723feeffa924ea39c48848b65dc6"
  "src/rw_bot/policy/campaign.py": "85a4a357c27a4fbc90a1d35fb39a18bfa380c96f"
  "wiki/sources/m6-wire/world-sample.ndjson": "201f82ea1c9071c70d20ee8b29952b0d2fc79455"
game_version: "1.15 (code 176, build #28)"
fact_checked: 2026-08-17
confidence: high
hubs: [bot-architecture]
---

# The Verdict — Asking the Engine Who Won

Every figure the run report carried for its first month measured the bot's
*activity*. Three of them cannot answer "did this help":

- **`engaged gone`** counts targets ordered against that are no longer visible.
  A hostile that retreated into fog reads identically to a dead one.
- **`enemies seen`** counts *visible* hostiles, so it rises when our own army
  walks somewhere new. It measures our vision as much as their army.
- **`army` at the end** is real, but three runs of identical code gave 3, 6 and
  14. The variance swamps most effects worth measuring.

The engine keeps the answer itself and names it. A player carries a
"was defeated" flag and a "has been wiped out" flag, and the world knows how
many players remain — the engine ends the match when that count reaches one.
None of the three can be inflated by re-targeting, by scouting, or by a lucky
sample.

## Order matters

Losing is checked before winning, because both can be true of the same
observation: when we are eliminated the survivor count falls towards one as
well, and reading that as a victory would grade a loss as a win. Wiped is
checked before defeated for the same reason — it is the stronger statement about
the same event.

```
wiped -> defeated -> won -> survived
```

`survived` is the honest reading of a match that was *stopped* rather than
decided, which is what a sample-budgeted run almost always is.

## It is also the loop's only early exit

The two-phase loop stopped on "no army left" and on "nothing hostile in sight".
Neither survived the move to one tick ([[policy-loop]]):

- **Nothing hostile in sight is the opening position of every match.** The map
  is fogged and the opponents are across it, so it would have ended the run on
  its first observation.
- **An army of zero is no longer terminal** now that production runs every tick.
  Losing a wave is a setback to rebuild from, not a reason to stop playing.

Both were proxies for a verdict the engine states outright, so the loop now
takes the verdict and nothing else.

## What it does not tell you

The verdict is binary and late. It says a match was lost; it does not say the
army value was 500 against a leader on 4,150 forty seconds earlier, which is the
figure that would have predicted it. That is the engine's per-player scoreboard,
which rides on the same stream ([[perception-visibility]]) and is what
`army value` and `income` in the match report are read from.
