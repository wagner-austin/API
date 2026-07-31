---
title: "How Humans Win: the Community Strategy Corpus"
tags: [strategy, community, pvp, scouting, raiding, micro, economy]
related:
  - "[[policy-holding-ground]]"
  - "[[policy-combat]]"
  - "[[policy-economy]]"
  - "[[engine-ai-zones]]"
  - "[[mechanics-unit-value]]"
sources:
  - "https://steamcommunity.com/sharedfiles/filedetails/?id=1858207046 (Multiplayer Basics)"
  - "https://steamcommunity.com/sharedfiles/filedetails/?id=1449760671 (RTS Defence)"
  - "https://www.chaptercheats.com/cheat/pc/388370/rusted-warfare/hint/136775 (Text Walk Through)"
  - "https://steamcommunity.com/sharedfiles/filedetails/?id=3124371498 (How to win)"
game_version: "community guides, various patches; claims unverified against 1.15 (code 176, build #28)"
fact_checked: "2026-07-29"
confidence: low
hubs: [game-mechanics, bot-architecture]
---

# How Humans Win: the Community Strategy Corpus

Secondary sources, recorded verbatim and kept apart from the measured pages:
these are claims by players, not findings from our harness, and each one is a
sweep waiting to be run. Confidence is low by policy until a claim survives a
measurement, at which point it graduates to the owning page.

## The loop the guides agree on

The Multiplayer Basics guide states the whole game as one cycle: **"Expand map
control (move your military forward; build extractors and turrets together
away from your base), then Build military, then Grow economy (upgrade
Fabricators, then build more of them)"** — with the military sized "only
ever-so-slightly more than your opponent" and everything else banked into
economy.[^1]

Three things in that sentence the bot does not do: military moves *forward*
(ours rallies at base), extractors are built *with turrets together, away from
base* (ours are naked), and the economy grows through *Fabricators* (ours has
never built one).

## Scouting is stated as the difference between winning and losing

"KEEP. DOING. THIS." — continuous scouting with Interceptors (large maps) or
Scouts (small maps), so the opponent's production type is known before it
arrives.[^1] The counter matrix below is unusable without it, which is the
guide's own point: every counter is cheap *if chosen in advance*. The bot
scouts never; it learns the opponent's composition from whatever walks into
its own vision.

## The counter matrix (Multiplayer Basics, verbatim)

| Opponent fields | Answer |
|---|---|
| mass T1 land (tanks, hovertanks, small mechs) | one or two Mobile Turrets buffered by smaller units (Mech Factory T1) |
| T1 air (light gunship, helicopter) | enough Interceptors to hold, AA Mechs ASAP |
| T2 land (plasma tank, heavy tank) | as many Minigun Mechs as possible |
| T2 mechs (minigun, plasma) | Amphibious Jets |
| T1 navy on water maps | Amphibious Jets |
| T2 air (gunships, heavy interceptors) | Amphibious Jets |

The same guide flags the meta and its limit: **"Amphibious Jets are only a
HARD COUNTER to UNAWARE PLAYERS"** — they lose to AA turret walls, massed
missile ships, or massed AA mechs.[^1]

**Version caveat, registry-checked:** in our pinned build the amphibious jet
is not an airFactory unit at all — `amphibiousJet` is produced by
`combatEngineer` behind the experimental chain, its own weapon does not reach
underwater (its second form `c_amphibiousJet_underwater` does, at 100), and
every underwater-capable producer is seaFactory or experimental-tier. The
matrix's favourite unit is a mid-game investment here, not the cheap answer
the guide implies (log: 2026-07-29, the amphib arm).

## Micro exists, and it is exactly one move

"Exploit the range of your Minigun Mechs and walk backwards when engaging to
avoid damage while still cutting them down."[^1] Kiting: hold range against
shorter-reach units. The bot's combat has reach data for both sides and no
verb that uses it — units close to point-blank and trade until dead
([[policy-combat]]).

**Registry-checked, refuted by arithmetic (2026-07-30):** the claim's own
unit walks at 0.6 against its chaser's 1.1, so the range cannot be *held* —
only the first ~80 units of closure are bought — and the mech-family
composition was independently refuted 0/7 on that same speed axis before
kiting was ever tried. The composition that wins here has no envelope to
hold (`c_tank` 130 vs `c_tank` 130). The claim likely holds only against
players who fail to close, or in versions where mechs were faster.

## Defence doctrine (RTS Defence guide)

Per expansion: **"two ground turrets and two Anti-Air turrets, upgrade one
ground to Tier 2, the other to Artillery, add repair bays."**[^2] Placement
rule: **turret ranges must overlap** so they support each other; AA sits
deeper than ground.[^2] And the warning that matches our own four refuted
turret arms: turrets without an army forfeit "ALL map control beyond where
you place your turret."[^1]

The bot's defence policy knows exactly one turret type (`c_turret_t1`), has
never upgraded one, never placed AA, never built a repair bay, and never
paired turrets with expansions deliberately ([[policy-holding-ground]]).

## Harassment and the mid-game

Early harass: "a few Heavy Tanks or Amphibious Jets to harass your opponents
while otherwise doing your build as normal."[^3] Mid-game aggression at scale:
"considerable invasion force, 100-300 tanks."[^4] Attack streams flow — "send
a constant stream of tanks onto the frontline" — but *behind* a scouting
picture, at targets chosen for value, with builders travelling along to claim
ground.[^4]

## Economy notes the bot has never tested

- **Fabricators**: buildable anywhere (no pool needed), generate credits,
  upgradable, and the guides treat massing + upgrading them as the late-game
  economy engine.[^3][^4] Our economy is extractor-only by construction.
  **Registry-checked, refuted at our match length (2026-07-30):**
  `fabricatorT1.ini` prices it at 2,200 for 2 credits/s — an 1,100-second
  payback against ~890 seconds of engine clock in a full match. The advice
  describes hour-long games; here it is dead credits by the engine's own
  numbers.
- **Extractor T3 choice**: reinforcement (health/shields) versus overclock
  (faster credits, less health).[^3] The bot walks the upgrade chain blind to
  this fork.
- **Watch towers**: buildable fog vision, "see through a long range of fog,"
  from mech/combat engineers.[^3] Standing vision without spending units.

## What this corpus says the bot is missing, ranked by the guides' own emphasis

1. **Scouting** — every other decision is conditioned on it.
2. **Forward posture** — military between the enemy and the extractors;
   extractors born with cover, not naked.
3. **Counter-picking from scouted information** — the tilt exists
   ([[policy-combat]]) but reacts to what is already shooting at us.
4. **Kiting** — one verb, large claimed payoff against short-reach armies.
5. **Raiding** — implied throughout (harass early, streams at the front,
   claim while advancing); no guide treats sitting at home as viable.

[^1]: https://steamcommunity.com/sharedfiles/filedetails/?id=1858207046 — "Multiplayer Basics", Steam guide.
[^2]: https://steamcommunity.com/sharedfiles/filedetails/?id=1449760671 — "RTS Defence", Steam guide.
[^3]: https://www.chaptercheats.com/cheat/pc/388370/rusted-warfare/hint/136775 — "Text Walk Through".
[^4]: https://steamcommunity.com/sharedfiles/filedetails/?id=3124371498 — "How to win in Rusted Warfare", Steam guide (author notes untested above Medium).
