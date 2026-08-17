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
fact_checked: 2026-08-17
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

## The damage-type layer under the matrix (2026-08-01)

The counter matrix works through a mechanics layer the bot's combat profiles
do not carry: **damage types and shields**. The registry's own descriptions
name it -- the amphibious jet "shoots lightning" at Direct Damage "45
(total: 90.0)", doubled against the right target and "weak vs grounded
buildings"; the minigun mech "has shield (weak vs lightning)"; the tesla
mech is "very strong vs shields".[^7] Our profiles read hp and reach only,
so a shielded unit fights stronger than the bot's volley math believes and
a lightning unit against the right target fights double. This is the
largest named gap in the shallow-mechanics audit ([[mechanics-healing]]
records the healing half of that audit).

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

## The prescriptions, measured at Impossible (2026-08-01)

One night, one seed (12345, duel_lake, difficulty 3), one knob per arm off
the techprobe champion (baseline: dies ~2,650-2,850 samples, rival worst
dip 2,450). Every arm still loses; what moved and what did not:[^8]

- **Fabricators** (tech-fab): wiped 2,750, dip 2,350. Both built, tech
  unlocked, heavies fielded — and nothing changed. Confirms the corpus's
  own warning and the 2026-07-30 arithmetic above: at this match length
  fabricator income is dead credits.
- **Scouting** (tech-scout): wiped 2,590, dip 2,500. Thirteen scouts died
  for 9,100 credits with no measurable counter payoff. The prescription
  assumes surviving long enough to act on the intel; as a composition tax
  under this pressure it is refuted.
- **Mobile Turrets** (tech-mech v1-v3): the corpus's named anti-horde
  counter, three iterations. v1 exposed a doctrine trap — a 4,500-credit
  composition entry silently raises the derived reserve to 4,500 and
  starves every channel (defeated 2,655, dip 1,400, income 0/s). v2 pinned
  the reserve at 900: the economy breathed and produced the **best rival
  dip ever measured, 3,900 — with zero bunkers fielded** (production never
  once held 4,500; asked 1,178, got 0). v3 funded two through the saving
  hire channel: they fielded, cost 9,000, and the dip *fell* to 2,050.
  Expensive AOE is refuted; what v2 shows is that economy health drives
  the damage we do.
- **Cheap AOE instead** (tech-mecharty): six Artillery Mechs (1,400 each,
  area 100) through ordinary production — **defeated 3,190, the longest
  survival of the batch**, dip 2,800. Cheap AOE beats expensive AOE on
  both axes.
- **Reflex pair at Impossible** (kite + flee-hurt): wiped 2,983, dip
  2,250. Buys survival time, zero lethality ([[policy-exact-timing]]).
- **The shared death**: every arm ends `income 0/s, extractors 0` — the
  AI kills the extractors and the bot dies broke, identically, regardless
  of composition.
- **Guarded cover** (tech-cover): the prescription aimed at that shared
  death, and the batch's breakthrough — **survival 3,215 and rival worst
  dip 9,100, both records**, the dip more than double the previous best.
  Two turrets for 1,000 credits (defence starved after them) still
  transformed the whole match: 202 attack orders against every other
  arm's 25-131, 610 interceptions, eight heavy tanks — survival
  compounding into tech throughput. Caveats recorded in the code: cover
  is granted nearest-the-anchor first, so the frontier extractors that
  actually die are covered last, and the defence channel ran out of
  funding at two turrets ([[policy-holding-ground]]).
- **The winners do not stack** (warden): cover + pinned reserve +
  artillery mechs in goals *and* heavies — wiped 2,385, dip 2,000, the
  worst survival of the batch. Nine 1,400-credit mechs (12,600) tilted
  the production ratio away from the cheap swarm and starved the heavy
  tanks (one, against the cover run's eight), the turrets (one) and the
  attacks (18 against 202) — the same cannibalisation that sank the
  funded bunkers, one price tier down. tech-cover stands as the champion
  candidate on its own, pending multi-seed validation — the AA arm's
  12-seed doubling dissolving at 24 seeds is the protocol's own warning
  against reading one seed's 9,100 as real ([[policy-holding-ground]]).
- **Seed validation** (2026-08-01, seeds 777/424242): the survival gain
  held (3,203 and a then-record 3,698); the dip is high-variance — 2,050
  and 6,950 against seed 12345's 9,100. Cover is real; the 9,100 was
  partly luck.
- **Funded defence** (tech-cover-fund, same doctrine, the carried-deficit
  withhold in code): the starvation is *solved* — defence asked 19, got
  19, plus 9 AA flak, ending on "every structure already has cover", the
  first match in the record where the cover channel finished its job.
  Single-seed outcome sat inside the family's spread (defeated 2,753,
  dip 3,300), so the fix is judged on paired seeds, not one run
  ([[policy-economy]]).
- **Flame turrets** (tech-flame: cover + funded defence + two flame
  conversions, the community's anti-horde static): **wiped at 3,938 —
  the longest survival ever recorded, near the 4,000-sample cap** — dip
  6,200, 88 enemies destroyed, 8 turrets landed, both conversions
  completed at the engine's own 700-credit price.
- **THE SWEEP VERDICT (12 seeds, identical conditions, 2026-08-01 —
  supersedes every single-run claim below)**: tech-flame 0/12,
  techprobe 0/12. Dip distributions indistinguishable — stack median
  ~4,100 (1,850-8,250) against control ~4,200 (1,850-6,100); the
  "reliable 6-7k dips" dissolved at twelve seeds exactly as the AA
  doubling did at twenty-four. What survives: extractor retention (54
  drops vs 71), median ending worth (350 vs 0), income alive at the end
  in 3 matches vs 1. The funded-defence and flame mechanisms work as
  built; they do not change verdicts. The single-run records below are
  preserved as the variance lesson they turned out to be.[^10]
- **The stack validated — and the first non-loss at Impossible**
  (2026-08-01, paired seeds): tech-flame against the unfunded champion
  on identical seeds — 3,938 vs 3,215 (seed 12345), 2,827 vs 3,203
  (777), and at 424242 **`survived (sample_limit)`: the full 4,000-sample
  match with the bot still standing, the first non-loss in roughly
  ninety Impossible attempts**, 100 enemies destroyed. The dips tell the
  deeper story: 6,200 / 7,050 / 6,550 where the unfunded champion swung
  2,050-9,100 — the fortification line kills 6-7k of their army
  *reliably*, which is the precondition the momentum strike window was
  shelved for ("no wave dies on our line" — [[policy-exact-timing]]). The
  strike arm was later measured on the champion at Impossible and closed:
  -104 mean survival in the hands screen (imp-hands84, log 2026-08-08).[^9]
- **Strike on the wall — NOT refuted; the run measured something else**
  (flame-strike, strike 5000, seed 12345): wiped 2,253, dip 2,050,
  marches 0. Corrected reading (the first version of this entry blamed
  hold semantics and was wrong): ``strike_window`` needs a rival drop of
  5,000 and this run's worst drop was 2,050, so **the window never
  opened and the strike machinery never acted** — the decisions were
  those of plain tech-flame. The divergence from tech-flame's 3,938 /
  6,200 at the *same seed* is therefore a determinism finding, not a
  strike finding: tech-flame ran beside a second engine, flame-strike
  ran solo, and the sweep caveat (parallel runs are not bit-identical)
  evidently extends to ad-hoc concurrency — same-seed runs are
  comparable only under identical load, and single-run survival deltas
  of ~1,100 samples sit inside this noise band. Two consequences: every
  single-run comparison above carries that band, and the *consistency*
  claims (dips clustering 6-7k across varied conditions) are the robust
  ones. Strike was subsequently measured on the champion and closed at -104
  (strike5, imp-hands84, log 2026-08-08); the mechanical note stands — it
  needs ``rush 1`` to convert a release into a march, released units
  without visible targets stand still ([[policy-exact-timing]],
  [[policy-combat]]).
- **The walking wall, refuted on the champion** (flame-creep, creep 50,
  seed 12345): wiped 2,769, dip 3,500 against the champion's 3,938 /
  6,200. Eleven thousand credits of advancing turrets and repair bays
  starved the home fortress, and forward pieces die in contested ground
  — bastion's fire_bridge lesson, confirmed on the new chassis.

[^10]: `runs/sweeps/imp-flame/` — 24 scorecards; `python -m
    scripts.analyze_sweep imp-flame` for the table, 2026-08-01.

[^9]: `runs/tech-flame{,-s777,-s424242}.out` — verdicts, dips and the
    `convert:c_turret_t2_flame` ledger lines, 2026-08-01.

[^8]: `runs/tech-{fab,scout2,mech,mech2,mech3,mecharty2}.out`,
    `runs/reflex-imp3.out` — full spend ledgers and reach censuses,
    2026-08-01.

[^7]: `wiki/sources/m0-probe/printunits.log` -- the Amphibious Jet, Minigun
    Mech and Tesla Mech stat blocks, with the quoted descriptions.
[^1]: https://steamcommunity.com/sharedfiles/filedetails/?id=1858207046 — "Multiplayer Basics", Steam guide.
[^2]: https://steamcommunity.com/sharedfiles/filedetails/?id=1449760671 — "RTS Defence", Steam guide.
[^3]: https://www.chaptercheats.com/cheat/pc/388370/rusted-warfare/hint/136775 — "Text Walk Through".
[^4]: https://steamcommunity.com/sharedfiles/filedetails/?id=3124371498 — "How to win in Rusted Warfare", Steam guide (author notes untested above Medium).

## The nuke, measured: every mechanical claim true, every timing claim conditional

The community names the nuke as the Impossible finisher. Six measurements
in (2026-08-05), the split verdict:

**Mechanically confirmed, to the crater.** The launcher places by the
ordinary builder (45,000, no tech gate), `buildNuke` stockpiles an 11,000
warhead, `launchNuke` fires the wire's targeted ability, and the strike
lands where the planner points -- an owned extractor erased at (2370, 510)
on the fourth probe (`runs/nuke-probe4.out`). Two engine traps are now
law: the launch action reads *available at zero ammo* (the flag does not
carry the ammo price), and an early launch is dropped silently -- so every
launch is refired until the world answers.

**"Nuke your way out of Impossible" is refuted for any bot that cannot
first solve the Impossible economy.** Funding 45,000 by withhold from tick
one starved all three hosts before a wall stood (`runs/sweeps/imp-nuke`);
gating on 50/s income moved the same three deaths one layer later
(`imp-nuke2`). Cross-checked against every Impossible sweep on disk:
exactly one survivor exists, holding 18/s with 214 credits banked. There
is no measured Impossible state carrying the launcher's price -- the
community accounts presuppose a compounding fortress economy no arm here
has yet achieved. The finisher is blocked on that, not on the machinery.

**At Very Hard the channel funds only from a decided game.** Mid-contest
withholding cost baseline wins even income-gated (`vh-nuke`, 90210); the
funding gate is now the closer's sustained-dominance commitment, the one
measured state where surplus is real. Targeting aims at the richest
250-radius circle of hostile structures, not the priciest single one --
an area weapon pointed at one building pays back less than it costs.
