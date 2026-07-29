---
title: "Combat Profiles — What Can Shoot What"
tags: [mechanics, combat, reverse-engineering, wire]
related:
  - "[[policy-combat]]"
  - "[[policy-threat]]"
  - "[[mechanics-unit-catalogue]]"
  - "[[mechanics-movement-layers]]"
  - "[[engine-name-oracle]]"
source_paths:
  - "wiki/sources/m11-pools/type-flags.ndjson"
  - ".game/assets/units/tanks/tank.ini"
  - ".game/assets/units/extractor/extractor.ini"
  - "src/rw_bot/mechanics/combat_profile.py"
  - "agent/src/rwbot/agent/TypeFlags.java"
  - "agent/src/rwbot/agent/EngineNames.java"
game_version: "1.15 (code 176, build #28)"
fact_checked: 2026-07-26
confidence: high
hubs: [game-mechanics, bot-architecture, engine-internals]
---

# Combat Profiles — What Can Shoot What

Reach was only half the question, and the missing half could hang a match.

`c_tank` is the only unit the opening plan builds. Its own `.ini` says:

```
[attack]
canAttack: true
canAttackFlyingUnits: false
canAttackLandUnits:   true
canAttackUnderwaterUnits: false
maxAttackRange: 130
```

Nothing in the planner read the three `canAttack*` lines. Combat selected units
by *having* a weapon and never by the weapon *reaching the target*, so on a
water map the army could commit to a helicopter, hold it for as long as it
stayed visible — because commitment keeps a visible target — and never fire.

## The engine's own test, transcribed

The engine answers "can this shoot that" in one method on the orderable base
class, and it branches four ways.[^1] Rendered from the decompile:

```java
if (target.isFlying())      return this.canAttackAir();
if (target.isUnderwater())  return this.canAttackUnderwater();
if (!this.canAttackLandOutOfWater() && !target.isTouchingWater()) return false;
return this.canAttackLand();
```

`rw_bot.mechanics.combat_profile.can_engage` is that branch, transcribed rather
than modelled. **The attacker's four predicates ride on the type record; the
target's three states ride on the entity record**, so neither side is inferred.

The mapping from accessor to `.ini` key is confirmed through the asset loader
rather than guessed from ordering: the custom-unit overrides return the
`canAttackUnderwaterUnits`, `canAttackFlyingUnits` and `canAttackLandUnits`
predicates respectively.[^2]

| Pinned accessor | `.ini` key | Meaning |
| --- | --- | --- |
| `ae` | `canAttackUnderwaterUnits` | reaches a submerged target |
| `af` | `canAttackFlyingUnits` | reaches an airborne target |
| `ag` | `canAttackLandUnits` | reaches a ground target |
| `ah` | `canAttackNotTouchingWaterUnits` | ground fire reaches a target clear of water |

Target state is per-entity and **dynamic**, which is why it is sampled rather
than derived from the type once: a gunship that has landed answers `flying`
false and becomes shootable by units that cannot hit aircraft.[^3]

## What the live dump shows

All 173 registered types, from `make type-flags`:[^4]

| Type | range | land | air | underwater | out of water |
| --- | --- | --- | --- | --- | --- |
| `c_tank` | 130 | yes | **no** | no | yes |
| `antiAirTurret` | 250 | **no** | yes | no | yes |
| `heavySub` | 210 | yes | no | yes | **no** |
| `builder` | 0 | no | no | no | no |

Two rows changed real behaviour.

**`antiAirTurret` cannot hit the ground.** It is armed with a 250-unit reach, so
a threat model reading reach alone ruled out every resource pool within 250
units of one — ground the builder could have crossed safely.

**Every submarine carries a torpedo, not a gun.** `heavySub`, `lightSub`,
`nautilusSubmarine` and `c_amphibiousJet_underwater` are the only armed types in
the whole game with `hits_land_out_of_water` false, and they are exactly the
four that matter on a water map. A submarine lurking offshore is not a reason to
refuse a pool inland.

## Unarmed types report no layer at all

The base predicates return **true** for air and land regardless of armament,
because the engine only consults them once a weapon is established. Reporting
them unfiltered would put "a Builder can shoot aircraft" on the wire, so the
agent forces all four false when the type is unarmed.[^5]

That keeps zero an *answer* rather than an absence — the distinction that made
the stat catalogue unsafe to read reach from ([[mechanics-unit-catalogue]]).

## What this deliberately does not model

The predicates are read **on the prototype**, so a unit declaring
`canAttackCondition` as a logic expression evaluated against the live unit would
be answered against the prototype instead. Nothing in the base game's buildable
set does that. The alternative — asking per attacker-target pair every sample —
is a reflective call per pair per tick for an answer that does not change.

[^1]: `com/corrodinggames/rts/game/units/y.java:3086` in the decompiled tree —
    the four-branch predicate, with the base implementations at `:3042`-`:3058`
    returning false for underwater and true for air, land and out-of-water.
[^2]: `com/corrodinggames/rts/game/units/custom/j.java:2375`-`:2391`, whose
    overrides read `x.er`, `x.eq` and `x.es`; those three fields are assigned
    from `"canAttackLandUnits"`, `"canAttackFlyingUnits"` and
    `"canAttackUnderwaterUnits"` at `com/corrodinggames/rts/game/units/custom/ag.java:2233`-`:2235`.
[^3]: `com/corrodinggames/rts/game/units/b/c.java:251` implements the airborne
    test as `!this.b()`, and `:74` implements the submerged test as a height
    comparison — both state rather than type.
[^4]: `wiki/sources/m11-pools/type-flags.ndjson` — one `unitcombat` record per
    registered type, 173 of them.
[^5]: `agent/src/rwbot/agent/TypeFlags.java`, `combatOf` — returns
    `Combat.UNARMED` before reading any layer predicate.
