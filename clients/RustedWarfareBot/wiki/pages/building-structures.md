---
title: Building Structures
tags: [engine, commands, dispatch, building, economy]
related:
  - "[[issuing-orders]]"
  - "[[engine-entity-model]]"
  - "[[runtime-split-java-agent-python-brain]]"
source_paths:
  - "wiki/sources/m8-build/placement-setter.txt:3"
  - "wiki/sources/m8-build/waypoint-validator.txt:8"
  - "wiki/sources/m8-build/build-action-lookup.txt:11"
  - "wiki/sources/m8-build/build-rejected-selector-zero.txt:6"
  - "wiki/sources/m8-build/build-succeeded.txt:20"
  - "wiki/sources/m8-build/buildable-type-names.txt"
  - "agent/src/rwbot/agent/Orders.java"
source_git_blobs:
  "wiki/sources/m8-build/placement-setter.txt": "4d2604ee8e85ccafcf01fc09370802bdabc4d42b"
  "wiki/sources/m8-build/waypoint-validator.txt": "5445542aa1a61e25f913cea92c853051d0216100"
  "wiki/sources/m8-build/build-action-lookup.txt": "a4d157cf481abecc3d9e659387d1b5bbcd8611ce"
  "wiki/sources/m8-build/build-rejected-selector-zero.txt": "901a8300156e03ff2db9fa06229d4dda3fc93c03"
  "wiki/sources/m8-build/build-succeeded.txt": "8e81f717d133995332ddcfde4b4728f82b7c6ab9"
  "wiki/sources/m8-build/buildable-type-names.txt": "92138d74643c4b62bdcd61a17cff55400135c5d4"
  "agent/src/rwbot/agent/Orders.java": "846c66b42fcf439dc5ad3534424b42d0da6d598a"
game_version: "1.15 (code 176, build #28)"
fact_checked: 2026-08-17
confidence: high
hubs: [engine-internals, game-mechanics]
---

# Building Structures

Construction is the economic verb: a bot that can only move cannot open a game. It turned out to reuse the move machinery entirely, differing in one setter and one integer ([[issuing-orders]]).

## It is not a special action

The special-action vocabulary is unit abilities. The game's own translation table enumerates them — reclaim, repair, patrol, guardUnit, setRally, unload, dive, fly, surface, upgradeT2, buildNuke, launchNuke — and carries nothing for placing a structure.[^1]

Placement instead rides the same waypoint slot a move order uses, in a third target mode. The setter takes a position, a unit type and an integer, and sets the target kind to the same one the engine's system-spawn path requires:[^2]

```java
e cmd = engine.cf.a(team);
cmd.a(builder);                               // subject
cmd.a(x, y, type, Orders.ANY_BUILD_ACTION);   // place type here
```

## The integer is a selector, not a rotation

Reading it as a rotation cost a run. A builder holds a list of build actions, and the engine matches one by type **and** by that integer, short-circuiting only when it is `-1`:[^3]

```java
if (as4 != as2 || n2 != -1 && n2 != s3.t()) continue;
```

Passing `0` asks for the action whose own index is `0`. When no action carries that index the lookup returns null, waypoint validation rejects the order, and nothing happens.[^4] `-1` means "any action that builds this type" and is the only value that does not require knowing a builder's internal action ordering.

## The validator names its own reasons

Build waypoints are checked before acceptance, and each rejection has a distinct message: a missing build type, a builder that cannot queue the type at all, a locked building, an unavailable one.[^5] Those strings are the fastest diagnostic available for a rejected placement — the caller only logs `isValidNewWaypoint==false`, which says a waypoint was refused but not why.

## What a successful build looks like

Ordering the Builder to place a `landFactory` 200 east and 120 south of itself:[^6]

| sample | builder | roster |
|---|---|---|
| t+0 | (4250.0, 2610.0) | 3 entities |
| t+2s | (4323.6, 2654.1) | 3 entities |
| t+5s | (4380.6, 2688.3) | **4** — `units.d.m` at (4450.0, 2730.0) |
| t+10s | (4380.6, 2688.3) | 4 |

The builder pathfinds toward the site, the structure appears at exactly the requested coordinates, and the builder then stops — in range, constructing. The new entity's drawables are `land_factory_back`, `land_factory_front` and `land_factory_dead`, which identifies it independently of the type name that was asked for.[^7]

## The buildable vocabulary

Ninety type names are registered, extracted from the engine's own `-printunits` catalogue where each entry carries `unit:<name>`.[^8] Names resolve through a registry lookup that tries mod-defined types, then a built-in enum, then aliases.

The built-in enum arm can never match. Its constants are obfuscated to single letters and it compares against `Enum.name()`, so `"landFactory"` cannot equal `"m"`. Every name that resolves does so through the `.ini`-defined registry — which is where the built-in units live too, since the engine loads them from `builtin_mods` at boot ([[engine-entity-model]]).

## Selection stays with the planner

The agent takes a type name and a position and dispatches. It does not decide what to build, where a structure will fit, or whether the player can afford it — the engine's own validator answers the last of those, and the first two are planner work ([[runtime-split-java-agent-python-brain]]).

[^1]: `.game/assets/translations/Strings.properties` — the `gui.actions.*` key set, 55 keys covering unit abilities with no structure-placement entry among them.
[^2]: `wiki/sources/m8-build/placement-setter.txt:3` — `this.a = av.c;` in `au.a(float, float, as, int)`, which also stores the type at `:6` and the selector at `:7`.
[^3]: `wiki/sources/m8-build/build-action-lookup.txt:11` — `if (as4 != as2 || n2 != -1 && n2 != s3.t()) continue;` inside the builder's build-action lookup.
[^4]: `wiki/sources/m8-build/build-rejected-selector-zero.txt:6` — `isValidNewWaypoint==false on: builder(pos:4250,2610 id:214 t:0)` after the order at `:3`; the builder never left `(4250.0, 2610.0)`, recorded at `:5`.
[^5]: `wiki/sources/m8-build/waypoint-validator.txt:8` — the `av.c` branch, with `"Skipping build waypoint with no buildType"` at `:11`, `"can not queue build:"` at `:18`, `"tried to queue a locked building:"` at `:25` and `"tried to queue a unavailable building:"` at `:31`.
[^6]: `wiki/sources/m8-build/build-succeeded.txt:20` — `[3] com.corrodinggames.rts.game.units.d.m at (4450.0, 2730.0)` first present in the t+5s roster, against the order recorded at `:7`; builder positions at `:15` and in the t+2s roster.
[^7]: Decompiled `com/corrodinggames/rts/game/units/d/m.java` [synthesis] — its drawable references are `land_factory_back`, `land_factory_front`, `land_factory_front_t` and `land_factory_dead`. The decompiled tree is gitignored; regenerate with `make decompile`.
[^8]: `wiki/sources/m8-build/buildable-type-names.txt` — 90 names, extracted from the `unit:<name>` image references in the `-printunits` catalogue archived at `wiki/sources/m0-probe/printunits.log`.
