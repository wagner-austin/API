---
title: "Engine Entity Model"
tags: [engine, units, entities, players, obfuscation, decompilation]
related:
  - "[[engine-tick-and-clock]]"
  - "[[engine-name-oracle]]"
  - "[[runtime-split-java-agent-python-brain]]"
source_paths:
  - "wiki/sources/m4-entities/entity-count-loop.txt:11"
  - "wiki/sources/m4-entities/entity-count-loop.txt:12"
  - "wiki/sources/m4-entities/entity-count-loop.txt:31"
  - "wiki/sources/m4-entities/entity-count-loop.txt:34"
  - "wiki/sources/m4-entities/entity-count-loop.txt:37"
  - "wiki/sources/m4-entities/player-class.txt:18"
  - "wiki/sources/m4-entities/live-graph-search.txt:9"
game_version: "1.15 (code 176, build #28)"
fact_checked: "2026-07-25"
confidence: high
hubs: [engine-internals, game-mechanics]
---

# Engine Entity Model

Every world object — units and trees alike — is a `com.corrodinggames.rts.game.units.am`, and the master list of them is the static field `am.bE`.[^1][^3] Ownership, position and the unit/tree distinction all hang off that one class, which makes `am` the type the agent's perception layer will serialise ([[runtime-split-java-agent-python-brain]]).

## The census that names everything

The engine's own post-load census settles the model in six lines: it iterates `am.bE`, counts an element as a tree when it is `instanceof al` and as a unit otherwise, and skips elements whose `bX` is not the current player.[^1][^2] That single loop identifies the list, the base class, the tree subclass, the owner field and the position pair at once, because it uses all of them.

- `am.bE` — `public static final u bE = new u()`, the master entity list.[^3]
- `am` — the entity base class; units and trees are both `am`.[^1]
- `al` — the **tree** class, not the unit class.[^2]
- `am.bX` — the owning player, typed `game.n`.[^1]
- `am.eo`, `am.ep` — world position, passed straight to the camera-centring call.[^1]

`al` is confirmed as trees independently of the census: it loads `palm_tree`, `trees` and `trees_snow` drawables and throws `"Tree sub type format error:"` on a malformed subtype.[^4][^5]

## Players

`com.corrodinggames.rts.game.n` is the player class, and it says so itself. It carries a field initialised to `4000.0` beside a string literal addressed to modders — *"Changing credits will not allow you to cheat in multiplayer games, but it will only break sync"* — which names the credits field and confirms the lockstep model in the same breath.[^6] Players are held in a static `n[]`, and the engine's `bs` field points at the current one.[^6]

## Why the first two answers were wrong

This page exists because two earlier readings were confidently wrong, and both failures are instructive.[^7]

A sprite registry was read as the unit list because its size, eleven, sat plausibly next to the ten units the map reported.[^7] Size coincidence is not identification. A graph node class was then read as the unit class because its query methods had the right shape. Neither survived contact with the decompiled census.

The correction came from decompiling rather than disassembling. `javap` answers narrow questions well — it is what proved the tick increment — but it does not make relationships legible, and relationships were the entire question here. Decompiling the jar with CFR turned a twenty-step inference into three greps, the decisive one being a search for the log string the engine already prints.[^1]

## Reproducing this

The decompiled tree is a derived work of a commercial game and is deliberately not versioned; only the cited excerpts are.[^1] Regenerate it with `java -jar cfr.jar .game/game-lib.jar --outputdir runs/decompiled`, which is gitignored ([[engine-name-oracle]] on why every name here expires with the build).

[^1]: `wiki/sources/m4-entities/entity-count-loop.txt:11` — `for (com.corrodinggames.rts.game.units.am am4 : com.corrodinggames.rts.game.units.am.bE)`, with the owner and position use `if (am4.bX != this.bs || !am4.bP) continue; this.b(am4.eo, am4.ep);` following it, and the emitted string `"there are " + n3 + " units on this map and " + n11 + " trees"` at `:27`.
[^2]: `wiki/sources/m4-entities/entity-count-loop.txt:12` — `if (am4 instanceof al) { ++n11; } else { ++n3; }`, where `n11` is the tree count and `n3` the unit count in the emitted line at `:27`.
[^3]: `wiki/sources/m4-entities/entity-count-loop.txt:31` — `public static final u bE = new u();` declared on `com.corrodinggames.rts.game.units.am`.
[^4]: `wiki/sources/m4-entities/entity-count-loop.txt:34` — `al.a[0] = l2.bO.a(R$drawable.palm_tree);`, with `trees` at `:35` and `trees_snow` at `:36`.
[^5]: `wiki/sources/m4-entities/entity-count-loop.txt:37` — `throw new RuntimeException("Tree sub type format error:" + stringArray[1]);` inside `al`.
[^6]: `wiki/sources/m4-entities/player-class.txt:18` — `public double o = 4000.0;` on `com.corrodinggames.rts.game.n`, with the modder-addressed string literal at `:15` and `static n[] b = new n[0];` at `:4`.
[^7]: `wiki/sources/m4-entities/live-graph-search.txt:9` — the agent's graph search reporting `.bs.s.bE  u size=221 of=com.corrodinggames.rts.game.units.al`, with two further 221-element views of the same list at `:10` and `:13`; 221 against a map reporting 10 units and 206 trees is what the `al`-is-a-tree finding explains.
