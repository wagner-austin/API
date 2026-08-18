---
title: Engine Name Oracle
tags: [engine, obfuscation, mapping, boot, reverse-engineering]
related:
  - "[[harness-nodisplay]]"
  - "[[engine-ai-zones]]"
  - "[[mechanics-build-actions]]"
source_paths:
  - "wiki/sources/m0-probe/nodisplay-boot.log:45"
  - "wiki/sources/m0-probe/jar-classes.txt:380"
  - "wiki/sources/m16-enums/enum-names.txt"
source_git_blobs:
  "wiki/sources/m0-probe/nodisplay-boot.log": "c41e035d12ee89b66455389031be9fef55cf0b44"
  "wiki/sources/m0-probe/jar-classes.txt": "ff1afd0e174df4a07930aad57b22cb14eca511ec"
  "wiki/sources/m16-enums/enum-names.txt": "919c505b3f4d41a4c8dc599a37653ddbfb4e5e1e"
game_version: "1.15 (code 176, build #28)"
fact_checked: 2026-08-17
confidence: high
hubs: [engine-internals]
---

# Engine Name Oracle

`game-lib.jar` is obfuscated, but not uniformly, and the running game leaks most of the mapping for free. Recovering usable class names is a matter of reading the boot log rather than decompiling all 1698 classes in the jar.[^1]

## What the obfuscator did

Package paths survived; class names did not. `com/corrodinggames/rts/gameFramework/` contains 93 classes named `a` through `bu`, plus exactly one readable name, `SettingsEngine`. The same shape holds in `gameFramework/`, `game/`, `game/units/`, `appFramework/` and `java/` — single-letter classes inside meaningful package paths.[^1]

Two areas escaped entirely, and both for the same reason: something instantiates them by name at runtime, so the obfuscator was told to keep them.[^1]

`com/corrodinggames/librocket/scripts/` holds `Root`, `Multiplayer`, `Mods`, `Debug`, `ScriptEngine` and `ScriptContext` with names intact.[^2] The game's menus are RML documents that dispatch into these objects — the boot log shows the engine processing `onShowNewScreen();` through the script engine as the main menu loads.[^3]

`com/corrodinggames/rts/game/units/custom/logicBooleans/` holds 215 unobfuscated classes — `LogicBoolean`, `CompareJoinerBoolean`, `BooleanParseException`, and inner classes including `CallContext_self`, `CallContext_selfAndTarget` and the arithmetic and comparison joiners.[^4] This is an expression language evaluated against units, and the mod `.ini` parser resolves its operators by name.

## The oracle

The engine narrates its own construction. During init the log emits a `--Now loading:<Name>` line for each subsystem, using the readable name: `SettingsEngine` at line 60, `GroupController` at 146, `CollisionEngine` at 147, `InterfaceEngine` at 148, `StatsHandler` at 171, `ModEngine` at 172, `GameSaver` at 184, `UnitGeoIndex` at 186 — alongside GameEngine, PathEngine, NetworkEngine, CommandController and ReplayEngine.[^5]

More usefully, at least one line prints the readable name and the obfuscated class together: **`GameEngine` is `com.corrodinggames.rts.game.i`**.[^6] That class is present in the jar inventory as expected.[^7]

So the mapping strategy is: run the game, read the narration, and use the construction order plus the printed mapping to anchor a decompiler pass — rather than decompiling first and guessing what each single-letter class does.[^9]

## A second oracle: every enum names itself

The obfuscator renamed enum **fields** and left their **constant names** alone. An enum's constant name is the string passed to the `Enum` constructor in the class's own static initialiser, and ProGuard did not rewrite those strings — so a decompile that shows nothing but `enum j { a, b, c; }` sits on a class whose bytecode still carries `Pre`, `Prepare`, `Active`.[^11]

Recovering them needs no running game and no decompiler:

```
javap -p -c -cp game-lib.jar com.corrodinggames.rts.game.a.j | grep '// String '
```

Fifty-three enums across `com.corrodinggames.*` yield to this, and the result is archived.[^12] Several are worth more than the class names this page was written about:

- `game.units.ao` — **`NONE LAND BUILDING AIR WATER HOVER OVER_CLIFF OVER_CLIFF_WATER`**. The movement-layer model, which [[mechanics-resource-pools]] records as the missing half of the reachability problem. It was in the jar the whole time.
- `game.units.av` — the full order vocabulary: `move attack build repair loadInto unloadAt reclaim attackMove loadUp patrol guard guardAt touchTarget follow …`, which bounds what a command can ever ask for ([[issuing-orders]]).
- `game.units.a` — attack stances: `outOfRange onlyInRange returnFire holdFire guardArea aggressive mixed`.
- `game.units.a.t` and `a.u` — the action taxonomy behind the two build verbs: `none rally upgrade queueUnit building action infoOnly …` and `none placeBuilding popupQueue setRally reclaimTarget repairTarget targetGround attackMove …` ([[mechanics-build-actions]]).
- `game.a.j` and `game.a.k` — the AI's zone state and kind: `Pre Prepare Active`, and `Main ResourceOutpost ForwardOutpost` ([[engine-ai-zones]]).

**This is a strictly better oracle than the boot log for the things it covers.** The boot log names classes the engine happens to announce; this names every constant of every enum, from the jar alone, reproducibly, with no game running. It found the movement layers that three pages had recorded as unavailable.

It generalises to a rule worth applying before inferring anything about an obfuscated enum: **check whether the value already tells you what it is.** The same trick is why the AI's debug overlay is readable — string literals survived wholesale — and the two together cover most of what the engine calls its own concepts.

## Why this expires

Every name recovered this way is a fact about build 1.15, game code 176, build #28 — the version the probe run reports.[^10] Obfuscators reassign letters on each release, so `game.i` is meaningless in 1.16 and nothing will error; the agent simply binds to the wrong class. This is why the working copy is pinned at `.game/` outside Steam's update path, and why every page here carries `game_version` ([[harness-nodisplay]]).

Community mapping sets exist for older builds (TapeRTS/Tape publishes CC0 mappings with per-version folders topping out around 1.12b) and are useful as a cross-check for names that survived several releases, but they do not cover this build.[^8]

## What this does not give us

The narration names subsystems, not methods. Knowing `GameEngine` is `game.i` does not identify its tick method, its unit list, or the entry point on `CommandController` that queues an order[^6] — those still require decompilation, and they are the actual prerequisites for both a state-reading agent and the pure-dispatch rule in [[multiplayer-portability-invariants]]. The oracle removes the "which of 1698 classes do I even open" problem; it does not remove the work behind it.

[^1]: `wiki/sources/m0-probe/jar-classes.txt` — full sorted inventory of all 1698 `.class` entries in `.game/game-lib.jar`, extracted from the zip central directory.
[^2]: `wiki/sources/m0-probe/jar-classes.txt:236` — "com/corrodinggames/librocket/scripts/Multiplayer.class", with `Root.class` at `:251` and `ScriptEngine.class` at `:256`.
[^3]: `wiki/sources/m0-probe/nodisplay-boot.log:305` — "ScriptEngine:HandleEvent:onShowNewScreen();".
[^4]: `wiki/sources/m0-probe/jar-classes.txt:689` — "com/corrodinggames/rts/game/units/custom/logicBooleans/LogicBoolean.class"; the surrounding block lists the 215 entries under that package.
[^5]: `wiki/sources/m0-probe/nodisplay-boot.log:60` — "--Now loading:SettingsEngine"; the remaining subsystem lines follow at the line numbers given, through `:186`.
[^6]: `wiki/sources/m0-probe/nodisplay-boot.log:45` — "Created new gameEngine of:com.corrodinggames.rts.game.i".
[^7]: `wiki/sources/m0-probe/jar-classes.txt:380` — "com/corrodinggames/rts/game/i.class".
[^8]: https://github.com/TapeRTS/Tape — repository description "Rusted Warfare RTS mappings, free to use for everyone", CC0, with per-version directories (0.80, 1.09, 1.12b). Last pushed 2023-02-19; no 1.15 directory. Retrieved 2026-07-25.
[^9]: [synthesis] — conclusion drawn from the printed mapping at `wiki/sources/m0-probe/nodisplay-boot.log:45` plus the subsystem narration at `:60`–`:186`; no separate source.
[^10]: `wiki/sources/m0-probe/nodisplay-boot.log:12` — "Game Version: 1.15", with "Build Number: #28" at `:11` and "Game Code: 176" at `:13`.
[^11]: `runs/decompiled/com/corrodinggames/rts/game/a/j.java:6` shows `enum j { a, b, c; }`, while `javap -p -c` on the same class emits `ldc // String Pre`, `// String Prepare` and `// String Active` from its `<clinit>`.
[^12]: `wiki/sources/m16-enums/enum-names.txt` — 53 enums with their constants, and the one-line command that regenerates the file from the jar.
