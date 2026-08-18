# Engine Internals

What is actually inside `game-lib.jar`: the engine objects the game constructs at boot, how the ProGuard obfuscation was applied and where it wasn't, the reflection-driven surfaces that survived with readable names, and the techniques for recovering a mapping without decompiling everything.

Scope: the JVM-side game code (`com.corrodinggames.rts.*`) and its libRocket script bindings. The rules the engine *implements* (unit costs, movement, combat) belong in [Game Mechanics](game-mechanics.md); how we launch the process belongs in [Headless Harness](headless-harness.md).

Every page here is pinned to a specific game build. An obfuscated class name is a fact about one build only.

[Engine Name Oracle](../pages/engine-name-oracle.md) -- the boot log names engine objects and prints their obfuscated classes, replacing most manual mapping work
[Engine Entity Model](../pages/engine-entity-model.md) -- `am` is every world object, `am.bE` the master list, `al` the tree class; owner and position fields named
[Engine Tick Method and Clock](../pages/engine-tick-and-clock.md) -- `game.i.a(float)` is the simulation tick; `bx` counts it at ~300 Hz, `by` is the millisecond clock, and five other writers only restore them

[Issuing Orders](../pages/issuing-orders.md) -- the three-call command path, its threading rule, and the order that finally moved a unit

[Building Structures](../pages/building-structures.md) -- construction reuses the move machinery; the integer is a build-action selector, not a rotation

[Command Channel](../pages/command-channel.md) -- orders originate in Python: one loopback socket, id-addressed units, and the backpressure rule that protects the tick

[Unit value table](../pages/mechanics-unit-value.md) -- per-type worth derived from the live type registry dump, with the arithmetic error that was nearly acted on recorded beside it
[Economy policy](../pages/policy-economy.md) -- reserve sizing against the room settings object whose b() dumps startingCredits, and what refusals actually signal
<!-- Add pages here as they're written. Format: [Title](../pages/<slug>.md) -- one-line description -->
[Perception: Visible Entities, Economy and Health](../pages/perception-visibility.md) -- `am.d(n)` is the engine's per-player fog test; credits, team and hit-point fields
[Resource Pools and the Placement Rule](../pages/mechanics-resource-pools.md) -- credits come from a tileset property: `res_pool` tiles, the `placeOnlyOnResPool` flag, and where an extractor may stand
[The Build Tree, and Planning From Goals](../pages/mechanics-build-tree.md) -- what each type can make, dumped from the registry; the static half the option stream cannot answer
[Build Actions: Two Families, Two Verbs, Five Gates](../pages/mechanics-build-actions.md) -- placing and producing are one mechanism dispatched two ways, and the five conditions that stop an order, four of which say nothing
[How the Built-in AI Plays](../pages/ai-opponent-strategy.md) -- the opponents' own code: a weighted unit mix, and attacks that delay, mass, then commit
[The Shipped AI's Zone System](../pages/engine-ai-zones.md) -- the AI's unit of place: five zone kinds, one-unit-one-zone, and expansion sited on resource pools at random
[The Shipped AI's Build and Attack Triggers](../pages/engine-ai-triggers.md) -- when it builds, makes units and commits: a credit ladder, a unit budget, and fill-then-commit attack groups of 3, 5, 7
[Fighting: the Attack Verb, and Keeping an Army Alive](../pages/policy-combat.md) -- av.b is attack, proven by a target losing exactly the attacker's catalogued damage
[The AI Zone Probe, and Why It Is Not Perception](../pages/engine-ai-probe.md) -- reading the opposing AI's plans is research, never perception: why, and the two structural guarantees that keep it that way
[Combat Profiles — What Can Shoot What](../pages/mechanics-combat-profile.md) -- the pinned layer predicates, traced to their .ini keys through the asset loader
