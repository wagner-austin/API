# Game Mechanics

How Rusted Warfare actually plays: unit and building stats, the economy (credits, extractors, income curves), the tech/build tree, movement layers (ground, air, water, hover) and what each can traverse, fog of war and vision, combat resolution, and map structure.

Scope: the rules of the game as a player experiences them. The code that implements them is [Engine Internals](engine-internals.md); how the bot exploits them is [Bot Architecture](bot-architecture.md).

The game is its own oracle here. `-printunits` emits a complete stat catalogue on demand, and the mod `.ini` files under `.game/mods/` are declarative source for custom units — prefer both over inferring numbers from play. Stats change between builds and are affected by enabled mods, so pin `game_version` and state which mods were active.

[Building Structures](../pages/building-structures.md) -- construction reuses the move machinery; the integer is a build-action selector, not a rotation

[The Policy Loop](../pages/policy-loop.md) -- the bot plays: pure decisions from observed state, one order per plan slot, and a scorecard

<!-- Add pages here as they're written. Format: [Title](../pages/<slug>.md) -- one-line description -->

[Engine Entity Model](../pages/engine-entity-model.md) -- what a unit actually is in the engine: base class, master list, trees, owning player, starting credits
[Unit Catalogue and the Mobility Predicate](../pages/mechanics-unit-catalogue.md) -- 90 units with prices, HP and weapons from the engine's own -printunits; speed>0 is the read mobility test
[Resource Pools and the Placement Rule](../pages/mechanics-resource-pools.md) -- credits come from a tileset property: `res_pool` tiles, the `placeOnlyOnResPool` flag, and where an extractor may stand
[Build Actions: Two Families, Two Verbs, Five Gates](../pages/mechanics-build-actions.md) -- placing and producing are one mechanism dispatched two ways, and the five conditions that stop an order, four of which say nothing
[Threat: Choosing Ground the Builder Survives](../pages/policy-threat.md) -- pools are chosen by who can shoot the walk there, with hostility read from the engine's alliance test rather than from ownership
[The Shipped AI's Zone System](../pages/engine-ai-zones.md) -- the AI's unit of place: five zone kinds, one-unit-one-zone, and expansion sited on resource pools at random
[The Shipped AI's Build and Attack Triggers](../pages/engine-ai-triggers.md) -- when it builds, makes units and commits: a credit ladder, a unit budget, and fill-then-commit attack groups of 3, 5, 7
[Movement Layers and Reachability](../pages/mechanics-movement-layers.md) -- eight layers named by the engine, reachability as a component comparison, and the twelve pools no land builder can reach
[Combat Profiles — What Can Shoot What](../pages/mechanics-combat-profile.md) -- the engine's own attackability test, and the four submarines that cannot shoot the shore
[What a Credit Buys — The Unit Value Table](../pages/mechanics-unit-value.md) -- price against dps, hit points, reach and whether it can shoot at aircraft, joined from the catalogue and the combat dump
[Holding Ground - 44 of 46 Pools, and Why the Bot Loses](../pages/policy-holding-ground.md) -- who ends up owning the map's resource pools, and what that costs
