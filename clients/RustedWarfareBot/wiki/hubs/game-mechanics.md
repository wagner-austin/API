# Game Mechanics

How Rusted Warfare actually plays: unit and building stats, the economy (credits, extractors, income curves), the tech/build tree, movement layers (ground, air, water, hover) and what each can traverse, fog of war and vision, combat resolution, and map structure.

Scope: the rules of the game as a player experiences them. The code that implements them is [Engine Internals](engine-internals.md); how the bot exploits them is [Bot Architecture](bot-architecture.md).

The game is its own oracle here. `-printunits` emits a complete stat catalogue on demand, and the mod `.ini` files under `.game/mods/` are declarative source for custom units — prefer both over inferring numbers from play. Stats change between builds and are affected by enabled mods, so pin `game_version` and state which mods were active.

[Building Structures](../pages/building-structures.md) -- construction reuses the move machinery; the integer is a build-action selector, not a rotation

[The Policy Loop](../pages/policy-loop.md) -- the bot plays: pure decisions from observed state, one order per plan slot, and a scorecard

<!-- Add pages here as they're written. Format: [Title](../pages/<slug>.md) -- one-line description -->

[Engine Entity Model](../pages/engine-entity-model.md) -- what a unit actually is in the engine: base class, master list, trees, owning player, starting credits
[Unit Catalogue and the Mobility Predicate](../pages/mechanics-unit-catalogue.md) -- 90 units with prices, HP and weapons from the engine's own -printunits; speed>0 is the read mobility test
