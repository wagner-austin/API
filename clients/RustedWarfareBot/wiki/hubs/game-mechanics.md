# Game Mechanics

How Rusted Warfare actually plays: unit and building stats, the economy (credits, extractors, income curves), the tech/build tree, movement layers (ground, air, water, hover) and what each can traverse, fog of war and vision, combat resolution, and map structure.

Scope: the rules of the game as a player experiences them. The code that implements them is [Engine Internals](engine-internals.md); how the bot exploits them is [Bot Architecture](bot-architecture.md).

The game is its own oracle here. `-printunits` emits a complete stat catalogue on demand, and the mod `.ini` files under `.game/mods/` are declarative source for custom units — prefer both over inferring numbers from play. Stats change between builds and are affected by enabled mods, so pin `game_version` and state which mods were active.

<!-- Add pages here as they're written. Format: [Title](../pages/<slug>.md) -- one-line description -->
