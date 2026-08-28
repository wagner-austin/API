# Combat

Fighting strategy, weapon behavior, enemy AI patterns, and diagnosed combat bugs.

[Shot Range](../pages/shot-range.md) -- Manhattan 1 proven (255/255), range 2 has 1 sample, 4+ all miss, cardinal adjacency required
[Enemy Bot Behavior](../pages/enemy-bot-behavior.md) -- bots stand ground, only flee at low HP, never collect resources, never fight each other
[Weapon Log Markers](../pages/weapon-log-markers.md) -- dual="You hit", homing="You fire", no miss lines in range
[Weapon Selection](../pages/weapon-selection.md) -- server-side: dual at adjacent, homing when target moves same tick, single at empty ground
[Serve Cadence](../pages/serve-cadence.md) -- one action per 2 s global beat, moves share the slot, excess dispatches queue
[Combat Chase Bug](../pages/combat-chase-bug.md) -- teleport-chase loop diagnosed and fixed: teleport directly to target
[Gameplay Loop](../pages/gameplay-loop.md) -- the full combat→refill→radar conservation cycle as played by a human
[Equipment Refill Strategy](../pages/equipment-refill-strategy.md) -- low-radar grid walk, extra radar conservation, container randomness
[Movable Concrete Blocks](../pages/movable-blocks.md) -- shot shielding (one angle only, missiles ignore it), mine destruction on placement; user contract 2026-07-20
[Mine Mechanics](../pages/mine-mechanics.md) -- 3x3 placement filter, mine-on-mine destruction, cascade chain detonation, real-combat wire evidence 2026-06-20
[Game Economy](../pages/game-economy.md) -- empirical fuel costs (walk 1/tile, single shot 6, radar 10), damage taken (single 45, dual 90, mine 45), max fuel cap 1100, container_pickup remaining_volume semantic (2026-06-20)
[Tournament Strategy (Sigma v3.4)](../pages/tournament-strategy.md) -- preserved 2015 human tournament meta: initial fill, fill-fighting, kill types, PPH, equipment management, endgame shield-fighting
[Flag Triage 2026-07-29](../pages/flag-triage-20260729.md) -- forage-economy findings from the first flag session: 63% zero-yield hops, direction-blind top-off hop, mine-covered equipment counterplay (rank-gated blast in [[mine-mechanics]])
