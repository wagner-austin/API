# Game Mechanics

How TankPit works at the game level. Viewport geometry, movement, scanning, fuel, equipment, map, and ferries.

[Viewport Frame](../pages/viewport-frame.md) -- fixed 16x16 actionable frame, 18x18 observable with radar fringe, recenters on edge walk
[Walk Mechanics](../pages/walk-mechanics.md) -- server movement is INSTANT (route, billing, pickup all in one tick); walking on screen is client animation that input-locks humans but not the bot
[Teleport Mechanics](../pages/teleport-mechanics.md) -- definitive placement, fuel cost, scatter on obstruction, landing auto-pickup
[Radar Mechanics](../pages/radar-mechanics.md) -- extra=full viewport, built-in=5x5, auto-consumes extras, death spiral at 0
[Movable Concrete Blocks](../pages/movable-blocks.md) -- pickup-and-place terrain: bridge on water, obstacle on land/stacked, destroys mines on its tile, blocks non-missile shots; user contract 2026-07-20, NOT yet wire-verified; bot has no knowledge of them
[Fuel System](../pages/fuel-system.md) -- thresholds, fuel dots (MAP_DATA atlas), dot freshness ~40%, marooning hazard
[Equipment System](../pages/equipment-system.md) -- container types, priority when depleted, equipment-only refill for radars
[Map Mechanics](../pages/map-mechanics.md) -- CMD_MAP_OPEN is one-way, no wire close, client-side toggle, server-cached data
[Ferry Mechanics](../pages/ferry-mechanics.md) -- free water movement, boarding/landing each cost one queue slot, terrain type 5
[Game Rules](../pages/game-rules.md) -- official How To Play screens: rank/promotion table, equipment capacity, demotion, radar scales with rank
