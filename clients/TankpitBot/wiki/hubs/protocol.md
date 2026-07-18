# Protocol

Wire protocol format, message types, and client data structures. Everything learned from sniffing WebSocket traffic and reading tpclient JS.

[ShootEvent Format](../pages/shoot-event-format.md) -- 0x53 wire layout (team, shooter, source pos, target pos, weapon), hit/miss disambiguation, damage tiers
[Deactivation Format](../pages/deactivation-format.md) -- 0x41 kill detection: status, victim, promo_eligible, killer + mine-kill sentinel; never fires for own kills
[MAP DATA Decode](../pages/map-data-decode.md) -- 0x14 blob, skip-RLE fuel dot atlas, tank entries, cache semantics
[Tank Registry](../pages/tank-registry.md) -- activeGame.P.j fields: damage tier, team, viewport position, rank points
[Weapon Selection](../pages/weapon-selection.md) -- server decides weapon type based on what's at the fired tile
[Decode Coverage Map](../pages/decode-coverage.md) -- every message type vs our decoder: gaps, wrong constants, dropped fields (2026-06-18 audit)
[rank_category Bug](../pages/rank-category-bug.md) -- damage_state field in 3 decoders is actually rank_category; combat targeting affected (2026-06-19)
[Viewport Shift Protocol](../pages/viewport-shift-protocol.md) -- the three shift triggers (teleport, Rb/Sb scope commands, Ia autoscroll) + 22-event corpus proof; the bot uses none of them
