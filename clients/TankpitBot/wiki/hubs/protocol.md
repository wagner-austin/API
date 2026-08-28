# Protocol

Wire protocol format, message types, and client data structures. Everything learned from sniffing WebSocket traffic and reading tpclient JS.

[ShootEvent Format](../pages/shoot-event-format.md) -- 0x53 wire layout (team, shooter, source pos, target pos, weapon), hit/miss disambiguation, damage tiers
[Deactivation Format](../pages/deactivation-format.md) -- 0x41 kill detection: status, victim, promo_eligible, killer + mine-kill sentinel; fires for ALL kills incl. own (0x2E-tunneled; "never for own kills" falsified 2026-07-19)
[MAP DATA Decode](../pages/map-data-decode.md) -- 0x14 blob, skip-RLE fuel dot atlas, tank entries, cache semantics
[Tank Registry](../pages/tank-registry.md) -- activeGame.P.j fields: damage tier, team, viewport position, rank points
[Weapon Selection](../pages/weapon-selection.md) -- server decides weapon type based on what's at the fired tile
[Serve Cadence](../pages/serve-cadence.md) -- echo-measured: one served action per tank per 2 s, on a room-global grid
[Decode Coverage Map](../pages/decode-coverage.md) -- every message type vs our decoder: gaps, wrong constants, dropped fields (2026-06-18 audit)
[rank_category Bug](../pages/rank-category-bug.md) -- damage_state field in 3 decoders is actually rank_category; combat targeting affected (2026-06-19)
[Viewport Shift Protocol](../pages/viewport-shift-protocol.md) -- the three shift triggers (teleport, Rb scope pans the bot NOW SENDS, Ia autoscroll), the measured Rb anchor law, and the 22-event corpus proof
[Capture Differ](../pages/capture-differ.md) -- the sim-fidelity pipeline: longitudinal container atlas, --from-atlas world seeding, and the response-shape differ that mechanically finds sim law gaps (5 found+closed 2026-08-01)
[Server Push Gating](../pages/server-push-gating.md) -- play-to-receive: periodic push (0x2E/0x3F/0x47) flows only to acting clients; queries/keep-alives never count; seven-run proof (2026-07-24)
