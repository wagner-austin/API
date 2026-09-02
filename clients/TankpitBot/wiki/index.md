# TankpitBot Wiki

**Read this first.** 6 topic hubs, 80 content pages. Follow the hub link for your topic; each hub lists its pages with one-line descriptions.

## Hubs

[Game Mechanics](hubs/game-mechanics.md) -- how the game works: viewport, teleport, radar, fuel, ferries, map, equipment, official rules, movable blocks, walk mechanics (10 pages)
[Protocol](hubs/protocol.md) -- wire format: combat hits, deactivation, MAP_DATA, viewport entities, tank registry, weapon selection, decode coverage, viewport shift, server push gating, the capture differ, serve cadence, recipient policy (12 pages)
[Combat](hubs/combat.md) -- fighting strategy: shot range, enemy behavior, weapon selection, gameplay loop, equipment refill, mine mechanics, movable blocks, game economy, tournament strategy, serve cadence, flag triage 2026-09-02 (14 pages)
[JS Client](hubs/js-client.md) -- reverse-engineered tpclient.js: source map, commands, V table, constants, state machine, XOR, terrain, chat, connection, more (21 pages)
[Architecture](hubs/architecture.md) -- codebase decisions: inheritance chain, DI, test hooks, coding standards, tank freshness model, bot behavior contract, self-observing architecture, bot service, executor rejection loops, terrain composition, physics roadmap, committed intent, diagnostic HUD + flag channel, flag triage, larder plan, session-state de-globalisation, package layering, quad-sweep doctrine, guard mutation sweep, project history, fleet coordination, fleet lifecycle, fleet live reads, fleet forage allocation, flag triage 2026-09-02 (23 pages)
[Codebase](hubs/codebase.md) -- module map, services, testing patterns, make targets, how to add a probe, guard mutation sweep, project history (7 pages)

## How this works

**Three tiers:** this index (read every session) -> hub pages (read when topic matches) -> content pages (read when you need the facts). A content page can be linked from multiple hubs.

**Adding new content:** create the content page in `pages/`, add a link from the relevant hub(s), update the page count here and in the hub.

**After a live run:** update existing content pages with new findings (run IDs, verified constants). Create a bug page if a new behavior is diagnosed. Log the operation in `log.md`.

**Schema:** v1.1 (2026-07-31). Karpathy LLM-wiki + IWE pattern. See SCHEMA.md — frontmatter, provenance, navigation, and counts are guard-enforced on every `make check`.
