# Clients

User-facing clients that consume the platform's service surface. Three live in `clients/`:

- **DiscordBot** — integrates all platform services behind a Discord command interface (`platform_discord`, `platform_core`, `platform_workers`).
- **TankpitBot** — tankpit.com WebSocket protocol reverse-engineering and game bot logic (`platform_core`).
- **RustedWarfareBot** — headless Rusted Warfare client: a Java agent inside the game's JVM dispatches orders and serialises simulation state, a Python package plans and evaluates. Deliberately standalone — `monorepo_guards` is its only in-repo dependency.

This hub covers each client's architecture, its integration surface into the shared platform libs, and client-specific quirks.

## The game clients have their own dedicated wikis

Both game bots maintain full three-tier, schema-conformant wikis. They are the source of truth for their own domains — this hub does NOT restate any of it.

- [TankpitBot wiki index](../../clients/TankpitBot/wiki/index.md) — 6 hubs, 67 content pages: game mechanics, protocol, combat, JS client, architecture, codebase.
- [RustedWarfareBot wiki index](../../clients/RustedWarfareBot/wiki/index.md) — engine internals, the headless harness, game mechanics, bot architecture, multiplayer constraints. Claims about engine internals are pinned to a game build (`game_version` frontmatter) because the jar is obfuscated and class names change silently between releases.

Cross-platform client facts that would belong *here* rather than in a client-specific wiki are integration surfaces into the monorepo — `platform_core.logging` config, `monorepo_guards` conformance, hypothetical future `platform_workers` job-enqueue paths. None of that has been written yet; add pages here as those surfaces materialise.

<!-- Add pages here as they're written. Format: [Title](../pages/<slug>.md) -- one-line description -->
