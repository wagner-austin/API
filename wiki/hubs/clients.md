# Clients

User-facing clients that consume the platform's service surface. DiscordBot integrates all platform services behind a Discord command interface; TankpitBot handles tankpit.com WebSocket protocol reverse-engineering and game bot logic. This hub covers each client's architecture, its integration surface into the shared platform libs (`platform_discord`, `platform_core`, `platform_workers`, etc.), and client-specific quirks.

## TankpitBot has its own dedicated wiki

`clients/TankpitBot/wiki/` is a full three-tier wiki (schema-conformant, 6 hubs, 50+ content pages) covering the tankpit.com wire protocol, game mechanics, combat strategy, JS client reverse-engineering, and bot architecture. It is the source of truth for that client's domain — this hub does NOT restate any of it. If you want tank-side facts, start there:

- [TankpitBot wiki index](../../clients/TankpitBot/wiki/index.md) — game mechanics, protocol, combat, JS client, architecture, codebase hubs.

Cross-platform TankpitBot facts that would belong here (rather than in the client-specific wiki) are things like how TankpitBot integrates into the api monorepo's shared libs — e.g. `platform_core.logging` config, `monorepo_guards` conformance, hypothetical future `platform_workers` job-enqueue paths. None of that has been written yet; add pages here as those integration surfaces materialise.

<!-- Add pages here as they're written. Format: [Title](../pages/<slug>.md) -- one-line description -->
