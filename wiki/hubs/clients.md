# Clients

Clients in `clients/`. Three consume the platform's service surface; the fourth consumes nothing and is a measurement instrument rather than a user-facing client:

- **DiscordBot** — integrates all platform services behind a Discord command interface (`platform_discord`, `platform_core`, `platform_workers`).
- **TankpitBot** — tankpit.com WebSocket protocol reverse-engineering and game bot logic (`platform_core`).
- **RustedWarfareBot** — headless Rusted Warfare client: a Java agent inside the game's JVM dispatches orders and serialises simulation state, a Python package plans and evaluates. Deliberately standalone — `monorepo_guards` is its only in-repo dependency.
- **NavProbe** — a reproducibility instrument for simulated navigation, not a bot: it drives someone else's simulator (MJX on JAX, MuJoCo-Warp) and answers whether a fixed seed and a fixed action sequence produce the same bytes. Standalone like RustedWarfareBot — `monorepo_guards` is its only in-repo dependency.

This hub covers each client's architecture, its integration surface into the shared platform libs, and client-specific quirks.

## Three clients have their own dedicated wikis

They maintain full three-tier, schema-conformant wikis and are the source of truth for their own domains — this hub does NOT restate any of it.

- [TankpitBot wiki index](../../clients/TankpitBot/wiki/index.md) — 6 hubs, 75 content pages (counted 2026-09-01): game mechanics, protocol, combat, JS client, architecture, codebase.
- [RustedWarfareBot wiki index](../../clients/RustedWarfareBot/wiki/index.md) — engine internals, the headless harness, game mechanics, bot architecture, multiplayer constraints. Claims about engine internals are pinned to a game build (`game_version` frontmatter) because the jar is obfuscated and class names change silently between releases.
- [NavProbe wiki index](../../clients/NavProbe/wiki/index.md) — 5 hubs, 37 content pages (counted 2026-09-01): determinism measurement, rendered observations, instrument design, simulator adapters, platform constraints. A measurement page carries a `measured_with` block, because a determinism verdict without its seed, step count, batch width and backend states nothing.

  NavProbe measures determinism in *someone else's simulator*; `platform_core.determinism_env` configures it in *this monorepo's own training stack*. The two are the same question asked from opposite sides, and the shared finding — that reduction order is set by a variable read once at library load — is written up at [Reduction order is an environment variable read once](../pages/determinism-env-read-once-at-library-load.md).

Cross-platform client facts that would belong *here* rather than in a client-specific wiki are integration surfaces into the monorepo — `platform_core.logging` config, `monorepo_guards` conformance, hypothetical future `platform_workers` job-enqueue paths. None of that has been written yet; add pages here as those surfaces materialise.

<!-- Add pages here as they're written. Format: [Title](../pages/<slug>.md) -- one-line description -->
