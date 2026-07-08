# API Platform Wiki

**Read this first.** 4 topic hubs, 3 content pages. Follow the hub link for your topic; each hub lists its pages with one-line descriptions. This wiki documents the *api monorepo* — services, clients, shared libs, and the infrastructure that ties them together.

## Hubs

[Services](hubs/services.md) -- the FastAPI ML/NLP/media services (data-bank, Model-Trainer, Art-Trainer, transcript, turkic, covenant-radar, grandma, handwriting-ai, qr, music-wrapped, github-stats, opportunity-radar, procart) (1 page)
[Clients](hubs/clients.md) -- DiscordBot and TankpitBot — the user-facing clients that consume the service surface (0 pages)
[Libs](hubs/libs.md) -- shared platform_* libraries (core, workers, ml, discord, music, email, calendar, codebase, kaggle, stt, langid, translate) + domain libs (covenant_*, cleargbm, procart) + instrument_io + monorepo_guards (1 page)
[Infrastructure](hubs/infrastructure.md) -- docker-compose, Traefik, Redis/RQ, PostgreSQL, monorepo build + test + lint conventions (1 page)

## How this works

**Three tiers:** this index (read every session) → hub pages (read when topic matches) → content pages (read when you need the facts). A content page can be linked from multiple hubs.

**Adding new content:** create the content page in `pages/`, add an inclusion link from the relevant hub(s), bump page counts here. If the topic needs a new hub, create it in `hubs/` and add one line above.

**The rules:** see `SCHEMA.md` — atomicity, frontmatter, citations, hub-link discipline. Per-service READMEs are the entry point for a single service; this wiki extends them with cross-service context.
