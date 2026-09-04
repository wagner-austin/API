# API Platform Wiki

**Read this first.** 4 topic hubs, 29 content pages (counted 2026-09-04). Follow the hub link for your topic; each hub lists its pages with one-line descriptions. This wiki documents the *api monorepo* — services, clients, shared libs, and the infrastructure that ties them together.

**Known coverage shape, stated so nobody mistakes silence for absence:** 22 of these pages are ClearGBM. The services, clients and infrastructure hubs are thin relative to what the monorepo holds — 13 services and 4 clients against 2 service pages and 0 client pages. Client depth is deliberate (three clients keep their own full wikis). Infrastructure reads thinner than it is: cluster submission — the largest single body of it — lives in the `hpc3` sibling wiki below, registered 2026-09-02. **Service depth remains a real gap**: 13 services, 3 pages, and nothing else covers them.

## Hubs

[Services](hubs/services.md) -- the FastAPI ML/NLP/media services (data-bank, Model-Trainer, Art-Trainer, transcript, turkic, covenant-radar, grandma, handwriting-ai, qr, music-wrapped, github-stats, opportunity-radar, procart) (3 pages)
[Clients](hubs/clients.md) -- DiscordBot, TankpitBot and RustedWarfareBot, plus NavProbe, which is a simulator-determinism instrument rather than a user-facing client (1 page, cross-listed from infrastructure; TankpitBot, RustedWarfareBot and NavProbe each maintain their own dedicated wiki under `clients/<name>/wiki/`, which is why this hub stays thin by design)
[Libs](hubs/libs.md) -- shared platform_* libraries (core, workers, ml, discord, music, email, calendar, codebase, devpost, kaggle, stt, langid, translate) + domain libs (covenant_domain/ml/nn/persistence, cleargbm, cleargbm_rs, procart) + instrument_io + monorepo_guards (24 pages)
[Infrastructure](hubs/infrastructure.md) -- docker-compose, Traefik, Redis/RQ, PostgreSQL, monorepo build + test + lint conventions, run-comparability env (3 pages; cluster submission is NOT here — `tools/hpc3` keeps its own wiki, see below)

## Sibling wikis in this repo

Two trees under this repo maintain their own wikis rather than pages here.
Both are registered and searchable by slug through `wiki_search_query`.

- **`hpc3`** — [`tools/hpc3/wiki/`](../tools/hpc3/wiki/index.md), 4 hubs and
  21 pages: the design record and incident narrative behind every rule the
  `hpc3` package enforces (partitions and billing, submission rules, images
  and staging identity, job arrays, node-local scratch, triage, ledger
  closures, budgets). Registered 2026-09-02. If your question is "how do I
  run this on the cluster, and what will it cost", start there, not here.
- **`tankpitbot` / `navprobe` / `rustedwarfarebot`** — under
  `clients/<name>/wiki/`, which is why the Clients hub here stays thin.

## How this works

**Three tiers:** this index (read every session) → hub pages (read when topic matches) → content pages (read when you need the facts). A content page can be linked from multiple hubs.

**Adding new content:** create the content page in `pages/`, add an inclusion link from the relevant hub(s), bump page counts here. If the topic needs a new hub, create it in `hubs/` and add one line above.

**The rules:** see `SCHEMA.md` — atomicity, frontmatter, citations, hub-link discipline. Per-service READMEs are the entry point for a single service; this wiki extends them with cross-service context.
