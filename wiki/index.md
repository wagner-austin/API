# API Platform Wiki

**Read this first.** 4 topic hubs, 62 content pages (counted 2026-10-30). Follow the hub link for your topic; each hub lists its pages with one-line descriptions. This wiki documents the *api monorepo* — services, clients, shared libs, and the infrastructure that ties them together.

**Known coverage shape, stated so nobody mistakes silence for absence:** 22 of these pages are ClearGBM. The clients and infrastructure hubs are thin relative to what the monorepo holds. Client depth is deliberate (three clients keep their own full wikis). Infrastructure reads thinner than it is: cluster submission — the largest single body of it — lives in the `hpc3` sibling wiki below, registered 2026-09-02. **Service depth is narrower than it was but still a real gap** (counted 2026-10-05): covenant-radar has 5 architecture pages and Model-Trainer has 5 plus 10 research-result pages; Art-Trainer has 1; the other 10 services have nothing beyond the port map.

## Research entry points

- [Research registration directory](../docs/RESEARCH.md): projects, producing commands and links to their detailed records.
- [WRDS research](pages/wrds-equity-research-protocol.md): study/data contracts, evaluation, queue/admission, isolation, final scoring and baseline readiness.

## Hubs

[Services](hubs/services.md) -- the FastAPI ML/NLP/media services (data-bank, Model-Trainer, Art-Trainer, transcript, turkic, covenant-radar, grandma, handwriting-ai, qr, music-wrapped, github-stats, opportunity-radar, procart) (22 pages)
[Clients](hubs/clients.md) -- DiscordBot, TankpitBot and RustedWarfareBot, plus NavProbe, which is a simulator-determinism instrument rather than a user-facing client (1 page, cross-listed from infrastructure; TankpitBot, RustedWarfareBot and NavProbe each maintain their own dedicated wiki under `clients/<name>/wiki/`, which is why this hub stays thin by design)
[Libs](hubs/libs.md) -- shared platform_* libraries, domain libraries, evaluation contracts and the focused WRDS research pages (37 pages)
[Infrastructure](hubs/infrastructure.md) -- docker-compose, Traefik, Redis/RQ, PostgreSQL, monorepo build + test + lint conventions, run-comparability env, runner listener isolation, runner job trees and the on-host runner audit, the hub's tunnel route to loki (6 pages; cluster submission is NOT here — `tools/hpc3` keeps its own wiki, see below)

## Sibling wikis in this repo

The sibling trees below maintain their own wikis rather than pages here.
They are registered and searchable by slug through `wiki_search_query`.

- **`hpc3`** — [`tools/hpc3/wiki/`](../tools/hpc3/wiki/index.md), 4 hubs and
  31 pages: the design record and incident narrative behind every rule the
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
