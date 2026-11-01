---
title: Service Port Map — every FastAPI service and its port
tags: [services, ports, infrastructure, routing]
related:
  - "[[monorepo-discipline]]"
source_paths:
  - README.md
  - docker-compose.yml
  - services/covenant-radar-api/docker-compose.yml
  - services/covenant-radar-api/Dockerfile
  - services/grandma-api/docker-compose.yml
source_git_blobs:
  "README.md": 1fb94af064e86514127ee58817125fa836b0a509
  "docker-compose.yml": 538bb3e3c45ca1fd3010fecd535af8d9795e5c84
  "services/covenant-radar-api/docker-compose.yml": f18715070a89c105837afcff1bcef89b6c315896
  "services/covenant-radar-api/Dockerfile": 1609785a17251ca8a306cd640c83a117aa099883
  "services/grandma-api/docker-compose.yml": 698854143b29395878fa48d482405020bfbf0f30
fact_checked: "2026-11-01"
confidence: high
hubs: [services]
---

# Service Port Map

Every FastAPI service in the api monorepo binds to a fixed port. Ports are assigned once in the README's Services table (or, for services that layer their own compose on top of the root, in that service's own `docker-compose.yml`) and are stable across dev / staging / prod. New services claim the next free port in the 80xx range.[^1]

## Assigned ports (6 services)

| Port | Service | Purpose |
|---|---|---|
| 8003 | `transcript-api` | YouTube video transcription |
| 8005 | `Model-Trainer` | Language model training (LoRA / QLoRA / Unsloth fine-tuning, GPT-2, Char-LSTM, CUDA) |
| 8007 | `covenant-radar-api` | Multi-domain risk prediction, Kafka streaming (XGBoost, LightGBM, ClearGBM, LogReg, RF, MLP, LSTM) |
| 8008 | `grandma-api` | Multi-language audio-to-English translation (Whisper STT + language detection, GPT-4o-mini translation) |
| 8009 | `github-stats-api` | GitHub stats SVG card generation |
| 8010 | `opportunity-radar-api` | Hackathon and competition discovery |

8000, 8002, 8004, 8006 and 8011 belonged to turkic-api, qr-api, handwriting-ai, music-wrapped-api and Art-Trainer, which the operator had archived on 2026-11-01 with procart-api, which never had a port; 8001 belonged to data-bank-api, archived the same day once Model-Trainer and covenant-radar-api stopped storing through it; see [[api-package-use-inventory]].[^3]

## Port binding convention

The port lives in the hypercorn invocation rather than in the application config: covenant-radar-api's image starts with `exec hypercorn 'covenant_radar_api.api.main:create_app()' --bind [::]:${PORT:-8000}`.[^2] So:

- Local dev overrides port via `--bind` without touching config
- Docker maps container port → host port in `docker-compose.yml`
- No service reads its own port from env (`PORT` is only referenced by the launcher)

## Traefik and cross-service routing

Traefik (`traefik:v3`, the root compose's `gateway` service) fronts the fleet on entrypoint `:80` with `--providers.docker.exposedbydefault=false`, so a container joins the mesh only by opting in with labels.[^3] Each service exposes the standard health endpoints:[^4]

- `/healthz` — liveness (does the process respond?)
- `/readyz` — readiness (are dependencies — Redis, disk, downstream services — reachable?)

Routing is by **path prefix only**: all six routers in the repo declare a `PathPrefix` rule and there is no `Host(...)` rule anywhere, so Traefik does not route by hostname.[^5] Each router pairs a `stripprefix` middleware with `loadbalancer.server.port`, which targets the *container* port (8000 for five services, 8080 for grandma-api) — the 80xx numbers above are host-side mappings for local dev and `curl`, not inputs to Traefik's dispatch.[^6]

## Adding a new service

1. Claim the next port in the 80xx range (8012 and 8013 are both free; 8012 was doc-extract-api's until it was removed). The ports the 2026-11-01 archive freed are free too.
2. Add the row to the README's Services table.
3. Add a `services/<name>/docker-compose.yml` overlay carrying the service block, its `ports: "<host>:<container>"` mapping, and its five Traefik labels. The root `docker-compose.yml` is infrastructure only — `redis`, `kafka`, `postgres`, and `gateway` — and declares no application service and no `expose:` key at all.[^3]
4. Wire `/healthz` + `/readyz` per the [[platform-workers-rq-pattern]] readyz helpers if the service depends on Redis.
5. Follow [[monorepo-discipline]] — strict mypy, 100% coverage, `monorepo_guards`.

## Why fixed ports

Each overlay hardcodes its own host-port mapping and the README table fixes the assignment,[^1][^6] so the same service answers on the same port in every environment. Runbooks, log queries, and troubleshooting scripts can hardcode the tuple `(service, port)` without a lookup, and when Traefik or the docker network is down `curl localhost:8003/healthz` still reaches the container directly.[^4]

[^1]: [`README.md`](../../README.md) — Services table with the port assignments verbatim (rows for 8003, 8005 and 8007-8010).
[^2]: [`services/covenant-radar-api/Dockerfile`](../../services/covenant-radar-api/Dockerfile):83 — verbatim: `CMD ["sh", "-c", "exec hypercorn 'covenant_radar_api.api.main:create_app()' --bind [::]:${PORT:-8000}"]`. The Model-Trainer, github-stats-api and grandma-api images start the same way, grandma-api's defaulting to 8080. Replaces this page's earlier citation of data-bank-api's README, which left with that service on 2026-11-01. Verified 2026-11-01.
[^3]: [`docker-compose.yml`](../../docker-compose.yml):98-110 — the `gateway` service is `image: traefik:v3` with `--providers.docker.exposedbydefault=false` and `--entrypoints.web.address=:80`. The file's only top-level service keys are `redis`, `kafka`, `postgres`, `gateway`; a repo-wide grep of the file returns zero `expose:` occurrences. Its port-map comment, lines 14-25, names the six ports above and the six the archive freed. Verified 2026-11-01.
[^4]: [`README.md`](../../README.md):128 — verbatim: "- Health endpoints (`/healthz`, `/readyz`)".
[^5]: [`services/covenant-radar-api/docker-compose.yml`](../../services/covenant-radar-api/docker-compose.yml):25 and [`services/grandma-api/docker-compose.yml`](../../services/grandma-api/docker-compose.yml):17 are representative of the pattern — each a single `traefik.http.routers.<name>.rule=PathPrefix(...)` label. Repo-wide sweep across every `*.yml` **re-run 2026-11-01, after data-bank-api's archive: 6 `PathPrefix` rules, 0 `Host(` rules** — `/covenant`, `/github-stats`, `/grandma`, `/opportunity`, `/trainer`, `/transcript`. The 2026-08-05 sweep found twelve, the other six being `/art-trainer`, `/data-bank`, `/handwriting`, `/music`, `/qr` and `/turkic`. Corrects this page's pre-2026-07-31 claim that Traefik routed "by hostname / path prefix".
[^6]: [`services/covenant-radar-api/docker-compose.yml`](../../services/covenant-radar-api/docker-compose.yml):21-29 — `ports: - "8007:8000"` followed by the labels `traefik.enable=true`, `routers.covenant.rule=PathPrefix(\`/covenant\`)`, `routers.covenant.middlewares=covenant-strip`, `middlewares.covenant-strip.stripprefix.prefixes=/covenant`, `services.covenant.loadbalancer.server.port=8000` and `traefik.docker.network=platform-network`. [`services/grandma-api/docker-compose.yml`](../../services/grandma-api/docker-compose.yml) maps `"8008:8080"` (line 8) and points its load balancer at 8080 (line 20). This page used to cite deployed cross-service URLs on the hosting platform's private DNS; the only one, covenant-radar-api's `DATA_BANK_API_URL`, left with data-bank-api on 2026-11-01, and no service now calls another.
