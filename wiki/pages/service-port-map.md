---
title: Service Port Map — every FastAPI service and its port
tags: [services, ports, infrastructure, routing]
related:
  - "[[monorepo-discipline]]"
source_paths:
  - README.md
  - docker-compose.yml
  - services/data-bank-api/README.md
  - services/data-bank-api/docker-compose.yml
  - services/covenant-radar-api/docs/configuration.md
source_git_blobs:
  "README.md": a5bbc13914ac4cd4428f57ad76b987502990bfe7
  "docker-compose.yml": 3e57145cb052407cedd701354dc58c44d56b22d6
  "services/data-bank-api/README.md": 49db4ef01f97328ac723f8e7015127091228e778
  "services/data-bank-api/docker-compose.yml": 6ee313dc9ea01b0457cb348dc262c961383ac778
  "services/covenant-radar-api/docs/configuration.md": 79c9a943f34890feffc9ee4df290d147433b0da5
fact_checked: "2026-07-31"
confidence: high
hubs: [services]
---

# Service Port Map

Every FastAPI service in the api monorepo binds to a fixed port. Ports are assigned once in the README's Services table (or, for services that layer their own compose on top of the root, in that service's own `docker-compose.yml`) and are stable across dev / staging / prod. New services claim the next free port in the 80xx range.[^1]

## Assigned ports (13 services)

| Port | Service | Purpose |
|---|---|---|
| 8000 | `turkic-api` | Turkic language detection + IPA transliteration (Kazakh, Kyrgyz, Uzbek, Turkish, Russian, +4 more) |
| 8001 | `data-bank-api` | Content-addressed file storage (SHA256 keys, atomic writes, HTTP Range) |
| 8002 | `qr-api` | QR code generation |
| 8003 | `transcript-api` | YouTube video transcription |
| 8004 | `handwriting-ai` | MNIST digit recognition with calibrated confidence |
| 8005 | `Model-Trainer` | Language model training (LoRA / QLoRA / Unsloth fine-tuning, GPT-2, Char-LSTM, CUDA) |
| 8006 | `music-wrapped-api` | Music listening analytics (Spotify, Apple, YouTube Music, Last.fm) |
| 8007 | `covenant-radar-api` | Multi-domain risk prediction, Kafka streaming (XGBoost, LightGBM, ClearGBM, LogReg, RF, MLP, LSTM) |
| 8008 | `grandma-api` | Multi-language audio-to-English translation (Whisper STT + language detection, GPT-4o-mini translation) |
| 8009 | `github-stats-api` | GitHub stats SVG card generation |
| 8010 | `opportunity-radar-api` | Hackathon and competition discovery |
| 8011 | `Art-Trainer` | Image generation model training (SD 1.5, SDXL, FLUX LoRAs via Kohya-ss) |
| — | `procart-api` | Procedural art rendering orchestration (port not yet fixed) |

## Port binding convention

Per data-bank's README: "The server port is configured via hypercorn's `--bind` flag (e.g., `--bind [::]:${PORT:-8000}`), not as an application environment variable."[^2] The port lives in the hypercorn / uvicorn invocation rather than in the application config, so:

- Local dev overrides port via `--bind` without touching config
- Docker maps container port → host port in `docker-compose.yml`
- No service reads its own port from env (`PORT` is only referenced by the launcher)

## Traefik and cross-service routing

Traefik (`traefik:v3`, the root compose's `gateway` service) fronts the fleet on entrypoint `:80` with `--providers.docker.exposedbydefault=false`, so a container joins the mesh only by opting in with labels.[^3] Each service exposes the standard health endpoints:[^4]

- `/healthz` — liveness (does the process respond?)
- `/readyz` — readiness (are dependencies — Redis, disk, downstream services — reachable?)

Routing is by **path prefix only**: all twelve routers in the repo declare a `PathPrefix` rule and there is no `Host(...)` rule anywhere, so Traefik does not route by hostname.[^5] Each router pairs a `stripprefix` middleware with `loadbalancer.server.port=8000`, which targets the *container* port — the 80xx numbers above are host-side mappings for local dev and `curl`, not inputs to Traefik's dispatch.[^6] Deployed cross-service URLs use the hosting platform's private DNS rather than a host port, e.g. `DATA_BANK_API_URL=http://data-bank-api.railway.internal:8080`.[^7]

## Adding a new service

1. Claim the next port in the 80xx range (8012 and 8013 are both free; 8012 was doc-extract-api's until it was removed).
2. Add the row to the README's Services table.
3. Add a `services/<name>/docker-compose.yml` overlay carrying the service block, its `ports: "<host>:8000"` mapping, and its five Traefik labels. The root `docker-compose.yml` is infrastructure only — `redis`, `kafka`, `postgres`, and `gateway` — and declares no application service and no `expose:` key at all.[^3]
4. Wire `/healthz` + `/readyz` per the [[platform-workers-rq-pattern]] readyz helpers if the service depends on Redis.
5. Follow [[monorepo-discipline]] — strict mypy, 100% coverage, `monorepo_guards`.

## Why fixed ports

Each overlay hardcodes its own host-port mapping and the README table fixes the assignment,[^1][^6] so the same service answers on the same port in every environment. Runbooks, log queries, and troubleshooting scripts can hardcode the tuple `(service, port)` without a lookup, and when Traefik or the docker network is down `curl localhost:8003/healthz` still reaches the container directly.[^4]

[^1]: [`README.md`](../../README.md) — Services table with the port assignments verbatim (rows for 8000-8011 + procart-api).
[^2]: [`services/data-bank-api/README.md`](../../services/data-bank-api/README.md):78 — verbatim: "> **Note:** The server port is configured via hypercorn's `--bind` flag (e.g., `--bind [::]:${PORT:-8000}`), not as an application environment variable." Verified 2026-07-31.
[^3]: [`docker-compose.yml`](../../docker-compose.yml):101-113 — the `gateway` service is `image: traefik:v3` with `--providers.docker.exposedbydefault=false` and `--entrypoints.web.address=:80`. The file's only top-level service keys are `redis`, `kafka`, `postgres`, `gateway`; a repo-wide grep of the file returns zero `expose:` occurrences. Verified 2026-07-31.
[^4]: [`README.md`](../../README.md):142 — verbatim: "- Health endpoints (`/healthz`, `/readyz`)".
[^5]: `services/Art-Trainer/docker-compose.yml:29`, `services/covenant-radar-api/docker-compose.yml:25` and `services/data-bank-api/docker-compose.yml:24` are representative of the pattern — each a single `traefik.http.routers.<name>.rule=PathPrefix(...)` label. Repo-wide sweep across every `*.yml` **re-run 2026-08-05: 12 `PathPrefix` rules, 0 `Host(` rules, same twelve prefixes as the 2026-07-31 sweep.** Returns twelve rules — `/art-trainer`, `/covenant`, `/data-bank`, `/github-stats`, `/grandma`, `/handwriting`, `/music`, `/opportunity`, `/qr`, `/trainer`, `/transcript`, `/turkic` — every one a `PathPrefix`, and no `Host(` rule in the repo. Corrects this page's pre-2026-07-31 claim that Traefik routed "by hostname / path prefix".
[^6]: [`services/data-bank-api/docker-compose.yml`](../../services/data-bank-api/docker-compose.yml):20-28 — `ports: - "8001:8000"` followed by the five labels: `traefik.enable=true`, `routers.databank.rule=PathPrefix(\`/data-bank\`)`, `routers.databank.middlewares=databank-strip`, `middlewares.databank-strip.stripprefix.prefixes=/data-bank`, `services.databank.loadbalancer.server.port=8000`, plus `traefik.docker.network=platform-network`.
[^7]: [`services/covenant-radar-api/docs/configuration.md`](../../services/covenant-radar-api/docs/configuration.md):118 — `DATA_BANK_API_URL=http://data-bank-api.railway.internal:8080`. Corrects this page's pre-2026-07-31 claim that deployed clients hit `service-name.internal`.
