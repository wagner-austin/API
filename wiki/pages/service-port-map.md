---
title: Service Port Map — every FastAPI service and its port
tags: [services, ports, infrastructure, routing]
related:
  - "[[monorepo-discipline]]"
source_paths:
  - README.md
  - docker-compose.yml
  - services/doc-extract-api/docker-compose.yml
fact_checked: "2026-07-20"
confidence: high
hubs: [services]
---

# Service Port Map

Every FastAPI service in the api monorepo binds to a fixed port. Ports are assigned once in the README's Services table (or, for services that layer their own compose on top of the root, in that service's own `docker-compose.yml`) and are stable across dev / staging / prod. New services claim the next free port in the 80xx range.[^1]

## Assigned ports (14 services)

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
| 8012 | `doc-extract-api` | PDF text extraction (pdfplumber + docTR OCR); layered `docker-compose.yml` in the service dir, not the root compose[^2] |
| — | `procart-api` | Procedural art rendering orchestration (port not yet fixed) |

## Port binding convention

Per data-bank's README: "The server port is configured via hypercorn's `--bind` flag (e.g., `--bind [::]:${PORT:-8000}`), not as an application environment variable." Every service follows this — port lives in the hypercorn / uvicorn invocation, not in the application config. This means:

- Local dev overrides port via `--bind` without touching config
- Docker maps container port → host port in `docker-compose.yml`
- No service reads its own port from env (`PORT` is only referenced by the launcher)

## Traefik and cross-service routing

In deployed environments, Traefik fronts the service fleet. Each service exposes standard endpoints:

- `/healthz` — liveness (does the process respond?)
- `/readyz` — readiness (are dependencies — Redis, disk, downstream services — reachable?)

Traefik routes by hostname / path prefix, not by port directly. Ports matter for local dev and for the docker-compose network; deployed clients hit `service-name.internal` (or the Traefik-fronted URL), not `:80xx`.

## Adding a new service

1. Claim the next port in the 80xx range (currently 8013 is free).
2. Add the row to the README's Services table.
3. Add the service block to `docker-compose.yml` with `expose: [<port>]` and the standard Traefik labels — or, if the service needs its own compose overlay (e.g. GPU services with extra volumes, like doc-extract-api), add a `services/<name>/docker-compose.yml` that layers on top of the root compose.
4. Wire `/healthz` + `/readyz` per the [[platform-workers-rq-pattern]] readyz helpers if the service depends on Redis.
5. Follow [[monorepo-discipline]] — strict mypy, 100% coverage, `monorepo_guards`.

## Why fixed ports

Consistency across dev / staging / prod: the same service is always on the same port. Runbooks, log queries, and troubleshooting scripts can hardcode the tuple `(service, port)` without a lookup. When Traefik or the docker network fails, you can still `curl localhost:8003/healthz` and get a useful signal.

[^1]: [`README.md`](../../README.md) — Services table with the port assignments verbatim (rows for 8000-8012 + procart-api). doc-extract-api's 8012 is listed there and independently confirmed by [`services/doc-extract-api/docker-compose.yml`](../../services/doc-extract-api/docker-compose.yml), which is where it was authoritative before the root README row was added.
[^2]: [`services/doc-extract-api/docker-compose.yml`](../../services/doc-extract-api/docker-compose.yml) — `ports: "8012:8000"` maps host 8012 → container 8000. The compose file declares "Requires root docker-compose to be running first (provides platform-redis + network)."
