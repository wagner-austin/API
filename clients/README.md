# clients/

Standalone applications, each a Poetry package with strict typing and 100% test
coverage. They fall into two groups: **service clients** that consume backend
services from the monorepo, and **system-identification instruments** that infer the
rules of undocumented systems and prove the inferred model matches reality.

## Clients

| Client | Description | Backend Services |
|--------|-------------|------------------|
| [DiscordBot](./DiscordBot) | Discord bot with slash commands for QR codes, transcripts, digit recognition, and model training | qr-api, transcript-api, handwriting-ai, model-trainer |
| [TankpitBot](./TankpitBot) | System identification against an undocumented network protocol — CDP capture, XOR codec, live probes, provenance-carrying beliefs, archive-priced validators, server twin | (standalone) |
| [RustedWarfareBot](./RustedWarfareBot) | System identification against an obfuscated binary — JVM bytecode injection, build-pinned engine claims, doctrine-controlled experiments, seeded reproducible matches | (standalone) |
| [NavProbe](./NavProbe) | Reproducibility instrument for simulated navigation — seeded rollouts, canonical digests, determinism verdicts across MJX and MuJoCo-Warp | (standalone) |

### The instruments

TankpitBot and RustedWarfareBot solve the same problem from opposite ends: one
reverses a network protocol from the outside, the other instruments a runtime from
the inside. Both target systems that publish no spec, cannot be verified directly,
and drift without notice — so both invest more in *learning and checking* the system
than in acting on it: controlled probes, beliefs carrying confidence and provenance,
conformance validation against an archive, and claims pinned to the build or capture
they were measured on.

## Design Principles

All clients follow these conventions:

- **Strict typing**: No `Any`, `cast`, `type: ignore`, or `.pyi` stubs
- **100% test coverage**: Statement and branch coverage enforced
- **Event-driven**: Redis pub/sub for real-time updates
- **Rate limiting**: Per-user rate limiting with configurable windows
- **Service abstraction**: HTTP clients wrap backend APIs with typed responses

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Clients                                  │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │                     DiscordBot                           │    │
│  │                                                          │    │
│  │  Cogs ─────► Services ─────► HTTP Clients ───────┐      │    │
│  │    │                                              │      │    │
│  │    └─► Event Notifiers ◄─── Redis PubSub ◄──┐    │      │    │
│  └───────────────────────────────────────────────│──│───────┘    │
└──────────────────────────────────────────────────│──│────────────┘
                                                   │  │
┌──────────────────────────────────────────────────│──│────────────┐
│                     Backend Services             │  │            │
│                                                  ▼  │            │
│  ┌────────────┐ ┌────────────┐ ┌──────────────┐    │            │
│  │  qr-api    │ │transcript  │ │ handwriting  │    │            │
│  │            │ │  -api      │ │    -ai       │────┘            │
│  └────────────┘ └────────────┘ └──────────────┘                 │
│                                                                  │
│  ┌────────────┐                                                  │
│  │  model-    │ Publishes events to Redis                       │
│  │  trainer   │ for progress updates                            │
│  └────────────┘                                                  │
└──────────────────────────────────────────────────────────────────┘
```

## Development

Each client has a Makefile with standard targets:

```bash
make lint   # Run guard checks, ruff, mypy
make test   # Run pytest with coverage
make check  # Run both lint and test
```

## Shared Libraries

Clients depend on shared libraries from `libs/`:

| Library | Purpose |
|---------|---------|
| `monorepo-guards` | Code quality enforcement (no Any, cast, type: ignore) |
| `platform-core` | Configuration, logging, errors, API clients |
| `platform-discord` | Discord protocols, rate limiting, embed helpers |
| `platform-workers` | Redis/RQ background job processing |

## Adding a New Client

See **[docs/adding-clients.md](./docs/adding-clients.md)** for a comprehensive guide covering:

- Directory structure and file layout
- pyproject.toml configuration (strict mypy, ruff, coverage)
- Test hooks pattern for dependency injection
- Service container and configuration
- Test fixtures and settings factory
- Guard scripts for code quality
- Makefile targets
- Docker and deployment configuration
- Connecting to backend services (HTTP, Redis pub/sub, RQ jobs)

## Deployment

Clients are deployed via Docker Compose or Railway:

### Docker Compose

```bash
# From client directory
docker compose up -d
```

Requires the root `docker-compose.yml` to be running first for shared infrastructure (Redis, network).

### Railway

1. Create new service from client directory
2. Set environment variables from `.env.example`
3. Connect to shared Redis addon

## Requirements

- Python 3.11+
- Poetry for dependency management
- Docker Desktop for containerized deployment
- Redis for background jobs and event subscriptions
