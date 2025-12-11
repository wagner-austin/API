# clients/

Client applications that consume backend services from the monorepo. Each client is a standalone Poetry package with strict typing, 100% test coverage, and event-driven architecture.

## Clients

| Client | Description | Backend Services |
|--------|-------------|------------------|
| [DiscordBot](./DiscordBot) | Discord bot with slash commands for QR codes, transcripts, digit recognition, and model training | qr-api, transcript-api, handwriting-ai, model-trainer |

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
