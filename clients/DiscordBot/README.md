# Discord Club Bot

Modular Python Discord bot with slash commands for QR code generation, YouTube transcripts, and handwritten digit recognition. Built with Poetry and discord.py (app commands), featuring event-driven notifications, Redis-backed job queues, and strict type safety.

## Features

- **Slash Commands**: QR codes, YouTube transcripts, digit recognition, model training
- **Event-Driven Notifications**: Redis pub/sub for real-time DM updates
- **Background Jobs**: RQ job queue with progress tracking
- **Rate Limiting**: Per-user rate limiting with configurable windows
- **Type Safety**: mypy strict mode, zero `Any` types
- **100% Test Coverage**: Statements and branches

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry 1.8+
- A Discord Application with Bot token
- Developer Portal > Bot > Privileged Gateway Intents: enable 'Message Content Intent'

### Installation

```bash
cd clients/DiscordBot
poetry install --with dev
```

### Configuration

```bash
# Copy environment file
cp .env.example .env

# Set required values (at minimum):
# - DISCORD_TOKEN
# - DISCORD_APPLICATION_ID (for invite URL)
```

### First Run

```bash
# One-time global sync (first run only)
# Set COMMANDS_SYNC_ON_START=true in .env, then:
poetry run python -m clubbot.main

# After "Performed global command sync", set COMMANDS_SYNC_ON_START=false
```

### Invite Bot to Server

```bash
# Generate invite URL
poetry run python scripts/invite.py
```

Or via Developer Portal:
- OAuth2 > URL Generator
- Scopes: `bot`, `applications.commands`
- Permissions: View Channels, Send Messages, Attach Files, Embed Links, Read Message History, Use Application Commands

---

## Commands Reference

### `/qrcode url:<URL>`

Generate a QR code PNG from a URL.

| Parameter | Type | Description |
|-----------|------|-------------|
| `url` | string | URL to encode (auto-normalizes to https://) |

**Features:**
- Friendly URL handling: `google.com`, `www.example.org/path`, IPv4/IPv6, `localhost`
- Brandable defaults via environment variables
- Response includes clickable hyperlink to destination

### `/transcript url:<YouTube URL>`

Fetch and clean a YouTube video transcript.

| Parameter | Type | Description |
|-----------|------|-------------|
| `url` | string | YouTube video URL |

**Features:**
- Uses transcript-api service for processing
- Returns transcript as downloadable text file
- Rate limited per user

### `/read image:<attachment>`

Recognize a handwritten digit from an image.

| Parameter | Type | Description |
|-----------|------|-------------|
| `image` | attachment | PNG or JPEG image file |

**Features:**
- Uses handwriting-ai service for inference
- Returns prediction with confidence score
- Top-3 probabilities displayed

### `/train`

Queue a background training job for the digits model.

**Features:**
- Enqueues job to RQ queue (processed by handwriting-ai)
- User receives DM updates with training progress
- Requires Redis for job queue

### `/train_model`

Start a model training run via Model-Trainer service.

| Parameter | Type | Description |
|-----------|------|-------------|
| `model_family` | string | Model family (e.g., gpt2) |
| `model_size` | string | Model size label (e.g., small) |
| `max_seq_len` | int | Max sequence length |
| `num_epochs` | int | Number of epochs |
| `batch_size` | int | Batch size |
| `learning_rate` | float | Learning rate |
| `corpus_path` | string | Path to corpus in API container |
| `tokenizer_id` | string | Tokenizer artifact ID |

**Features:**
- All parameters required (no defaults)
- User receives DM updates with progress
- Requires Model-Trainer API to be configured

### `/invite`

Generate an OAuth2 invite URL for the bot.

---

## Configuration

### Environment Variables

#### Discord (Required)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DISCORD_TOKEN` | string | **Required** | Bot token from Developer Portal |
| `DISCORD_APPLICATION_ID` | string | - | For invite URL generation |
| `DISCORD_GUILD_ID` | string | - | Guild-specific command sync |
| `DISCORD_GUILD_IDS` | string | - | Multiple guilds (comma-separated) |
| `COMMANDS_SYNC_GLOBAL` | bool | `false` | Use global command sync |
| `LOG_LEVEL` | string | `INFO` | Logging level |

#### Service URLs

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `API_GATEWAY_URL` | string | - | Gateway URL (routes all services) |
| `QR_API_URL` | string | - | QR API service URL |
| `TRANSCRIPT_API_URL` | string | - | Transcript API service URL |
| `HANDWRITING_API_URL` | string | - | Handwriting AI service URL |
| `MODEL_TRAINER_API_URL` | string | - | Model Trainer API URL |
| `REDIS_URL` | string | - | Redis connection URL |

#### Handwriting Service

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `HANDWRITING_API_KEY` | string | - | API key for handwriting-ai |
| `HANDWRITING_API_TIMEOUT_SECONDS` | int | `5` | Request timeout (seconds) |
| `HANDWRITING_API_MAX_RETRIES` | int | `1` | Max request retries |

#### Model Trainer Service

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MODEL_TRAINER_API_KEY` | string | - | API key for model-trainer |
| `MODEL_TRAINER_API_TIMEOUT_SECONDS` | int | `10` | Request timeout (seconds) |
| `MODEL_TRAINER_API_MAX_RETRIES` | int | `1` | Max request retries |

#### QR Code Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `QR_DEFAULT_ERROR_CORRECTION` | string | `M` | L, M, Q, or H |
| `QR_DEFAULT_BOX_SIZE` | int | `10` | Pixels per module |
| `QR_DEFAULT_BORDER` | int | `1` | Quiet zone modules |
| `QR_DEFAULT_FILL_COLOR` | string | `#000000` | Foreground color |
| `QR_DEFAULT_BACK_COLOR` | string | `#FFFFFF` | Background color |
| `QRCODE_RATE_LIMIT` | int | `1` | Requests per window |
| `QRCODE_RATE_WINDOW_SECONDS` | int | `1` | Rate limit window |
| `QR_PUBLIC_RESPONSES` | bool | `true` | Show responses publicly |

#### Transcript Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TRANSCRIPT_PROVIDER` | string | `api` | Provider type (must be `api`) |
| `TRANSCRIPT_RATE_LIMIT` | int | `2` | Requests per window |
| `TRANSCRIPT_RATE_WINDOW_SECONDS` | int | `60` | Rate limit window |
| `TRANSCRIPT_PUBLIC_RESPONSES` | bool | `false` | Show responses publicly |
| `TRANSCRIPT_MAX_ATTACHMENT_MB` | int | `25` | Max attachment size (MB) |
| `TRANSCRIPT_PREFERRED_LANGS` | string | `en,en-US,en-GB` | Preferred caption languages |
| `TRANSCRIPT_STT_API_TIMEOUT_SECONDS` | int | `900` | API request timeout |

#### Digits Settings

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DIGITS_RATE_LIMIT` | int | `2` | Requests per window |
| `DIGITS_RATE_WINDOW_SECONDS` | int | `60` | Rate limit window |
| `DIGITS_PUBLIC_RESPONSES` | bool | `false` | Show responses publicly |
| `DIGITS_MAX_IMAGE_MB` | int | `2` | Max image size |

#### Model Trainer Rate Limits

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MODEL_TRAINER_RATE_LIMIT` | int | `1` | Requests per window |
| `MODEL_TRAINER_RATE_WINDOW_SECONDS` | int | `10` | Rate limit window |

### Example .env

```bash
DISCORD_TOKEN=your-bot-token
DISCORD_APPLICATION_ID=123456789012345678
REDIS_URL=redis://localhost:6379/0
API_GATEWAY_URL=http://gateway:80
LOG_LEVEL=INFO
```

---

## Architecture

### Component Overview

```
┌─────────────────┐     ┌─────────────────┐
│  Discord API    │◄────│   Bot Client    │
│                 │     │  (discord.py)   │
└─────────────────┘     └────────┬────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Cogs Layer    │     │   Services      │     │  Event Notifiers│
│                 │     │                 │     │                 │
│  - qr.py        │     │  - QR Client    │     │  - Digits       │
│  - transcript   │     │  - Transcript   │     │  - Trainer      │
│  - digits.py    │     │  - HandAI       │     │                 │
│  - trainer.py   │     │  - Trainer      │     │   (Redis PubSub)│
│  - invite.py    │     │                 │     │                 │
└─────────────────┘     └────────┬────────┘     └────────┬────────┘
                                 │                       │
                                 ▼                       ▼
                        ┌─────────────────┐     ┌─────────────────┐
                        │  Backend APIs   │     │     Redis       │
                        │                 │     │                 │
                        │  - qr-api       │     │  digits:events  │
                        │  - transcript   │     │  trainer:events │
                        │  - handwriting  │     │                 │
                        │  - model-trainer│     │                 │
                        └─────────────────┘     └─────────────────┘
```

### Project Structure

```
src/clubbot/
├── __init__.py             # Package init
├── main.py                 # Bot entry point
├── config.py               # Re-exports config from platform_core
├── container.py            # Service container (DI composition)
├── orchestrator.py         # Bot lifecycle management
├── _test_hooks.py          # Test hooks for DI
│
├── cogs/
│   ├── base.py             # Shared BaseCog (logging, error handling)
│   ├── qr.py               # /qrcode command
│   ├── transcript.py       # /transcript command
│   ├── digits.py           # /read and /train commands
│   ├── trainer.py          # /train_model command
│   ├── invite.py           # /invite command
│   └── example.py          # Example cog template (not auto-loaded)
│
├── services/
│   ├── registry.py         # Service registry
│   ├── enqueue.py          # Shared enqueueing utilities
│   ├── qr/
│   │   ├── __init__.py
│   │   └── client.py       # QR code generation service
│   ├── transcript/
│   │   ├── __init__.py
│   │   ├── client.py       # Transcript service orchestrator
│   │   └── api_client.py   # HTTP client to transcript-api
│   ├── digits/
│   │   └── app.py          # Digit recognition service
│   ├── handai/
│   │   └── client.py       # HTTP client to handwriting-ai
│   └── jobs/
│       ├── digits_enqueuer.py   # RQ job enqueuer
│       ├── digits_notifier.py   # Redis pub/sub for digits
│       ├── trainer_notifier.py  # Redis pub/sub for trainer
│       └── turkic_notifier.py   # Redis pub/sub for turkic
│
└── utils/
    └── youtube.py          # YouTube URL validation
```

---

## Background Jobs

### Job Queue Architecture

```
┌─────────────────┐
│   Discord Bot   │
│   (Cogs)        │
└────────┬────────┘
         │ enqueue
         ▼
┌─────────────────┐     ┌─────────────────┐
│     Redis       │◄────│  RQ Workers     │
│   Job Queue     │     │                 │
│                 │     │  - Digits       │
│                 │     │  - Trainer      │
└────────┬────────┘     └─────────────────┘
         │ publish
         ▼
┌─────────────────┐     ┌─────────────────┐
│  Redis PubSub   │────►│  Event Notifiers│
│                 │     │                 │
│  digits:events  │     │  DM Progress    │
│  trainer:events │     │  Updates        │
└─────────────────┘     └─────────────────┘
```

### Event Types

**Digits Training Events (`digits:events`):**
- `digits.metrics.config.v1` - Training configuration
- `digits.metrics.batch.v1` - Batch-level metrics
- `digits.metrics.epoch.v1` - Epoch-level metrics
- `digits.metrics.best.v1` - New best model checkpoint
- `digits.metrics.completed.v1` - Training completed
- `digits.job.failed.v1` - Training failed

**Model Trainer Events (`trainer:events`):**
- `trainer.job.started.v1`
- `trainer.job.progress.v1`
- `trainer.job.completed.v1`
- `trainer.job.failed.v1`

**Turkic Events (`turkic:events`):**
- `turkic.job.started.v1`
- `turkic.job.progress.v1`
- `turkic.job.completed.v1`
- `turkic.job.failed.v1`

### Notifier Pattern

Event notifiers are thin wrappers extending `BotEventSubscriber`. Event decoding and handling logic lives in `platform_discord` domain kits:

```python
from platform_discord.bot_subscriber import BotEventSubscriber
from platform_discord.domain import (
    DomainEventV1,
    DomainRuntime,
    decode_domain_event_safe,
    handle_domain_event,
    new_runtime,
)
from platform_discord.protocols import BotProto

class DomainEventSubscriber(BotEventSubscriber[DomainEventV1]):
    __slots__ = ("_runtime",)

    def __init__(self, *, bot: BotProto, redis_url: str) -> None:
        super().__init__(
            bot,
            redis_url=redis_url,
            events_channel="domain:events",
            task_name="domain-subscriber",
            decode=decode_domain_event_safe,  # from platform_discord
        )
        self._runtime: DomainRuntime = new_runtime()

    async def _handle_event(self, ev: DomainEventV1) -> None:
        action = handle_domain_event(self._runtime, ev)  # from platform_discord
        if action is not None:
            await self._maybe_notify(action)
```

---

## Development

### Commands

```bash
make lint   # Run guards + ruff + mypy (also installs deps)
make test   # Run pytest with coverage (also installs deps)
make check  # Run lint then test
```

### Quality Gates

All code must pass:

1. **Guard Scripts**: No `Any`, no `cast`, no `type: ignore`
2. **Ruff**: Linting and formatting
3. **Mypy**: Strict type checking
4. **Pytest**: 100% statement and branch coverage

### Running Tests

```bash
# Run all tests
make test

# Run specific test file
poetry run pytest tests/clubbot/cogs/test_qr_cog.py -v

# Run with coverage report
poetry run pytest --cov-report=html
```

### Building New Cogs

1. Create a cog in `src/clubbot/cogs/<name>.py`
2. Inherit from `BaseCog` for consistent logging and error handling
3. Use `@app_commands.command` with context/install decorators
4. In command handlers:
   - Call `await self._safe_defer(interaction, ephemeral=...)` first
   - Set up request-scoped logging via `self.new_request_id()` and `self.request_logger(req_id)`
   - For user errors: `await self.handle_user_error(interaction, log, message)`
   - For system errors: `await self.handle_exception(interaction, log, exc)`
5. Use `RateLimiter` from `platform_discord.rate_limiter` for per-user rate limiting
6. Provide an `async def setup(bot)` function for standalone cog loading

---

## Deployment

### Docker Compose

```yaml
services:
  bot:
    build:
      context: ../..
      dockerfile: clients/DiscordBot/Dockerfile
      target: bot
    env_file:
      - .env
    environment:
      - REDIS_URL=redis://platform-redis:6379/0
    networks:
      - platform-network

  worker:
    build:
      context: ../..
      dockerfile: clients/DiscordBot/Dockerfile
      target: worker
    env_file:
      - .env
    environment:
      - REDIS_URL=redis://platform-redis:6379/0
    networks:
      - platform-network

networks:
  platform-network:
    external: true
```

### Railway Deployment

1. **Create services**: Bot and Worker from same Dockerfile
2. **Add Redis addon** or connect to existing Redis
3. **Set environment variables**:
   ```
   DISCORD_TOKEN=${{Secrets.DISCORD_TOKEN}}
   REDIS_URL=${{Redis.REDIS_URL}}
   API_GATEWAY_URL=https://your-gateway.railway.app
   ```

---

## Dependencies

### Runtime

| Package | Purpose |
|---------|---------|
| `discord.py` | Discord API client |
| `Pillow` | Image processing |
| `httpx` | HTTP client |
| `redis` | Redis client |
| `rq` | Redis Queue |
| `platform-core` | Logging, errors, config |
| `platform-discord` | Discord protocols, rate limiting |
| `platform-workers` | RQ worker harness |

### Development

| Package | Purpose |
|---------|---------|
| `pytest` | Test runner |
| `pytest-cov` | Coverage reporting |
| `pytest-asyncio` | Async test support |
| `pytest-xdist` | Parallel tests |
| `mypy` | Type checking |
| `ruff` | Linting/formatting |

---

## QR Error Correction Levels

| Level | Recovery | Use Case |
|-------|----------|----------|
| `L` | ~7% | Smallest code, most data-efficient |
| `M` | ~15% | Good default for most cases |
| `Q` | ~25% | More robust to damage/overlays |
| `H` | ~30% | Most robust (logos/occlusion) |

**Note:** Default border is 1 module for compact codes. QR spec recommends 4 for maximum scanner compatibility.

---

## Notes

- **Global Commands**: Sync globally via `bot.tree.sync()` and allow DMs
- **Propagation**: Initial global registration may take up to ~1 hour
- **Validation**: Happens before rate limiting for clear error messages
- **Defer**: Commands defer immediately (ACK first) to avoid Discord's 3s timeout

---

## License

Apache-2.0
