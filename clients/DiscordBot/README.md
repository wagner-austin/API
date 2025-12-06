# Discord Club Bot

Modular Python Discord bot with slash commands for QR code generation, YouTube transcripts, and handwritten digit recognition. Built with Poetry and discord.py (app commands).

## Features

### `/qrcode url:<URL>`
Generate a QR code PNG from a URL.
- Friendly URL handling: `google.com`, `www.example.org/path`, IPv4/IPv6, and `localhost` are accepted and normalized to `https://...`
- Brandable defaults via env vars (ECC, box size, border, colors)
- Input validation with clear error messages
- Response includes a clickable hyperlink to the destination URL
- Public responses by default (configurable)

### `/transcript url:<YouTube URL>`
Fetch and clean a YouTube video transcript.
- Uses the transcript-api service for processing
- Returns transcript as a downloadable text file
- Rate limited per user

### `/read image:<attachment>`
Recognize a handwritten digit from an image.
- Accepts PNG or JPEG images
- Uses the handwriting-ai service for inference
- Returns prediction with confidence score and top-3 probabilities

### `/train`
Queue a background training job for the digits model.
- Enqueues job to RQ queue processed by handwriting-ai
- User receives DM updates with training progress
- Requires Redis for job queue

### `/train_model`
Start a model training run via the Model-Trainer service.
- Configurable model family, size, epochs, batch size, learning rate
- Requires Model-Trainer API to be configured
- User receives DM updates with training progress

### `/invite`
Generate an OAuth2 invite URL for the bot.

## Prerequisites

- Python 3.11+
- Poetry
- A Discord Application with a Bot token
- Developer Portal > Bot > Privileged Gateway Intents: enable 'Message Content Intent'

## Setup

1. Copy `.env.example` to `.env` and fill in values (at least `DISCORD_TOKEN`).
2. Install deps: `poetry install`
3. One-time global sync (first run only): set `COMMANDS_SYNC_ON_START=true` in `.env`, then run `make run`. After you see "Performed global command sync", set it back to `false`.
4. Invite the bot to a server:
   - Developer Portal > OAuth2 > URL Generator
   - Scopes: `bot`, `applications.commands`
   - Permissions: View Channels, Send Messages, Attach Files, Embed Links, Read Message History, Use Application Commands
   - Or run `poetry run python scripts/invite.py`
5. Use commands in a server or DM. Global command propagation can take up to ~1 hour on first registration.

## Project Layout

```
src/clubbot/
  main.py                 # Bot entry point
  config.py               # Re-exports config from platform_core
  container.py            # Service container (DI composition)
  orchestrator.py         # Bot lifecycle management

  cogs/
    base.py               # Shared BaseCog (request-scoped logging, error handling)
    qr.py                 # /qrcode command
    transcript.py         # /transcript command
    digits.py             # /read and /train commands
    trainer.py            # /train_model command
    invite.py             # /invite command
    example.py            # Example cog template (not auto-loaded)

  services/
    qr/client.py          # QR code generation service
    transcript/
      client.py           # Transcript service orchestrator
      api_client.py       # HTTP client to transcript-api
    digits/app.py         # Digit recognition service
    handai/client.py      # HTTP client to handwriting-ai service
    jobs/
      digits_enqueuer.py  # RQ job enqueuer for digits training
      digits_notifier.py  # Redis pub/sub subscriber for digits events
      trainer_notifier.py # Redis pub/sub subscriber for trainer events
    enqueue.py            # Shared enqueueing utilities
    registry.py           # Service registry

  utils/
    youtube.py            # YouTube URL validation
```

## Dependencies

This bot uses shared packages from the monorepo:
- `platform_core` - Configuration, logging, errors, event schemas, API clients
- `platform_discord` - Discord protocols, rate limiting, embed helpers

## Environment Variables

### Required
- `DISCORD_TOKEN` - Bot token from Discord Developer Portal

### QR Code (`/qrcode`)
- `QR_API_URL` - Base URL for qr-api service
- `QR_DEFAULT_ERROR_CORRECTION` - L, M, Q, H (default: M)
- `QR_DEFAULT_BOX_SIZE` - Pixels per module (default: 10)
- `QR_DEFAULT_BORDER` - Quiet zone modules (default: 1)
- `QR_DEFAULT_FILL_COLOR` - Foreground color (default: #000000)
- `QR_DEFAULT_BACK_COLOR` - Background color (default: #FFFFFF)
- `QRCODE_RATE_LIMIT` - Requests per window (default: 1)
- `QRCODE_RATE_WINDOW_SECONDS` - Rate limit window (default: 1)
- `QR_PUBLIC_RESPONSES` - Show responses publicly (default: true)

### Transcript (`/transcript`)
- `TRANSCRIPT_API_URL` - Base URL for transcript-api service
- `TRANSCRIPT_RATE_LIMIT` - Requests per window (default: 2)
- `TRANSCRIPT_RATE_WINDOW_SECONDS` - Rate limit window (default: 60)
- `TRANSCRIPT_PUBLIC_RESPONSES` - Show responses publicly (default: false)
- `TRANSCRIPT_MAX_ATTACHMENT_MB` - Max attachment size (default: 25)

### Digits (`/read`, `/train`)
- `HANDWRITING_API_URL` - Base URL for handwriting-ai service
- `HANDWRITING_API_KEY` - API key for handwriting-ai
- `DIGITS_RATE_LIMIT` - Requests per window
- `DIGITS_RATE_WINDOW_SECONDS` - Rate limit window
- `DIGITS_PUBLIC_RESPONSES` - Show responses publicly
- `DIGITS_MAX_IMAGE_MB` - Max image size

### Model Trainer (`/train_model`)
- `MODEL_TRAINER_API_URL` - Base URL for model-trainer service
- `MODEL_TRAINER_API_KEY` - API key for model-trainer
- `MODEL_TRAINER_API_TIMEOUT_SECONDS` - Request timeout
- `MODEL_TRAINER_RATE_LIMIT` - Requests per window
- `MODEL_TRAINER_RATE_WINDOW_SECONDS` - Rate limit window

### Redis (for background jobs)
- `REDIS_URL` - Redis connection URL (required for `/train` and `/train_model`)

### Discord
- `DISCORD_GUILD_ID` or `DISCORD_GUILD_IDS` - For guild-specific command sync (optional)
- `DISCORD_APPLICATION_ID` - For invite URL generation
- `COMMANDS_SYNC_ON_START` - Sync commands on startup (default: false)
- `COMMANDS_SYNC_GLOBAL` - Use global command sync (default: false)
- `LOG_LEVEL` - Logging level (default: INFO)

## Linting & Formatting

- Lint: `make lint` (ruff check)
- Auto-fix: `make lint-fix`
- Format: `make format`
- Type check: `make typecheck` (strict mypy on src/)
- Full check: `make check` (ruff --fix, format, mypy, pytest)
- Tests only: `make test`

## Building New Cogs

1. Create a cog in `src/clubbot/cogs/<name>.py`
2. Inherit from `BaseCog` for consistent logging and error handling
3. Use `@app_commands.command` with `@app_commands.allowed_contexts` and `@app_commands.allowed_installs`
4. In command handlers:
   - Call `await self._safe_defer(interaction, ephemeral=...)` first
   - Set up request-scoped logging via `self.new_request_id()` and `self.request_logger(req_id)`
   - For user errors, use `await self.handle_user_error(interaction, log, message)`
   - For system errors, use `await self.handle_exception(interaction, log, exc)`
5. Use `RateLimiter` from `platform_discord.rate_limiter` for per-user rate limiting
6. Provide an `async def setup(bot)` function for standalone cog loading

## Background Jobs

Background jobs use RQ (Redis Queue) for durable job processing:

- **Digits training** (`/train`): Jobs enqueued via `RQDigitsEnqueuer`, processed by handwriting-ai worker
- **Model training** (`/train_model`): Jobs submitted directly to Model-Trainer API

Event-driven progress updates:
- Bot subscribes to Redis pub/sub channels (`digits:events`, `trainer:events`)
- Users receive DM updates on job start, progress, completion, or failure
- See `services/jobs/digits_notifier.py` and `services/jobs/trainer_notifier.py`

## Notifications (Runner Pattern)

Use the shared TaskRunner to standardize long-running subscribers. Each notifier:
- Builds a `RedisEventSubscriber` via a `build()` function that returns `{ runnable, closable }`.
- Wires strict event runtime helpers from `platform_discord.<domain>.runtime`.
- Delegates lifecycle to `TaskRunner` for `start()`, `stop()`, and one-off `run_once()`.

Example skeleton:

```python
from platform_core.<domain>_events import DEFAULT_<DOMAIN>_EVENTS_CHANNEL, EventV1, try_decode_event
from platform_discord.subscriber import RedisEventSubscriber
from platform_discord.message_source import RedisPubSubSource
from platform_discord.task_runner import TaskRunner
from platform_discord.<domain>.runtime import new_runtime, on_started, on_progress, on_completed, on_failed

class <Domain>EventSubscriber:
    def __init__(self, *, redis_url: str, events_channel: str | None = None) -> None:
        self._redis_url = redis_url
        self.events_channel = events_channel or DEFAULT_<DOMAIN>_EVENTS_CHANNEL
        self._runtime = new_runtime()
        self._source = None
        self._subscriber = None

        def _build():
            source = RedisPubSubSource(self._redis_url)
            sub: RedisEventSubscriber[EventV1] = RedisEventSubscriber(
                channel=self.events_channel,
                source=source,
                decode=try_decode_event,
                handle=self._handle_event,
            )
            self._source = source
            self._subscriber = sub
            return {"runnable": sub, "closable": source}

        self._runner = TaskRunner(build=_build, name="<domain>-event-subscriber")

    def start(self) -> None:
        self._runner.start()

    async def stop(self) -> None:
        try:
            await self._runner.stop()
        finally:
            if self._source is not None:
                await self._source.close()
            self._source = None
            self._subscriber = None

    async def _handle_event(self, ev: EventV1) -> None:
        # Call appropriate on_* runtime helper, then DM if an embed is returned
        ...
```

See concrete implementations under `src/clubbot/services/jobs/*_notifier.py` and the domain kits under `libs/platform_discord`.

## Notes

- Global-only commands: we sync globally via `bot.tree.sync()` and allow DMs
- Propagation: initial global registration may take up to ~1 hour; subsequent edits are often faster
- Validation happens before rate limiting so users see clear input errors rather than generic cooldown messages
- Commands defer immediately (ACK first) to avoid Discord's 3s timeout

## QR Error Correction Levels

- `L` (Low): ~7% of codewords can be restored. Smallest QR code, most data-efficient; least robust.
- `M` (Medium): ~15% restored. Good default for most cases.
- `Q` (Quartile): ~25% restored. More robust to damage/overlays; larger code.
- `H` (High): ~30% restored. Most robust (e.g., logos/occlusion), largest code.

Note: Default border (quiet zone) is set to 1 module for compact codes. The QR spec recommends 4 for maximum scanner compatibility; increase via `QR_DEFAULT_BORDER` if needed.
