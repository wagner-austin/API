# Transcript API

Strictly typed transcript service providing YouTube caption extraction and OpenAI Whisper speech-to-text for Discord bot and other clients. Features protocol-based adapters, audio chunking for long videos, and parallel processing.

## Features

- **YouTube Captions**: Native transcript extraction with language preferences and auto-translation
- **Speech-to-Text**: OpenAI Whisper integration with verbose timing output
- **Audio Chunking**: Automatic splitting at silence points for large files
- **Parallel Processing**: Concurrent chunk transcription with retry logic
- **Type Safety**: mypy strict mode, zero `Any` types, Protocol-based adapters
- **100% Test Coverage**: Statements and branches

## Quick Start

### Prerequisites

- Python 3.11+
- Poetry 1.8+
- OpenAI API key
- ffmpeg/ffprobe (required for STT chunking)

### Installation

```bash
cd services/transcript-api
poetry install --with dev
```

### Environment Setup

```bash
# Required
export OPENAI_API_KEY=sk-...

# Optional
export TRANSCRIPT_MAX_VIDEO_SECONDS=3600  # Max video duration (0 = unlimited)
export TRANSCRIPT_MAX_FILE_MB=100         # Max file size (0 = unlimited)
export TRANSCRIPT_ENABLE_CHUNKING=1       # Enable audio chunking
```

### Run the Service

```bash
# Development
poetry run hypercorn transcript_api.asgi:app --bind 0.0.0.0:8000 --reload

# Production
poetry run hypercorn transcript_api.asgi:app --bind [::]:${PORT:-8000}
```

### Verify

```bash
curl http://localhost:8000/healthz
# {"status": "ok"}
```

## API Reference

For complete API documentation, see [docs/api.md](./docs/api.md).

### Quick Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/healthz` | GET | Liveness probe |
| `/readyz` | GET | Readiness probe (checks Redis) |
| `/v1/captions` | POST | Extract YouTube native captions |
| `/v1/stt` | POST | Transcribe video audio via Whisper (sync) |
| `/v1/stt/jobs` | POST | Enqueue async STT job |
| `/v1/stt/jobs/{job_id}` | GET | Get async STT job status |

---

## Configuration

### Environment Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `OPENAI_API_KEY` | string | **Required** | OpenAI API key for Whisper |
| `PORT` | int | `8000` | Server port |
| `TRANSCRIPT_MAX_VIDEO_SECONDS` | int | `0` | Max video duration (0 = unlimited) |
| `TRANSCRIPT_MAX_FILE_MB` | int | `0` | Max audio file size (0 = unlimited) |
| `TRANSCRIPT_ENABLE_CHUNKING` | bool | `false` | Enable audio chunking for large files |
| `TRANSCRIPT_CHUNK_THRESHOLD_MB` | float | `20.0` | File size threshold to trigger chunking |
| `TRANSCRIPT_TARGET_CHUNK_MB` | float | `20.0` | Target size for each audio chunk |
| `TRANSCRIPT_MAX_CHUNK_DURATION_SECONDS` | float | `600.0` | Max duration per chunk (10 min) |
| `TRANSCRIPT_MAX_CONCURRENT_CHUNKS` | int | `3` | Parallel chunk transcription workers |
| `TRANSCRIPT_SILENCE_THRESHOLD_DB` | float | `-40.0` | Silence detection threshold (dB) |
| `TRANSCRIPT_SILENCE_DURATION_SECONDS` | float | `0.5` | Min silence duration for split point |
| `TRANSCRIPT_STT_RTF` | float | `0.5` | Real-time factor for STT timeout estimation |
| `TRANSCRIPT_DL_MIB_PER_SEC` | float | `4.0` | Estimated download speed (MiB/s) for timeout |
| `TRANSCRIPT_PREFERRED_LANGS` | string | - | Comma-separated default languages |
| `REDIS_URL` | string | **Required** | Redis URL for /readyz health check and events |

### Example Configurations

**Development (with chunking):**
```bash
export OPENAI_API_KEY=sk-...
export REDIS_URL=redis://localhost:6379/0
export TRANSCRIPT_ENABLE_CHUNKING=1
export TRANSCRIPT_CHUNK_THRESHOLD_MB=10
export TRANSCRIPT_MAX_CONCURRENT_CHUNKS=5
```

**Production (strict limits):**
```bash
export OPENAI_API_KEY=sk-...
export REDIS_URL=redis://redis:6379/0
export TRANSCRIPT_MAX_VIDEO_SECONDS=3600
export TRANSCRIPT_MAX_FILE_MB=100
export TRANSCRIPT_ENABLE_CHUNKING=1
export TRANSCRIPT_PREFERRED_LANGS=en,es,fr
```

---

## Error Handling

All errors return JSON with consistent format:

```json
{
  "code": "ERROR_CODE",
  "message": "Human-readable description",
  "request_id": "uuid-for-tracing"
}
```

### Error Codes

| Code | HTTP | Description |
|------|------|-------------|
| `YOUTUBE_URL_REQUIRED` | 400 | Empty or null URL provided |
| `YOUTUBE_URL_INVALID` | 400 | Malformed YouTube URL |
| `YOUTUBE_URL_UNSUPPORTED` | 400 | Non-YouTube domain |
| `YOUTUBE_VIDEO_ID_INVALID` | 400 | Could not extract valid video ID |
| `TRANSCRIPT_LISTING_FAILED` | 400 | Video unavailable or transcripts disabled |
| `TRANSCRIPT_LANGUAGE_UNAVAILABLE` | 400 | No transcript in preferred languages |
| `TRANSCRIPT_UNAVAILABLE` | 400 | No captions available for video |
| `STT_DURATION_UNKNOWN` | 400 | Could not determine video duration |
| `STT_TOO_LONG` | 400 | Video exceeds max duration limit |
| `STT_DOWNLOAD_FAILED` | 400 | Failed to download audio |
| `STT_CHUNKING_DISABLED` | 400 | File too large, chunking not enabled |
| `STT_CHUNK_FAILED` | 400 | Audio chunking/splitting failed |
| `STT_FFMPEG_MISSING` | 400 | ffmpeg/ffprobe not found in PATH |

---

## Supported YouTube URL Formats

The service accepts various YouTube URL formats:

```
https://www.youtube.com/watch?v=VIDEO_ID
https://youtube.com/watch?v=VIDEO_ID
https://m.youtube.com/watch?v=VIDEO_ID
https://youtu.be/VIDEO_ID
https://www.youtube.com/shorts/VIDEO_ID
https://www.youtube.com/live/VIDEO_ID
```

**Video ID Requirements:**
- Exactly 11 characters
- Alphanumeric plus `-` and `_`

---

## Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI App                             │
│  POST /v1/captions  │  POST /v1/stt  │  GET /healthz        │
└──────────┬──────────┴────────┬───────┴──────────────────────┘
           │                   │
           ▼                   ▼
┌──────────────────┐  ┌────────────────────────────────────────┐
│ YouTubeTranscript│  │         STTTranscriptProvider          │
│    Provider      │  │                                        │
│                  │  │  ┌─────────┐  ┌─────────┐  ┌────────┐  │
│  youtube_transcript  │  Probe   │  │Download │  │Transcribe│ │
│       _api       │  │  (yt-dlp)│  │(yt-dlp) │  │(Whisper) │ │
└──────────────────┘  │  └────┬────┘  └────┬────┘  └────┬────┘  │
                      │       │            │            │       │
                      │       ▼            ▼            ▼       │
                      │  ┌─────────────────────────────────┐   │
                      │  │     AudioChunker (if enabled)   │   │
                      │  │  • Silence detection (ffmpeg)   │   │
                      │  │  • Optimal split calculation    │   │
                      │  │  • Stream copy splitting        │   │
                      │  └─────────────┬───────────────────┘   │
                      │                │                        │
                      │                ▼                        │
                      │  ┌─────────────────────────────────┐   │
                      │  │    ParallelTranscriber          │   │
                      │  │  • ThreadPoolExecutor           │   │
                      │  │  • Bounded concurrency          │   │
                      │  │  • Retry logic                  │   │
                      │  └─────────────┬───────────────────┘   │
                      │                │                        │
                      │                ▼                        │
                      │  ┌─────────────────────────────────┐   │
                      │  │    TranscriptMerger             │   │
                      │  │  • Timestamp adjustment         │   │
                      │  │  • Segment ordering             │   │
                      │  └─────────────────────────────────┘   │
                      └────────────────────────────────────────┘
```

### Data Flow

**Captions Flow:**
```
URL → extract_video_id → YouTubeTranscriptApi → clean_segments → response
```

**STT Flow:**
```
URL → probe(duration) → download_audio → [chunk if needed] →
      transcribe (parallel) → merge_segments → clean → response
```

### Protocol-Based Adapters

External dependencies are abstracted via Protocols:

```python
class YouTubeTranscriptClient(Protocol):
    def get_transcript(self, video_id: str, languages: list[str]) -> list[RawTranscriptItem]: ...
    def list_transcripts(self, video_id: str) -> TranscriptListLike: ...

class STTClient(Protocol):
    def transcribe_verbose(self, *, file: BinaryIO, timeout: float | None) -> VerboseResponseTD: ...

class ProbeDownloadClient(Protocol):
    def probe(self, url: str) -> YtInfoTD: ...
    def download_audio(self, url: str, *, cookies_path: str | None) -> str: ...
```

---

## Audio Chunking

When enabled, the service automatically splits large audio files for better reliability:

### How It Works

1. **Threshold Check**: If file size > `TRANSCRIPT_CHUNK_THRESHOLD_MB` or duration > `TRANSCRIPT_MAX_CHUNK_DURATION_SECONDS`
2. **Silence Detection**: Run `ffmpeg -af silencedetect` to find quiet sections
3. **Optimal Splits**: Calculate split points snapped to silence (within 30% tolerance)
4. **Stream Copy**: Use `ffmpeg -c copy` for fast, lossless splitting
5. **Parallel Transcription**: Process chunks concurrently with `ThreadPoolExecutor`
6. **Merge**: Adjust timestamps and concatenate segments

### Requirements

- `ffmpeg` and `ffprobe` must be in PATH
- Recommended: At least 2GB RAM for concurrent chunk processing

### Configuration

```bash
TRANSCRIPT_ENABLE_CHUNKING=1
TRANSCRIPT_CHUNK_THRESHOLD_MB=20      # Trigger chunking above this size
TRANSCRIPT_TARGET_CHUNK_MB=20         # Target size per chunk
TRANSCRIPT_MAX_CHUNK_DURATION_SECONDS=600  # Max 10 min per chunk
TRANSCRIPT_MAX_CONCURRENT_CHUNKS=3    # Parallel workers
TRANSCRIPT_SILENCE_THRESHOLD_DB=-40   # Silence detection sensitivity
TRANSCRIPT_SILENCE_DURATION_SECONDS=0.5  # Min silence duration
```

---

## Development

### Commands

```bash
make install      # Install dependencies
make install-dev  # Install with dev dependencies
make lint         # Run guards + ruff + mypy
make test         # Run pytest with coverage
make check        # Run lint + test
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
poetry run pytest tests/test_chunker.py -v

# Run with coverage report
poetry run pytest --cov-report=html
```

---

## Project Structure

```
transcript-api/
├── src/transcript_api/
│   ├── __init__.py
│   ├── asgi.py             # ASGI entry point
│   ├── startup.py          # App initialization
│   ├── settings.py         # Config loading
│   ├── types.py            # TypedDict models
│   ├── service.py          # Service orchestration
│   ├── provider.py         # YouTube caption provider
│   ├── stt_provider.py     # STT provider (Whisper)
│   ├── chunker.py          # Audio chunking logic
│   ├── merger.py           # Segment merging
│   ├── parallel.py         # Parallel transcription
│   ├── whisper_parse.py    # OpenAI response parsing
│   ├── vtt_parser.py       # VTT subtitle parsing
│   ├── youtube.py          # URL validation
│   ├── cleaner.py          # Text cleaning
│   ├── events.py           # Redis event publishing
│   ├── health.py           # Health check logic
│   ├── jobs.py             # Async job processing
│   ├── job_store.py        # Redis job state storage
│   ├── json_util.py        # JSON serialization helpers
│   ├── dependencies.py     # FastAPI dependencies
│   ├── worker_entry.py     # RQ worker entry point
│   ├── api/
│   │   ├── main.py         # FastAPI app factory
│   │   └── routes/
│   │       ├── health.py       # Health endpoints
│   │       ├── transcripts.py  # Captions/STT endpoints
│   │       └── jobs.py         # Async job endpoints
│   └── adapters/
│       ├── youtube_client.py   # youtube_transcript_api wrapper
│       ├── openai_client.py    # OpenAI Whisper client
│       └── yt_dlp_client.py    # yt-dlp wrapper
├── tests/
│   ├── test_app_api.py
│   ├── test_chunker*.py
│   ├── test_parallel_transcriber.py
│   ├── test_transcript_*.py
│   └── ...
├── scripts/
│   └── guard.py
├── Dockerfile
├── pyproject.toml
└── Makefile
```

---

## Deployment

### Docker

```bash
# Build
docker build -t transcript-api:latest .

# Run
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -e REDIS_URL=redis://host.docker.internal:6379/0 \
  -e TRANSCRIPT_ENABLE_CHUNKING=1 \
  transcript-api:latest
```

### Railway

```bash
# Set environment variables in Railway dashboard
# - OPENAI_API_KEY (required)
# - TRANSCRIPT_* variables as needed

railway up
```

**Health Check Path:** `/healthz`

---

## Dependencies

### Runtime

| Package | Purpose |
|---------|---------|
| `fastapi` | Web framework |
| `hypercorn` | ASGI server |
| `youtube-transcript-api` | YouTube caption extraction |
| `openai` | Whisper STT API |
| `yt-dlp` | YouTube audio download |
| `platform-core` | Logging, errors, config |

### System Requirements

| Tool | Purpose |
|------|---------|
| `ffmpeg` | Audio processing, silence detection |
| `ffprobe` | Audio file inspection |

### Development

| Package | Purpose |
|---------|---------|
| `pytest` | Test runner |
| `pytest-cov` | Coverage reporting |
| `pytest-xdist` | Parallel tests |
| `mypy` | Type checking |
| `ruff` | Linting/formatting |

---

## Discord Bot Integration

The Discord bot (`clients/DiscordBot`) integrates via:

1. **Direct HTTP**: `/transcript url:<YouTube URL>` calls `/v1/captions` or `/v1/stt`
2. **Redis Events**: Async job progress published to `transcript:events` channel

**Channel:** `transcript:events` (via `platform_core.job_events`)

**Events (generic job schema):**
- `transcript.job.started.v1` — `{type, domain, job_id, user_id, queue}`
- `transcript.job.progress.v1` — `{type, domain, job_id, user_id, progress, message?, payload?}`
- `transcript.job.completed.v1` — `{type, domain, job_id, user_id, result_id, result_bytes}`
- `transcript.job.failed.v1` — `{type, domain, job_id, user_id, error_kind, message}`

`result_id` is the canonical video ID (STT) or request URL (captions); `result_bytes` is the size of the transcript text in bytes. Progress messages are simple strings; no transcript-specific event variants remain.

---

## Quality Standards

- **Type Safety**: mypy strict mode, no `Any`, no `cast`
- **Coverage**: 100% statements and branches
- **Guard Rules**: Enforced via `scripts/guard.py`
- **Logging**: Structured JSON via platform_core
- **Errors**: Consistent `{code, message, request_id}` format

---

## License

Apache-2.0
