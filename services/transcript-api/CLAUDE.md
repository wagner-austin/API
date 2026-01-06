# AI Instructions for transcript-api

## What This Service Does

- Transcribes YouTube videos using OpenAI Whisper (STT)
- Falls back to YouTube captions when available
- Downloads audio via yt-dlp with cookie support
- Chunks large audio files for processing
- Parallel transcription for long videos

## Quick Usage (Direct Python)

```python
import sys
sys.path.insert(0, 'src')

from transcript_api.adapters.openai_client import OpenAISttClient
from transcript_api.adapters.yt_dlp_client import YtDlpAdapter
from transcript_api.stt_provider import STTTranscriptProvider

# Load from .env or hardcode
api_key = "sk-..."
cookies_text = "..."  # base64-encoded Netscape cookies

stt_client = OpenAISttClient(api_key=api_key)
probe_client = YtDlpAdapter()

provider = STTTranscriptProvider(
    stt_client=stt_client,
    probe_client=probe_client,
    max_video_seconds=3600,
    max_file_mb=100,
    cookies_text=cookies_text,
    enable_chunking=True,
)

video_id = "dQw4w9WgXcQ"  # YouTube video ID
segments = provider.fetch(video_id, {"preferred_langs": ["en"]})
full_text = " ".join([s["text"] for s in segments])
print(full_text)
```

## Production API

Base URL: `https://transcript-api-production-2753.up.railway.app`

```bash
# Health check (liveness probe)
curl https://transcript-api-production-2753.up.railway.app/healthz

# Get captions (YouTube native, faster but may fail)
curl -X POST https://transcript-api-production-2753.up.railway.app/v1/captions \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=VIDEO_ID", "preferred_langs": ["en"]}'

# Transcribe a video (STT/Whisper, requires OPENAI_API_KEY)
curl -X POST https://transcript-api-production-2753.up.railway.app/v1/stt \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=VIDEO_ID"}'
```

## Local Development

```bash
# Health check
curl http://localhost:8000/healthz

# Captions
curl -X POST http://localhost:8000/v1/captions \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=VIDEO_ID", "preferred_langs": ["en"]}'

# STT
curl -X POST http://localhost:8000/v1/stt \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=VIDEO_ID"}'
```

## Architecture

```
transcript-api/
├── src/transcript_api/
│   ├── adapters/
│   │   ├── openai_client.py   # Whisper API client
│   │   ├── yt_dlp_client.py   # Audio/subtitle download
│   │   └── youtube_client.py  # YouTube captions API
│   ├── stt_provider.py        # Main STT orchestration
│   ├── provider.py            # Caption provider
│   ├── chunker.py             # Audio chunking (ffmpeg)
│   ├── parallel.py            # Concurrent transcription
│   ├── merger.py              # Segment merging
│   ├── vtt_parser.py          # VTT subtitle parsing
│   └── api/                   # FastAPI routes
```

## Key Environment Variables

```bash
OPENAI_API_KEY=sk-...                    # Required for Whisper
TRANSCRIPT_COOKIES_TEXT=...              # Base64 Netscape cookies for yt-dlp
TRANSCRIPT_MAX_VIDEO_SECONDS=3600        # Max video length (default 1hr)
TRANSCRIPT_MAX_FILE_MB=100               # Max audio file size
TRANSCRIPT_ENABLE_CHUNKING=true          # Enable audio chunking
```

## Related Libraries

- **platform_stt** (`libs/platform_stt`) - Shared STT library with Whisper client, chunker, merger
- Use `platform_stt` directly when you have audio files (no YouTube download needed)

## Transcription Strategy

1. **Try YouTube captions first** (fast, free) via `youtube_transcript_api`
2. **Fall back to Whisper STT** if captions unavailable/malformed
3. **Chunk large files** (>20MB) using ffmpeg silence detection
4. **Parallel transcription** for chunks with bounded concurrency

## Testing Hooks

```python
from transcript_api import _test_hooks

# Override YouTube API for testing
_test_hooks.yt_api_factory = lambda: FakeYouTubeApi()

# Override OpenAI client
_test_hooks.openai_client_factory = lambda **kw: FakeOpenAIClient()

# Override yt-dlp
_test_hooks.yt_dlp_factory = lambda opts: FakeYtDlp()
```

## Common Issues

| Issue | Solution |
|-------|----------|
| `ParseError: no element found` | YouTube captions malformed, use STT instead |
| `Video unavailable` | Check cookies, video may be private/age-restricted |
| `STT_TOO_LONG` | Video exceeds `max_video_seconds` limit |
| `ffmpeg not found` | Install ffmpeg for chunking support |
