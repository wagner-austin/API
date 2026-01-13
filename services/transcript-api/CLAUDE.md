# AI Instructions for transcript-api

## What This Service Does

- Transcribes videos from YouTube, Vimeo, and direct URLs using OpenAI Whisper (STT)
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
│   ├── url/                   # URL parsing module
│   │   ├── types.py           # ParsedURL TypedDicts
│   │   ├── youtube.py         # YouTube URL parser
│   │   ├── vimeo.py           # Vimeo URL parser
│   │   ├── direct.py          # Direct file URL parser
│   │   └── parse.py           # Unified parse_video_url()
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

## URL Parsing

The `url/` module provides unified URL parsing for multiple video sources:

```python
from transcript_api.url import parse_video_url

# YouTube (watch, shorts, live, youtu.be)
parsed = parse_video_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
# {"source": "youtube", "video_id": "dQw4w9WgXcQ", "canonical_url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ"}

# Vimeo (standard, player.vimeo.com)
parsed = parse_video_url("https://vimeo.com/123456789")
# {"source": "vimeo", "video_id": "123456789", "canonical_url": "https://vimeo.com/123456789"}

# Direct video/audio URLs (.mp4, .webm, .mp3, etc.)
parsed = parse_video_url("https://example.com/video.mp4")
# {"source": "direct", "video_id": "<md5>", "canonical_url": "https://example.com/video.mp4", "extension": "mp4"}
```

Supported formats:
- **YouTube**: watch, shorts, live, youtu.be
- **Vimeo**: vimeo.com, player.vimeo.com
- **Direct**: mp4, webm, mkv, avi, mov, mp3, wav, flac, m4a, ogg

## Common Issues

| Issue | Solution |
|-------|----------|
| `VIDEO_URL_UNSUPPORTED` | URL must be YouTube, Vimeo, or direct video file |
| `ParseError: no element found` | YouTube captions malformed, use STT instead |
| `Video unavailable` | Check cookies, video may be private/age-restricted |
| `STT_TOO_LONG` | Video exceeds `max_video_seconds` limit |
| `ffmpeg not found` | Install ffmpeg for chunking support |
