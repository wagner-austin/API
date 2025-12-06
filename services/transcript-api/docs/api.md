# Transcript API - API Reference

Complete API documentation for the transcript-api service.

**Base URL:** `http://localhost:8000` (default)

---

## Health Endpoints

### GET /healthz

Liveness probe for container orchestration.

**Response (200):**
```json
{
  "status": "ok"
}
```

---

## Transcript Endpoints

### POST /v1/captions

Extract YouTube native captions/transcripts.

**Content-Type:** `application/json`

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `url` | string | Yes | - | YouTube video URL |
| `preferred_langs` | string[] | No | `["en", "en-US", "en-GB"]` | Language preference order |

**Request Headers:**

| Header | Required | Description |
|--------|----------|-------------|
| `Content-Type` | Yes | Must be `application/json` |
| `X-Request-ID` | No | Correlation ID (auto-generated if omitted) |

**Response (200):**
```json
{
  "url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "video_id": "VIDEO_ID",
  "text": "Transcript text content..."
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `url` | string | Canonical YouTube URL |
| `video_id` | string | 11-character YouTube video ID |
| `text` | string | Full transcript text, cleaned and joined |

**Example - curl:**
```bash
curl -X POST http://localhost:8000/v1/captions \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "preferred_langs": ["en", "es"]
  }'
```

**Example - Python:**
```python
import httpx

response = httpx.post(
    "http://localhost:8000/v1/captions",
    json={
        "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "preferred_langs": ["en", "es"],
    },
)

response.raise_for_status()
result = response.json()
print(f"Transcript: {result['text'][:100]}...")
```

**Fallback Logic:**
1. Try direct transcript fetch in preferred languages
2. If unavailable, try `list_transcripts()` API
3. If still unavailable, attempt auto-translation to first preferred language
4. Return error if all attempts fail

---

### POST /v1/stt

Transcribe video audio using OpenAI Whisper.

**Content-Type:** `application/json`

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `url` | string | Yes | YouTube video URL |

**Request Headers:**

| Header | Required | Description |
|--------|----------|-------------|
| `Content-Type` | Yes | Must be `application/json` |
| `X-Request-ID` | No | Correlation ID (auto-generated if omitted) |

**Response (200):**
```json
{
  "url": "https://www.youtube.com/watch?v=VIDEO_ID",
  "video_id": "VIDEO_ID",
  "text": "Transcribed audio content..."
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `url` | string | Canonical YouTube URL |
| `video_id` | string | 11-character YouTube video ID |
| `text` | string | Full transcription text, cleaned |

**Example - curl:**
```bash
curl -X POST http://localhost:8000/v1/stt \
  -H "Content-Type: application/json" \
  -d '{"url": "https://youtu.be/dQw4w9WgXcQ"}'
```

**Example - Python:**
```python
import httpx

response = httpx.post(
    "http://localhost:8000/v1/stt",
    json={"url": "https://youtu.be/dQw4w9WgXcQ"},
    timeout=120.0,  # STT can be slow for long videos
)

response.raise_for_status()
result = response.json()
print(f"Transcription: {result['text'][:100]}...")
```

**Processing Flow:**
1. Probe video to get duration via yt-dlp
2. Validate against `TRANSCRIPT_MAX_VIDEO_SECONDS`
3. Download audio via yt-dlp
4. If file size exceeds threshold and chunking enabled:
   - Detect silence points with ffmpeg
   - Split audio at optimal points
   - Transcribe chunks in parallel
   - Merge segments with adjusted timestamps
5. Otherwise, transcribe entire file via OpenAI Whisper
6. Clean and return text

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

#### URL Validation Errors

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `YOUTUBE_URL_REQUIRED` | 400 | Empty or null URL provided |
| `YOUTUBE_URL_INVALID` | 400 | Malformed URL format |
| `YOUTUBE_URL_UNSUPPORTED` | 400 | Non-YouTube domain |
| `YOUTUBE_VIDEO_ID_INVALID` | 400 | Could not extract valid 11-character video ID |

#### Transcript Errors (Captions)

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `TRANSCRIPT_UNAVAILABLE` | 400 | No captions available for video |
| `TRANSCRIPT_LANGUAGE_UNAVAILABLE` | 400 | No transcript in preferred languages |
| `TRANSCRIPT_TRANSLATE_UNAVAILABLE` | 400 | Translation to preferred language failed |
| `TRANSCRIPT_LISTING_FAILED` | 400 | Video unavailable or transcripts disabled |
| `TRANSCRIPT_PAYLOAD_INVALID` | 400 | Invalid response structure from YouTube API |

#### STT Errors (Speech-to-Text)

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `STT_DURATION_UNKNOWN` | 400 | Could not determine video duration |
| `STT_TOO_LONG` | 400 | Video exceeds max duration limit |
| `STT_DOWNLOAD_FAILED` | 400 | Failed to download audio via yt-dlp |
| `STT_CHUNKING_DISABLED` | 400 | File too large, chunking not enabled |
| `STT_CHUNK_FAILED` | 400 | Audio chunking/splitting failed |
| `STT_FFMPEG_MISSING` | 400 | ffmpeg/ffprobe not found in PATH |

### Error Examples

**Invalid URL:**
```json
{
  "code": "YOUTUBE_URL_INVALID",
  "message": "Invalid YouTube URL format",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**No captions available:**
```json
{
  "code": "TRANSCRIPT_UNAVAILABLE",
  "message": "No captions available for this video",
  "request_id": "550e8400-e29b-41d4-a716-446655440001"
}
```

**Video too long:**
```json
{
  "code": "STT_TOO_LONG",
  "message": "Video duration 7200s exceeds limit of 3600s",
  "request_id": "550e8400-e29b-41d4-a716-446655440002"
}
```

---

## Configuration

| Constraint | Default | Environment Variable |
|------------|---------|---------------------|
| Max video duration | Unlimited | `TRANSCRIPT_MAX_VIDEO_SECONDS` |
| Max file size | Unlimited | `TRANSCRIPT_MAX_FILE_MB` |
| Chunk threshold | 20 MB | `TRANSCRIPT_CHUNK_THRESHOLD_MB` |
| Target chunk size | 20 MB | `TRANSCRIPT_TARGET_CHUNK_MB` |
| Max chunk duration | 600s | `TRANSCRIPT_MAX_CHUNK_DURATION_SECONDS` |
| Concurrent chunks | 3 | `TRANSCRIPT_MAX_CONCURRENT_CHUNKS` |

---

## Request ID Tracing

All requests are assigned a unique `request_id` for tracing:

- **Provided:** Pass `X-Request-ID` header
- **Generated:** UUID v4 auto-generated if header omitted

The `request_id` appears in:
- All error responses
- Structured logs
- Redis event payloads (if configured)

---

## Content Types

**Request content type:**
- `application/json`

**Response content type:**
- `application/json`

---

## Supported YouTube URL Formats

The service accepts various YouTube URL formats:

| Format | Example |
|--------|---------|
| Standard watch | `https://www.youtube.com/watch?v=VIDEO_ID` |
| Short domain | `https://youtu.be/VIDEO_ID` |
| Shorts | `https://www.youtube.com/shorts/VIDEO_ID` |
| Live | `https://www.youtube.com/live/VIDEO_ID` |
| Mobile | `https://m.youtube.com/watch?v=VIDEO_ID` |
| No www | `https://youtube.com/watch?v=VIDEO_ID` |

**Video ID Requirements:**
- Exactly 11 characters
- Alphanumeric plus `-` and `_`

---

## Event Publishing (Optional)

If `REDIS_URL` is configured, the service publishes events to Redis:

**Channel:** `transcript:events` (from `platform_core.default_events_channel`)

**Event Types (generic job schema):**
- `transcript.job.started.v1` — `{type, domain, job_id, user_id, queue}`
- `transcript.job.progress.v1` — `{type, domain, job_id, user_id, progress, message?, payload?}`
- `transcript.job.completed.v1` — `{type, domain, job_id, user_id, result_id, result_bytes}`
- `transcript.job.failed.v1` — `{type, domain, job_id, user_id, error_kind, message}`

`result_id` is the canonical video ID (STT) or request URL (captions); `result_bytes` is the size of the transcript text in bytes. Progress messages are simple strings; no legacy transcript-specific payloads remain.
