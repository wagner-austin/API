# Turkic API - API Reference

Complete API documentation for the turkic-api corpus processing service.

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

### GET /readyz

Readiness probe. Returns 503 if Redis or data volume unavailable.

**Response (200):**
```json
{
  "status": "ready",
  "reason": null
}
```

**Response (503):**
```json
{
  "status": "degraded",
  "reason": "redis unavailable"
}
```

---

## Job Endpoints

### POST /api/v1/jobs

Create a new corpus extraction job.

**Content-Type:** `application/json`

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `user_id` | int | Yes | - | User ID for tracking |
| `source` | string | Yes | - | Dataset source: `oscar`, `wikipedia`, `culturax` |
| `language` | string | Yes | - | Language code: `kk`, `ky`, `uz`, `tr`, `ug`, `fi`, `az`, `en` |
| `script` | string | No | `null` | Script filter: `Latn`, `Cyrl`, `Arab` |
| `max_sentences` | int | No | `1000` | Maximum sentences to extract (1-100000) |
| `transliterate` | bool | No | `true` | Apply IPA transliteration to output |
| `confidence_threshold` | float | No | `0.95` | Language ID confidence threshold (0.0-1.0) |

**Request Headers:**

| Header | Required | Description |
|--------|----------|-------------|
| `Content-Type` | Yes | Must be `application/json` |
| `X-Request-ID` | No | Correlation ID (auto-generated if omitted) |

**Response (200):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "user_id": 12345,
  "status": "queued",
  "created_at": "2024-01-15T10:30:00"
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | string | Unique job identifier (UUID) |
| `user_id` | int | User ID from request |
| `status` | string | Initial status, always `queued` |
| `created_at` | string | ISO 8601 timestamp |

**Example - curl:**
```bash
curl -X POST http://localhost:8000/api/v1/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": 12345,
    "source": "oscar",
    "language": "kk",
    "script": "Cyrl",
    "max_sentences": 1000,
    "transliterate": true,
    "confidence_threshold": 0.95
  }'
```

**Example - Python:**
```python
import httpx

response = httpx.post(
    "http://localhost:8000/api/v1/jobs",
    json={
        "user_id": 12345,
        "source": "oscar",
        "language": "kk",
        "script": "Cyrl",
        "max_sentences": 1000,
        "transliterate": True,
        "confidence_threshold": 0.95,
    },
)

response.raise_for_status()
result = response.json()
print(f"Job created: {result['job_id']}")
```

---

### GET /api/v1/jobs/{job_id}

Get the status of an existing job.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | string | Job UUID from create response |

**Response (200):**
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "progress": 100,
  "message": null,
  "result_url": "/api/v1/jobs/550e8400-e29b-41d4-a716-446655440000/result",
  "file_id": "abc123def456",
  "upload_status": "uploaded",
  "created_at": "2024-01-15T10:30:00",
  "updated_at": "2024-01-15T10:35:00",
  "error": null
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | string | Job identifier |
| `status` | string | `queued`, `processing`, `completed`, `failed` |
| `progress` | int | Progress percentage (0-100) |
| `message` | string\|null | Current processing step description |
| `result_url` | string\|null | URL to download result (when completed) |
| `file_id` | string\|null | Data-bank file ID (when uploaded) |
| `upload_status` | string\|null | `uploaded` when result is ready |
| `created_at` | string | Job creation timestamp |
| `updated_at` | string | Last status update timestamp |
| `error` | string\|null | Error message (when failed) |

**Response (404):**
```json
{
  "code": "JOB_NOT_FOUND",
  "message": "Job not found"
}
```

**Example - polling for completion:**
```python
import httpx
import time

job_id = "550e8400-e29b-41d4-a716-446655440000"

while True:
    response = httpx.get(f"http://localhost:8000/api/v1/jobs/{job_id}")
    response.raise_for_status()
    status = response.json()

    print(f"Status: {status['status']} ({status['progress']}%)")

    if status["status"] == "completed":
        print(f"Result ready: {status['result_url']}")
        break
    elif status["status"] == "failed":
        print(f"Job failed: {status['error']}")
        break

    time.sleep(5)
```

---

### GET /api/v1/jobs/{job_id}/result

Download the processed result file.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | string | Job UUID |

**Response (200):**

Streaming response with processed sentences, one per line (UTF-8 plain text).

**Response Headers:**

| Header | Value |
|--------|-------|
| `Content-Type` | `text/plain; charset=utf-8` |
| `Content-Disposition` | `attachment; filename="result_{job_id}.txt"` |
| `Transfer-Encoding` | `chunked` |

**Error Responses:**

| Status | Code | Description |
|--------|------|-------------|
| 404 | `JOB_NOT_FOUND` | Job does not exist |
| 410 | `JOB_FAILED` | Job failed or result expired |
| 425 | `JOB_NOT_READY` | Job not completed yet |
| 502 | `EXTERNAL_SERVICE_ERROR` | Data-bank streaming error |

**Example - download result:**
```bash
curl -O http://localhost:8000/api/v1/jobs/550e8400-e29b-41d4-a716-446655440000/result
```

**Example - Python streaming:**
```python
import httpx

job_id = "550e8400-e29b-41d4-a716-446655440000"

with httpx.stream("GET", f"http://localhost:8000/api/v1/jobs/{job_id}/result") as response:
    response.raise_for_status()
    with open(f"result_{job_id}.txt", "wb") as f:
        for chunk in response.iter_bytes():
            f.write(chunk)
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

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `INVALID_REQUEST` | 400 | Malformed request body or invalid parameters |
| `JOB_NOT_FOUND` | 404 | Job ID does not exist |
| `JOB_NOT_READY` | 425 | Job not completed, try again later |
| `JOB_FAILED` | 410 | Job failed or result expired |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Unexpected server error |
| `EXTERNAL_SERVICE_ERROR` | 502 | Data-bank or upstream service error |

### Error Examples

**Invalid language:**
```json
{
  "code": "INVALID_REQUEST",
  "message": "Invalid language",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Job not found:**
```json
{
  "code": "JOB_NOT_FOUND",
  "message": "Job not found",
  "request_id": "550e8400-e29b-41d4-a716-446655440001"
}
```

**Job still processing:**
```json
{
  "code": "JOB_NOT_READY",
  "message": "Job not completed",
  "request_id": "550e8400-e29b-41d4-a716-446655440002"
}
```

---

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TURKIC_REDIS_URL` | No | `redis://redis:6379/0` | Redis connection URL |
| `TURKIC_DATA_DIR` | No | `/data` | Path to data directory |
| `TURKIC_DATA_BANK_API_URL` | Conditional | - | Data-bank API base URL (or use `API_GATEWAY_URL`) |
| `TURKIC_DATA_BANK_API_KEY` | Yes | - | Data-bank API key |
| `API_GATEWAY_URL` | Conditional | - | API gateway URL (auto-appends `/data-bank`) |

---

## Request ID Tracing

All requests are assigned a unique `request_id` for tracing:

- **Provided:** Pass `X-Request-ID` header
- **Generated:** UUID v4 auto-generated if header omitted

The `request_id` appears in:
- All error responses
- Structured logs
- Response headers (when applicable)

---

## Content Types

**Request content type:**
- `application/json` (for job creation)

**Response content types:**
- `application/json` (for status and errors)
- `text/plain; charset=utf-8` (for result download)

---

## Supported Languages

| Code | Language | Scripts | IPA Rules |
|------|----------|---------|-----------|
| `kk` | Kazakh | Cyrillic, Latin | Yes |
| `ky` | Kyrgyz | Cyrillic, Latin | Yes |
| `uz` | Uzbek | Cyrillic, Latin | Yes |
| `tr` | Turkish | Latin | Yes |
| `ug` | Uyghur | Arabic | Yes |
| `fi` | Finnish | Latin | Yes |
| `az` | Azerbaijani | Latin | Yes |
| `en` | English | Latin | Yes |

---

## Data Sources

| Source | Description | Coverage |
|--------|-------------|----------|
| `oscar` | OSCAR corpus from HuggingFace | All languages |
| `wikipedia` | Wikipedia XML dumps | All languages |
| `culturax` | CulturaX multilingual corpus | All languages |

---

## Job Lifecycle

```
┌──────────┐     ┌────────────┐     ┌───────────┐     ┌──────────┐
│  queued  │ ──▶ │ processing │ ──▶ │ completed │ ──▶ │ download │
└──────────┘     └────────────┘     └───────────┘     └──────────┘
                       │
                       ▼
                 ┌──────────┐
                 │  failed  │
                 └──────────┘
```

1. **queued** - Job created, waiting for worker
2. **processing** - Worker extracting and processing corpus
3. **completed** - Result uploaded to data-bank, ready for download
4. **failed** - Error occurred, check `error` field for details
