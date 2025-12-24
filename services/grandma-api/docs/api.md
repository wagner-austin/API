# Grandma API - API Reference

Complete API documentation for the grandma-api service.

**Base URL:** `http://localhost:8080` (default)

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

## Translation Endpoints

### POST /translate

Translate audio to English text using OpenAI Whisper. Supports 57 input languages with automatic language detection.

**Request Headers:**

| Header | Required | Description |
|--------|----------|-------------|
| `Content-Type` | Yes | Must be `multipart/form-data` |
| `X-Request-ID` | No | Correlation ID (auto-generated if omitted) |

**Request Body (multipart/form-data):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio` | file | Yes | Audio file to translate (webm, mp3, wav, m4a, ogg supported) |
| `token` | string | Yes | Authentication token |

**Request Example (curl):**
```bash
curl -X POST http://localhost:8080/translate \
  -F "audio=@recording.webm" \
  -F "token=your-api-token"
```

**Request Example (Python):**
```python
import httpx

with open("recording.webm", "rb") as f:
    response = httpx.post(
        "http://localhost:8080/translate",
        files={"audio": ("recording.webm", f, "audio/webm")},
        data={"token": "your-api-token"},
    )
    result = response.json()
    print(result["text"])
```

**Response (200):**
```json
{
  "text": "Hello, how are you grandmother?"
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `text` | string | Translated English text from the input audio |

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
| `INVALID_INPUT` | 400 | Invalid request body or parameters (e.g., empty audio) |
| `UNAUTHORIZED` | 401 | Missing or invalid authentication token |
| `INTERNAL_ERROR` | 500 | Internal server error |

### Error Examples

**Empty audio file:**
```json
{
  "code": "INVALID_INPUT",
  "message": "No audio file provided",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Invalid token:**
```json
{
  "code": "UNAUTHORIZED",
  "message": "Invalid token",
  "request_id": "550e8400-e29b-41d4-a716-446655440001"
}
```

---

## Authentication

The `/translate` endpoint requires authentication via the `token` form field.

**Configure authentication:**
```bash
API_TOKEN=your-secret-token
```

**Send token:**
```bash
curl -X POST http://localhost:8080/translate \
  -F "audio=@recording.webm" \
  -F "token=your-secret-token"
```

---

## Request ID Tracing

All requests are assigned a unique `request_id` for tracing:

- **Provided:** Pass `X-Request-ID` header
- **Generated:** UUID v4 auto-generated if header omitted

The `request_id` appears in:
- All error responses
- Structured logs

---

## Content Types

**Request content type:**
- `multipart/form-data` (for `/translate` endpoint)

**Response content type:**
- `application/json` (all endpoints)

---

## Language Support

### Input Languages (57)

Whisper auto-detects the source language. Supported languages:

Afrikaans, Arabic, Armenian, Azerbaijani, Belarusian, Bosnian, Bulgarian, Catalan, Chinese, Croatian, Czech, Danish, Dutch, English, Estonian, Finnish, French, Galician, German, Greek, Hebrew, Hindi, Hungarian, Icelandic, Indonesian, Italian, Japanese, Kannada, Kazakh, Korean, Latvian, Lithuanian, Macedonian, Malay, Marathi, Maori, Nepali, Norwegian, Persian, Polish, Portuguese, Romanian, Russian, Serbian, Slovak, Slovenian, Spanish, Swahili, Swedish, Tagalog, Tamil, Thai, Turkish, Ukrainian, Urdu, Vietnamese, Welsh.

### Output Language

**English only** - Whisper API limitation; the translate endpoint only supports English output.

### Language Detection Notes

- Language detection is automatic; no parameter required
- Best accuracy for major languages (English, Spanish, French, German, Chinese, Japanese)
- Reduced accuracy for low-resource languages (Icelandic, Welsh, Maori, Swahili)

---

## Audio Format Support

The following audio formats are supported:
- WebM (`.webm`) - Recommended for browser recordings
- MP3 (`.mp3`)
- WAV (`.wav`)
- M4A (`.m4a`)
- OGG (`.ogg`)

Maximum audio duration: Determined by OpenAI API limits (typically 25MB file size)

---

## Rate Limits

Rate limits are enforced by the OpenAI Whisper API. Standard OpenAI rate limits apply:
- Requests per minute: Varies by account tier
- Audio minutes per day: Varies by account tier

---

## CORS

CORS is enabled for all origins to support browser-based clients:
- `Access-Control-Allow-Origin: *`
- `Access-Control-Allow-Methods: GET, POST, OPTIONS`
- `Access-Control-Allow-Headers: *`
