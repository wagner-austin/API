# music-wrapped-api - API Reference

Complete API documentation for the music wrapped service.

## Base URL

```
http://localhost:8006
```

---

## Health Endpoints

### GET /healthz

Liveness probe - checks if the service is running.

**Response:** `200 OK`

```json
{"status": "ok"}
```

### GET /readyz

Readiness probe - checks if the service can handle requests.

**Response:** `200 OK` or `503 Service Unavailable`

```json
{"status": "ready"}
```

---

## Authentication Endpoints

### GET /v1/wrapped/auth/spotify/start

Initiate Spotify OAuth 2.0 authorization flow.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `callback` | string | Yes | OAuth redirect URI |

**Response:** `200 OK`

```json
{
  "auth_url": "https://accounts.spotify.com/authorize?client_id=...&response_type=code&redirect_uri=...",
  "state": "random_state_token"
}
```

**Notes:**
- Requested scopes: `user-read-recently-played`, `user-top-read`
- State token is stored in Redis for CSRF protection

---

### GET /v1/wrapped/auth/spotify/callback

Complete Spotify OAuth flow by exchanging authorization code for tokens.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `code` | string | Yes | Authorization code from Spotify |
| `state` | string | Yes | State token for CSRF validation |
| `callback` | string | Yes | Original redirect URI |

**Response:** `200 OK`

```json
{
  "token_id": "abc123def456789...",
  "expires_in": 3600
}
```

**Errors:**

| HTTP | Code | Message |
|------|------|---------|
| 400 | `INVALID_INPUT` | invalid state |
| 502 | `EXTERNAL_SERVICE_ERROR` | invalid token fields |

---

### GET /v1/wrapped/auth/lastfm/start

Initiate Last.fm web authentication flow.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `callback` | string | Yes | Callback URL for auth completion |

**Response:** `200 OK`

```json
{
  "auth_url": "https://www.last.fm/api/auth/?api_key=...&cb=..."
}
```

---

### GET /v1/wrapped/auth/lastfm/callback

Exchange Last.fm authentication token for session key.

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `token` | string | Yes | Token from Last.fm callback |

**Response:** `200 OK`

```json
{
  "session_key": "abc123session...",
  "username": "lastfm_username"
}
```

**Errors:**

| HTTP | Code | Message |
|------|------|---------|
| 502 | `EXTERNAL_SERVICE_ERROR` | invalid lastfm json |
| 502 | `EXTERNAL_SERVICE_ERROR` | missing session |

---

### POST /v1/wrapped/auth/youtube/store

Store YouTube Music credentials (SAPISID and cookies).

**Request Body:**

```json
{
  "sapisid": "SAPISID_VALUE",
  "cookies": "FULL_COOKIE_STRING"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `sapisid` | string | Yes | SAPISID cookie value |
| `cookies` | string | Yes | Full cookie header string |

**Response:** `200 OK`

```json
{
  "token_id": "sha256_hash_first_32_chars"
}
```

**Errors:**

| HTTP | Code | Message |
|------|------|---------|
| 400 | `INVALID_INPUT` | invalid youtube_music credentials |

---

### POST /v1/wrapped/auth/apple/store

Store Apple Music user token.

**Request Body:**

```json
{
  "music_user_token": "MUSIC_USER_TOKEN_VALUE"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `music_user_token` | string | Yes | Apple Music user token |

**Response:** `200 OK`

```json
{
  "token_id": "sha256_hash_first_32_chars"
}
```

**Errors:**

| HTTP | Code | Message |
|------|------|---------|
| 400 | `INVALID_INPUT` | invalid apple store input |

---

## Wrapped Generation Endpoints

### POST /v1/wrapped/generate

Start a wrapped report generation job.

**Request Body (Token Reference):**

Use a previously stored token:

```json
{
  "year": 2024,
  "service": "spotify",
  "credentials": {
    "token_id": "stored_token_id"
  }
}
```

**Request Body (Full Credentials):**

Provide credentials directly:

```json
{
  "year": 2024,
  "service": "lastfm",
  "credentials": {
    "session_key": "session_key_value"
  }
}
```

**Services and Credential Formats:**

| Service | Credential Fields |
|---------|-------------------|
| `spotify` | `token_id` OR `access_token`, `refresh_token`, `expires_in` |
| `lastfm` | `session_key` (optional: `api_key`, `api_secret`) |
| `apple_music` | `token_id` OR `music_user_token`, `developer_token` |
| `youtube_music` | `token_id` OR `sapisid`, `cookies` |

**Response:** `200 OK`

```json
{
  "job_id": "rq_job_id_string",
  "status": "queued"
}
```

**Errors:**

| HTTP | Code | Message |
|------|------|---------|
| 400 | `INVALID_INPUT` | object body required |
| 400 | `INVALID_INPUT` | year must be int |
| 400 | `INVALID_INPUT` | service required |
| 400 | `INVALID_INPUT` | unsupported service |
| 400 | `INVALID_INPUT` | invalid spotify credentials |
| 404 | `NOT_FOUND` | spotify token not found |
| 404 | `NOT_FOUND` | apple token not found |
| 404 | `NOT_FOUND` | youtube token not found |

---

### POST /v1/wrapped/import/youtube-takeout

Import YouTube Music history from Google Takeout export.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | Takeout JSON or ZIP file |
| `year` | string | Yes | Year to generate wrapped for |

**Example:**

```bash
curl -X POST http://localhost:8006/v1/wrapped/import/youtube-takeout \
  -F "file=@watch-history.json" \
  -F "year=2024"
```

**Response:** `200 OK`

```json
{
  "job_id": "rq_job_id",
  "status": "queued",
  "token_id": "content_hash_id"
}
```

**Errors:**

| HTTP | Code | Message |
|------|------|---------|
| 400 | `INVALID_INPUT` | invalid multipart fields |
| 400 | `INVALID_INPUT` | invalid multipart counts |
| 400 | `INVALID_INPUT` | year must be int |

---

### GET /v1/wrapped/status/{job_id}

Get the status of a wrapped generation job.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | string | RQ job ID from generate response |

**Response:** `200 OK`

```json
{
  "job_id": "rq_job_id",
  "status": "finished",
  "progress": 100,
  "result_id": "redis_result_key"
}
```

**Status Values:**

| Status | Description |
|--------|-------------|
| `queued` | Job is waiting to be processed |
| `started` | Job is currently running |
| `finished` | Job completed successfully |
| `failed` | Job failed with error |

**Progress:** Integer 0-100 representing completion percentage.

---

### GET /v1/wrapped/result/{result_id}

Get the wrapped result as JSON.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `result_id` | string | Result ID from status response |

**Response:** `200 OK` `application/json`

```json
{
  "service": "spotify",
  "year": 2024,
  "generated_at": "2024-12-15T10:30:00Z",
  "total_scrobbles": 12345,
  "top_artists": [
    {"artist_name": "Artist One", "play_count": 500},
    {"artist_name": "Artist Two", "play_count": 350}
  ],
  "top_songs": [
    {"title": "Song Title", "artist_name": "Artist", "play_count": 100}
  ],
  "top_by_month": [
    {
      "month": 1,
      "top_artists": [{"artist_name": "January Artist", "play_count": 50}]
    }
  ]
}
```

**Errors:**

| HTTP | Code | Message |
|------|------|---------|
| 404 | `NOT_FOUND` | wrapped result not found |

---

### GET /v1/wrapped/download/{result_id}

Download the wrapped result as a PNG image.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `result_id` | string | Result ID from status response |

**Response:** `200 OK` `image/png`

Binary PNG image data.

**Errors:**

| HTTP | Code | Message |
|------|------|---------|
| 404 | `NOT_FOUND` | wrapped result not found |

---

### GET /v1/wrapped/schema

Get the JSON Schema for WrappedResult.

**Response:** `200 OK` `application/json`

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "title": "WrappedResult",
  "type": "object",
  "properties": {
    "service": {
      "type": "string",
      "enum": ["lastfm", "spotify", "apple_music", "youtube_music"]
    },
    "year": {"type": "integer"},
    "generated_at": {"type": "string"},
    "total_scrobbles": {"type": "integer", "minimum": 0},
    "top_artists": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "artist_name": {"type": "string"},
          "play_count": {"type": "integer", "minimum": 0}
        },
        "required": ["artist_name", "play_count"]
      }
    },
    "top_songs": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "title": {"type": "string"},
          "artist_name": {"type": "string"},
          "play_count": {"type": "integer", "minimum": 0}
        },
        "required": ["title", "artist_name", "play_count"]
      }
    },
    "top_by_month": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "month": {"type": "integer", "minimum": 1, "maximum": 12},
          "top_artists": {"type": "array"}
        },
        "required": ["month", "top_artists"]
      }
    }
  },
  "required": ["service", "year", "generated_at", "total_scrobbles", "top_artists", "top_songs", "top_by_month"]
}
```

---

## Data Types

### WrappedResult

Complete wrapped analytics result.

| Field | Type | Description |
|-------|------|-------------|
| `service` | string | Service name: `lastfm`, `spotify`, `apple_music`, `youtube_music` |
| `year` | integer | Year of the wrapped report |
| `generated_at` | string | ISO 8601 timestamp |
| `total_scrobbles` | integer | Total plays/scrobbles for the year |
| `top_artists` | TopArtist[] | Top artists by play count |
| `top_songs` | TopSong[] | Top songs by play count |
| `top_by_month` | MonthEntry[] | Top artists broken down by month |

### TopArtist

| Field | Type | Description |
|-------|------|-------------|
| `artist_name` | string | Artist name |
| `play_count` | integer | Number of plays |

### TopSong

| Field | Type | Description |
|-------|------|-------------|
| `title` | string | Song title |
| `artist_name` | string | Artist name |
| `play_count` | integer | Number of plays |

### MonthEntry

| Field | Type | Description |
|-------|------|-------------|
| `month` | integer | Month number (1-12) |
| `top_artists` | TopArtist[] | Top artists for that month |

---

## Error Response Format

All errors follow the platform standard error format:

```json
{
  "error": {
    "code": "ERROR_CODE",
    "message": "Human-readable error message"
  }
}
```

### Error Codes

| Code | HTTP | Description |
|------|------|-------------|
| `INVALID_INPUT` | 400 | Invalid request body or parameters |
| `NOT_FOUND` | 404 | Resource (token, result, job) not found |
| `EXTERNAL_SERVICE_ERROR` | 502 | Error from OAuth provider or external API |

---

## Rate Limits

No rate limiting is currently implemented. The service relies on external API rate limits (Spotify, Last.fm, etc.).

---

## Authentication Flow Diagrams

### Spotify OAuth Flow

```
User                    Client                  API                     Spotify
  |                        |                      |                        |
  |--- Clicks Login ------>|                      |                        |
  |                        |-- GET /auth/start -->|                        |
  |                        |<-- auth_url, state --|                        |
  |<-- Redirect to Spotify ------------------------|                        |
  |                        |                      |                        |
  |--- Authorize App -------------------------------------------------->|
  |<-- Redirect with code -------------------------------------------------|
  |                        |                      |                        |
  |--- Code + State ------>|                      |                        |
  |                        |-- GET /auth/callback |                        |
  |                        |   (code, state)      |                        |
  |                        |                      |-- Exchange code ------>|
  |                        |                      |<-- Tokens -------------|
  |                        |<-- token_id ---------|                        |
  |<-- Success ------------|                      |                        |
```

### Last.fm Auth Flow

```
User                    Client                  API                     Last.fm
  |                        |                      |                        |
  |--- Clicks Login ------>|                      |                        |
  |                        |-- GET /auth/start -->|                        |
  |                        |<-- auth_url ---------|                        |
  |<-- Redirect to Last.fm ----------------------|                        |
  |                        |                      |                        |
  |--- Authorize App -------------------------------------------------->|
  |<-- Redirect with token -------------------------------------------------|
  |                        |                      |                        |
  |--- Token ------------->|                      |                        |
  |                        |-- GET /auth/callback |                        |
  |                        |   (token)            |                        |
  |                        |                      |-- Get session -------->|
  |                        |                      |<-- Session key --------|
  |                        |<-- session_key ------|                        |
  |<-- Success ------------|                      |                        |
```

---

## Job Processing

Jobs are processed asynchronously using Redis Queue (RQ):

1. Client calls `POST /v1/wrapped/generate`
2. API enqueues job to `MUSIC_WRAPPED_QUEUE`
3. Worker picks up job, fetches listening history from service API
4. Worker aggregates data, calculates top artists/songs
5. Worker stores result in Redis with TTL
6. Client polls `GET /v1/wrapped/status/{job_id}`
7. When complete, client fetches result via `GET /v1/wrapped/result/{result_id}`

**Job Timeout:** 600 seconds (10 minutes)
**Result TTL:** 86400 seconds (24 hours)
