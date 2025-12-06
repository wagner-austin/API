# Data Bank API - API Reference

Complete API documentation for the data-bank-api file storage service.

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

Readiness probe. Checks storage writability and disk space.

**Response (200):**
```json
{
  "status": "ready"
}
```

**Response (503):**
```json
{
  "status": "degraded",
  "reason": "storage not writable | low disk"
}
```

---

## File Endpoints

### POST /files

Upload a file. Returns SHA256-based file ID.

**Content-Type:** `multipart/form-data`

**Request Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | Binary file content |

**Request Headers:**

| Header | Required | Description |
|--------|----------|-------------|
| `Content-Type` | Yes | Must be `multipart/form-data` |
| `X-API-Key` | Yes | Upload permission required |
| `X-Request-ID` | No | Correlation ID (auto-generated if omitted) |

**Response (201):**
```json
{
  "file_id": "a1b2c3d4e5f6...",
  "size": 1024,
  "sha256": "a1b2c3d4e5f6...",
  "content_type": "text/plain; charset=utf-8",
  "created_at": "2024-11-27T10:30:00+00:00"
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `file_id` | string | SHA256 hex digest (content-addressed ID) |
| `size` | int | File size in bytes |
| `sha256` | string | SHA256 hash of content |
| `content_type` | string | Detected or provided MIME type |
| `created_at` | string | ISO 8601 timestamp |

**Example - curl:**
```bash
curl -X POST http://localhost:8000/files \
  -H "X-API-Key: your-upload-key" \
  -F "file=@corpus.txt;type=text/plain"
```

**Example - Python:**
```python
import httpx

with open("corpus.txt", "rb") as f:
    response = httpx.post(
        "http://localhost:8000/files",
        headers={"X-API-Key": "your-upload-key"},
        files={"file": ("corpus.txt", f, "text/plain")},
    )

response.raise_for_status()
result = response.json()
print(f"Uploaded: {result['file_id']} ({result['size']} bytes)")
```

---

### GET /files/{file_id}

Download a file. Supports HTTP Range requests for partial content.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `file_id` | string | SHA256 file identifier |

**Request Headers:**

| Header | Required | Description |
|--------|----------|-------------|
| `X-API-Key` | Yes | Read permission required |
| `Range` | No | Byte range for partial content |

**Response (200):** Full file content

**Response (206):** Partial content (with Range header)

**Response Headers:**
```
Content-Range: bytes 0-99/1000
Content-Length: 100
ETag: a1b2c3d4...
Accept-Ranges: bytes
```

**Example - full download:**
```bash
curl http://localhost:8000/files/a1b2c3d4... \
  -H "X-API-Key: your-read-key" \
  -o downloaded.txt
```

**Example - range request:**
```bash
# First 100 bytes
curl http://localhost:8000/files/a1b2c3d4... \
  -H "X-API-Key: your-read-key" \
  -H "Range: bytes=0-99"
```

**Range Formats:**
```
Range: bytes=0-99      # First 100 bytes
Range: bytes=100-199   # Bytes 100-199
Range: bytes=100-      # From byte 100 to end
Range: bytes=-50       # Last 50 bytes
```

---

### HEAD /files/{file_id}

Probe file metadata without downloading content.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `file_id` | string | SHA256 file identifier |

**Request Headers:**

| Header | Required | Description |
|--------|----------|-------------|
| `X-API-Key` | Yes | Read permission required |

**Response (200) Headers:**
```
Content-Length: 1024
Content-Type: text/plain
ETag: a1b2c3d4...
Accept-Ranges: bytes
```

---

### GET /files/{file_id}/info

Get file metadata as JSON.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `file_id` | string | SHA256 file identifier |

**Request Headers:**

| Header | Required | Description |
|--------|----------|-------------|
| `X-API-Key` | Yes | Read permission required |

**Response (200):**
```json
{
  "file_id": "a1b2c3d4e5f6...",
  "size": 1024,
  "sha256": "a1b2c3d4e5f6...",
  "content_type": "text/plain",
  "created_at": "2024-11-27T10:30:00+00:00"
}
```

---

### DELETE /files/{file_id}

Delete a file. Idempotent by default.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `file_id` | string | SHA256 file identifier |

**Request Headers:**

| Header | Required | Description |
|--------|----------|-------------|
| `X-API-Key` | Yes | Delete permission required |

**Response (204):** No content

**Note:** Returns 204 even if file doesn't exist, unless `DELETE_STRICT_404=true`.

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
| `UNAUTHORIZED` | 401 | Missing API key |
| `FORBIDDEN` | 403 | Invalid API key or insufficient permission |
| `NOT_FOUND` | 404 | File doesn't exist |
| `INVALID_INPUT` | 400 | Bad file_id format or malformed multipart |
| `RANGE_NOT_SATISFIABLE` | 416 | Invalid byte range |
| `PAYLOAD_TOO_LARGE` | 413 | File exceeds `MAX_FILE_BYTES` |
| `INSUFFICIENT_STORAGE` | 507 | Disk space below `MIN_FREE_GB` |

### Error Examples

**Missing API key:**
```json
{
  "code": "UNAUTHORIZED",
  "message": "Missing or invalid API key",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**File not found:**
```json
{
  "code": "NOT_FOUND",
  "message": "File not found",
  "request_id": "550e8400-e29b-41d4-a716-446655440001"
}
```

**Disk full:**
```json
{
  "code": "INSUFFICIENT_STORAGE",
  "message": "Insufficient disk space",
  "request_id": "550e8400-e29b-41d4-a716-446655440002"
}
```

---

## Authentication

The service uses header-based authentication with separate permission scopes.

**Send API key:**
```bash
curl -H "X-API-Key: your-secret-key" http://localhost:8000/files/...
```

### Permission Scopes

| Scope | Endpoints | Use Case |
|-------|-----------|----------|
| `upload` | `POST /files` | Producer services |
| `read` | `GET`, `HEAD`, `/info` | Consumer services |
| `delete` | `DELETE /files/{id}` | Admin operations |

### Key Configuration

```bash
API_UPLOAD_KEYS=producer-key-1,producer-key-2
API_READ_KEYS=consumer-key-1,consumer-key-2
API_DELETE_KEYS=admin-key
```

### Key Inheritance

If `API_READ_KEYS` or `API_DELETE_KEYS` are empty, they inherit from `API_UPLOAD_KEYS`.

### Protected Endpoints

All file endpoints require authentication:
- `POST /files`
- `GET /files/{file_id}`
- `HEAD /files/{file_id}`
- `GET /files/{file_id}/info`
- `DELETE /files/{file_id}`

### Unprotected Endpoints

- `GET /healthz`
- `GET /readyz`

---

## Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `PORT` | int | `8000` | Server port |
| `DATA_ROOT` | string | `/data/files` | Storage root directory |
| `MIN_FREE_GB` | int | `1` | Minimum free disk space (GB) |
| `MAX_FILE_BYTES` | int | `0` | Max upload size (0 = unlimited) |
| `DELETE_STRICT_404` | bool | `false` | Return 404 on missing delete |
| `API_UPLOAD_KEYS` | string | - | Comma-separated upload keys |
| `API_READ_KEYS` | string | - | Comma-separated read keys |
| `API_DELETE_KEYS` | string | - | Comma-separated delete keys |

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

**Request content types:**
- `multipart/form-data` (for file uploads)

**Response content types:**
- `application/json` (metadata, errors)
- `application/octet-stream` or detected type (file downloads)
