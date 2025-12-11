# QR API - API Reference

Complete API documentation for the qr-api QR code generation service.

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

Readiness probe. Returns 503 if Redis unavailable or no workers registered.

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
  "reason": "redis-unavailable"
}
```

---

## QR Code Endpoints

### POST /v1/qr

Generate a QR code PNG from a URL.

**Content-Type:** `application/json`

**Request Fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `url` | string | Yes | - | URL to encode (max 2000 chars) |
| `ecc` | string | No | `"M"` | Error correction level: `L`, `M`, `Q`, `H` |
| `box_size` | int | No | `10` | Pixels per module (5-20) |
| `border` | int | No | `1` | Quiet zone width in modules (1-10) |
| `fill_color` | string | No | `"#000000"` | Dark module color (hex) |
| `back_color` | string | No | `"#FFFFFF"` | Light module color (hex) |

**Request Headers:**

| Header | Required | Description |
|--------|----------|-------------|
| `Content-Type` | Yes | Must be `application/json` |
| `X-Request-ID` | No | Correlation ID (auto-generated if omitted) |

**Response (200):**
- **Content-Type:** `image/png`
- **Body:** PNG image bytes

**Error Correction Levels:**

| Level | Recovery | Use Case |
|-------|----------|----------|
| `L` | ~7% | Maximum data capacity |
| `M` | ~15% | Default, balanced |
| `Q` | ~25% | Better durability |
| `H` | ~30% | Maximum error tolerance |

**Example - curl:**
```bash
curl -X POST http://localhost:8000/v1/qr \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://example.com",
    "ecc": "H",
    "box_size": 10,
    "border": 2,
    "fill_color": "#000000",
    "back_color": "#FFFFFF"
  }' \
  --output qr.png
```

**Example - minimal request:**
```bash
curl -X POST http://localhost:8000/v1/qr \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}' \
  --output qr.png
```

**Example - Python:**
```python
import httpx

response = httpx.post(
    "http://localhost:8000/v1/qr",
    json={
        "url": "https://example.com",
        "ecc": "M",
        "box_size": 10,
        "fill_color": "#1a1a2e",
        "back_color": "#eaeaea",
    },
)
response.raise_for_status()

with open("qr.png", "wb") as f:
    f.write(response.content)
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
| `INVALID_INPUT` | 400 | Field 'url' is required |
| `INVALID_INPUT` | 400 | Please provide a URL |
| `INVALID_INPUT` | 400 | URL is too long (max 2000 characters) |
| `INVALID_INPUT` | 400 | URL scheme must be http or https |
| `INVALID_INPUT` | 400 | Please check the URL and try again |
| `INVALID_INPUT` | 400 | Invalid color format. Use hex codes (e.g., #FF0000 or #F00) |
| `INVALID_INPUT` | 400 | Invalid error correction. Choose L, M, Q, H |
| `INVALID_INPUT` | 400 | box_size must be between 5 and 20 |
| `INVALID_INPUT` | 400 | border must be between 1 and 10 |
| `INVALID_INPUT` | 400 | Invalid JSON body |

### Error Examples

**Missing URL:**
```json
{
  "code": "INVALID_INPUT",
  "message": "Field 'url' is required",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Invalid scheme:**
```json
{
  "code": "INVALID_INPUT",
  "message": "URL scheme must be http or https",
  "request_id": "550e8400-e29b-41d4-a716-446655440001"
}
```

**Malformed JSON:**
```json
{
  "code": "INVALID_INPUT",
  "message": "Invalid JSON body",
  "request_id": "550e8400-e29b-41d4-a716-446655440002"
}
```

---

## Configuration

Default values can be configured via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Server port |
| `REDIS_URL` | - | Redis connection URL (required for /readyz) |
| `QR_DEFAULT_ERROR_CORRECTION` | `"M"` | Default ECC level (L/M/Q/H) |
| `QR_DEFAULT_BOX_SIZE` | `10` | Default pixels per module (5-20) |
| `QR_DEFAULT_BORDER` | `1` | Default quiet zone width (1-10) |
| `QR_DEFAULT_FILL_COLOR` | `"#000000"` | Default dark color |
| `QR_DEFAULT_BACK_COLOR` | `"#FFFFFF"` | Default light color |

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
- `application/json`

**Response content types:**
- `image/png` (success)
- `application/json` (errors)

---

## URL Validation

The service validates URLs with strict rules:

1. **Length**: Maximum 2000 characters
2. **Scheme**: Only `http://` and `https://` allowed
   - URLs without a scheme get `https://` prepended automatically
3. **Host**: Must be one of:
   - Valid domain name (e.g., `example.com`)
   - `localhost`
   - IPv4 address (e.g., `192.168.1.1`)
   - IPv6 address in brackets (e.g., `[::1]`)
4. **Fragments**: URL fragments (`#section`) are stripped

**Valid URLs:**
```
https://example.com
http://localhost:3000/path
example.com (becomes https://example.com)
https://192.168.1.1/api
https://[::1]:8080/test
```

**Invalid URLs:**
```
ftp://example.com (wrong scheme)
javascript:alert(1) (wrong scheme)
not a url at all (no valid host)
```
