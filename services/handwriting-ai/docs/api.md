# Handwriting AI - API Reference

Complete API documentation for the handwriting-ai digit recognition service.

**Base URL:** `http://localhost:8081` (default)

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

Readiness probe. Returns 503 if model not loaded.

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
  "reason": "model not loaded"
}
```

---

## Model Endpoints

### GET /v1/models/active

Get metadata for the currently loaded model.

**Response (200):**
```json
{
  "model_loaded": true,
  "model_id": "mnist_resnet18_v1",
  "arch": "resnet18",
  "n_classes": 10,
  "version": "1.0.0",
  "created_at": "2024-01-15T10:30:00Z",
  "schema_version": "v1.1",
  "val_acc": 0.9912,
  "temperature": 1.0
}
```

**Response (model not loaded):**
```json
{
  "model_loaded": false,
  "model_id": null
}
```

---

## Inference Endpoints

### POST /v1/read

Classify a handwritten digit image.

**Alias:** `POST /v1/predict`

**Content-Type:** `multipart/form-data`

**Request Fields:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `file` | file | Yes | - | PNG or JPEG image of handwritten digit |
| `invert` | bool | No | `null` | Force inversion (`null` = auto-detect background) |
| `center` | bool | No | `true` | Center digit on canvas before inference |
| `visualize` | bool | No | `false` | Return preprocessed image as base64 PNG |

**Request Headers:**

| Header | Required | Description |
|--------|----------|-------------|
| `Content-Type` | Yes | Must be `multipart/form-data` |
| `X-Api-Key` | Conditional | Required if `SECURITY__API_KEY_ENABLED=true` |
| `X-Request-ID` | No | Correlation ID (auto-generated if omitted) |
| `Content-Length` | No | Used for early size limit rejection |

**Response (200):**
```json
{
  "digit": 7,
  "confidence": 0.987,
  "probs": [0.001, 0.002, 0.001, 0.001, 0.001, 0.002, 0.003, 0.987, 0.001, 0.001],
  "model_id": "mnist_resnet18_v1",
  "visual_png_b64": null,
  "uncertain": false,
  "latency_ms": 15
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `digit` | int | Predicted digit (0-9) |
| `confidence` | float | Confidence score (0.0-1.0) |
| `probs` | float[] | Probability distribution over all 10 classes |
| `model_id` | string | ID of model used for prediction |
| `visual_png_b64` | string\|null | Base64-encoded preprocessed image (if `visualize=true`) |
| `uncertain` | bool | `true` if confidence below `DIGITS__UNCERTAIN_THRESHOLD` |
| `latency_ms` | int | Total processing time in milliseconds |

**Example - curl:**
```bash
curl -X POST http://localhost:8081/v1/read \
  -H "X-Api-Key: your-api-key" \
  -F "file=@digit.png;type=image/png" \
  -F "center=true" \
  -F "visualize=false"
```

**Example - Python:**
```python
import httpx

with open("digit.png", "rb") as f:
    response = httpx.post(
        "http://localhost:8081/v1/read",
        headers={"X-Api-Key": "your-api-key"},
        files={"file": ("digit.png", f, "image/png")},
        data={"center": "true", "visualize": "false"},
    )

response.raise_for_status()
result = response.json()
print(f"Digit: {result['digit']} ({result['confidence']:.1%} confidence)")
```

**Example - with visualization:**
```python
import base64
import httpx

response = httpx.post(
    "http://localhost:8081/v1/read",
    files={"file": ("digit.png", open("digit.png", "rb"), "image/png")},
    data={"visualize": "true"},
)

result = response.json()
if result["visual_png_b64"]:
    png_bytes = base64.b64decode(result["visual_png_b64"])
    with open("preprocessed.png", "wb") as f:
        f.write(png_bytes)
```

---

## Admin Endpoints

### POST /v1/admin/models/upload

Upload a new model to the service.

**Content-Type:** `multipart/form-data`

**Request Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model_id` | string | Yes | Unique model identifier |
| `activate` | bool | No | Activate model immediately after upload (default: `false`) |
| `manifest` | file | Yes | JSON manifest file |
| `model` | file | Yes | PyTorch state dict file (.pt) |

**Request Headers:**

| Header | Required | Description |
|--------|----------|-------------|
| `X-Api-Key` | Conditional | Required if `SECURITY__API_KEY_ENABLED=true` |

**Response (200):**
```json
{
  "ok": true,
  "model_id": "mnist_resnet18_v2",
  "run_id": "2024-01-15T10:30:00+00:00"
}
```

**Example:**
```bash
curl -X POST http://localhost:8081/v1/admin/models/upload \
  -H "X-Api-Key: your-api-key" \
  -F "model_id=mnist_resnet18_v2" \
  -F "activate=true" \
  -F "manifest=@manifest.json;type=application/json" \
  -F "model=@model.pt;type=application/octet-stream"
```

**Validation:**
- Manifest `preprocess_hash` must match current pipeline signature
- Manifest `model_id` must match request `model_id`
- If `activate=true`: model weights are validated before activation
- If `activate=false`: only basic transport validation (non-empty file)

---

## Training Endpoints

### POST /api/v1/training/jobs

Enqueue a new model training job for background processing.

**Content-Type:** `application/json`

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `user_id` | int | Yes | - | User ID for tracking |
| `model_id` | string | Yes | - | Unique model identifier |
| `epochs` | int | Yes | - | Number of training epochs |
| `batch_size` | int | Yes | - | Training batch size |
| `lr` | float | Yes | - | Learning rate |
| `seed` | int | Yes | - | Random seed for reproducibility |
| `augment` | bool | No | `false` | Enable data augmentation |
| `notes` | string | No | `null` | Optional notes about this training run |

**Request Example:**
```json
{
  "user_id": 12345,
  "model_id": "mnist_resnet18_v3",
  "epochs": 10,
  "batch_size": 64,
  "lr": 0.001,
  "seed": 42,
  "augment": true,
  "notes": "Experiment with augmentation"
}
```

**Response (202):**
```json
{
  "job_id": "rq-job-uuid",
  "request_id": "uuid-for-tracking",
  "user_id": 12345,
  "model_id": "mnist_resnet18_v3",
  "status": "queued"
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `job_id` | string | RQ job identifier |
| `request_id` | string | Request tracking UUID |
| `user_id` | int | User ID from request |
| `model_id` | string | Model identifier from request |
| `status` | string | Always `queued` on successful submission |

**Request Headers:**

| Header | Required | Description |
|--------|----------|-------------|
| `X-Api-Key` | Conditional | Required if `SECURITY__API_KEY_ENABLED=true` |

**Example - curl:**
```bash
curl -X POST http://localhost:8081/api/v1/training/jobs \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: your-api-key" \
  -d '{
    "user_id": 12345,
    "model_id": "mnist_resnet18_v3",
    "epochs": 10,
    "batch_size": 64,
    "lr": 0.001,
    "seed": 42,
    "augment": true
  }'
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
| `invalid_image` | 400 | Failed to decode image or unsupported format |
| `bad_dimensions` | 400 | Image exceeds maximum dimension limit |
| `malformed_multipart` | 400 | Invalid multipart structure (extra fields, missing file) |
| `preprocessing_failed` | 400 | Preprocessing pipeline error or manifest mismatch |
| `invalid_model` | 400 | Corrupt model weights or validation failure |
| `UNAUTHORIZED` | 401 | Missing API key when required |
| `FORBIDDEN` | 403 | Invalid API key |
| `PAYLOAD_TOO_LARGE` | 413 | File exceeds `DIGITS__MAX_IMAGE_MB` limit |
| `UNSUPPORTED_MEDIA_TYPE` | 415 | Not PNG or JPEG |
| `TIMEOUT` | 408 | Prediction exceeded `DIGITS__PREDICT_TIMEOUT_SECONDS` |
| `SERVICE_UNAVAILABLE` | 503 | Model not loaded |

### Error Examples

**Invalid image format:**
```json
{
  "code": "invalid_image",
  "message": "Failed to decode image",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Missing API key:**
```json
{
  "code": "UNAUTHORIZED",
  "message": "Missing or invalid API key",
  "request_id": "550e8400-e29b-41d4-a716-446655440001"
}
```

**Model not loaded:**
```json
{
  "code": "SERVICE_UNAVAILABLE",
  "message": "Model not loaded. Upload or train a model.",
  "request_id": "550e8400-e29b-41d4-a716-446655440002"
}
```

---

## Authentication

API key authentication is optional and controlled by environment variables.

**Enable authentication:**
```bash
SECURITY__API_KEY_ENABLED=true
SECURITY__API_KEY=your-secret-key
```

**Send API key:**
```bash
curl -H "X-Api-Key: your-secret-key" http://localhost:8081/v1/read ...
```

**Protected endpoints:**
- `POST /v1/read`
- `POST /v1/predict`
- `POST /v1/admin/models/upload`

**Unprotected endpoints:**
- `GET /healthz`
- `GET /readyz`
- `GET /v1/models/active`

---

## Configuration

| Constraint | Default | Environment Variable |
|------------|---------|---------------------|
| Max image size | 2 MB | `DIGITS__MAX_IMAGE_MB` |
| Max image dimension | 1024 px | `DIGITS__MAX_IMAGE_SIDE_PX` |
| Prediction timeout | 5 seconds | `DIGITS__PREDICT_TIMEOUT_SECONDS` |
| Uncertainty threshold | 0.85 | `DIGITS__UNCERTAIN_THRESHOLD` |

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

**Supported image types:**
- `image/png`
- `image/jpeg`
- `image/jpg`

**Request content type:**
- `multipart/form-data` (for file uploads)

**Response content type:**
- `application/json`
