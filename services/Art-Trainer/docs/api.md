# Art-Trainer - API Reference

Complete API documentation for the art-trainer service.

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
  "status": "ready",
  "reason": null
}
```

**Response (503 - Redis unavailable):**
```json
{
  "status": "degraded",
  "reason": "redis-unavailable"
}
```

**Response (503 - No workers):**
```json
{
  "status": "degraded",
  "reason": "no-worker"
}
```

---

## LoRA Training Endpoints

### POST /lora/train

Enqueue a LoRA training job.

**Request Headers:**

| Header | Required | Description |
|--------|----------|-------------|
| `Content-Type` | Yes | Must be `application/json` |
| `X-Api-Key` | Conditional | Required if `SECURITY__API_KEY` is set |

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_id` | int | Yes | User initiating the training |
| `base_model` | string | Yes | Base model: `sd15`, `sdxl`, or `flux` |
| `training_type` | string | Yes | Training type: `style`, `character`, or `concept` |
| `dataset_file_id` | string | Yes | File ID for the dataset in data-bank |
| `steps` | int | Yes | Number of training steps |
| `learning_rate` | float | Yes | Learning rate for training |
| `network_rank` | int | Yes | LoRA network rank |
| `network_alpha` | int | Yes | LoRA network alpha |
| `resolution` | int | Yes | Training image resolution |
| `batch_size` | int | Yes | Training batch size |
| `seed` | int | Yes | Random seed for reproducibility |
| `caption_extension` | string | Yes | File extension for captions (e.g., `.txt`) |
| `shuffle_caption` | bool | Yes | Whether to shuffle caption tokens |
| `keep_tokens` | int | Yes | Number of tokens to keep unshuffled |

**Request Example:**
```json
{
  "user_id": 12345,
  "base_model": "sdxl",
  "training_type": "character",
  "dataset_file_id": "abc123-def456",
  "steps": 1500,
  "learning_rate": 0.0001,
  "network_rank": 32,
  "network_alpha": 16,
  "resolution": 1024,
  "batch_size": 2,
  "seed": 42,
  "caption_extension": ".txt",
  "shuffle_caption": true,
  "keep_tokens": 1
}
```

**Response (200):**
```json
{
  "job_id": "lora-job-uuid"
}
```

---

### GET /lora/{job_id}

Get the status of a training job.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | string | Unique job identifier |

**Response (200):**
```json
{
  "job_id": "lora-job-uuid",
  "status": "running",
  "message": "Training in progress",
  "lora_file_id": null,
  "lora_name": null
}
```

**Status Values:**
- `queued` - Job waiting to be processed
- `running` - Training in progress
- `completed` - Training finished successfully
- `failed` - Training encountered an error
- `cancelled` - Training was cancelled

**Completed Response:**
```json
{
  "job_id": "lora-job-uuid",
  "status": "completed",
  "message": null,
  "lora_file_id": "databank-file-id",
  "lora_name": "my-lora-v1.safetensors"
}
```

---

### GET /lora/{job_id}/progress

Get detailed progress of a training job.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | string | Unique job identifier |

**Response (200):**
```json
{
  "job_id": "lora-job-uuid",
  "phase": "training",
  "step": 750,
  "total_steps": 1500,
  "loss": 0.0234,
  "learning_rate": 0.0001,
  "updated_at": "2024-01-15T10:30:00Z",
  "lora_file_id": null,
  "lora_name": null
}
```

**Phase Values:**
- `queued` - Job waiting to be processed
- `preparing` - Preparing dataset and model
- `training` - Training in progress
- `saving` - Saving model weights
- `uploading` - Uploading to data-bank
- `completed` - Training finished successfully
- `failed` - Training encountered an error
- `cancelled` - Training was cancelled

---

### POST /lora/{job_id}/cancel

Request cancellation of a training job.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `job_id` | string | Unique job identifier |

**Response (200):**
```json
{
  "status": "cancellation-requested"
}
```

---

## Dataset Endpoints

### POST /dataset/upload

Upload images for a training dataset with optional auto-captioning.

**Request:** `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `files` | File[] | Yes | Image files to upload (PNG, JPG, JPEG, WEBP, GIF) |
| `trigger_word` | string | Yes | Trigger word for captions (e.g., `sks person`) |
| `training_type` | string | Yes | Training type: `style`, `character`, or `concept` |
| `auto_caption` | bool | Yes | Whether to auto-generate captions using BLIP |

**Response (200):**
```json
{
  "dataset_id": "dataset-uuid",
  "image_count": 20,
  "caption_count": 20,
  "dataset_path": "/data/datasets/dataset-uuid"
}
```

---

### GET /dataset/{dataset_id}

Get information about a dataset.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `dataset_id` | string | Dataset identifier |

**Response (200):**
```json
{
  "dataset_id": "dataset-uuid",
  "image_count": 20,
  "caption_count": 20,
  "dataset_path": "/data/datasets/dataset-uuid"
}
```

**Response (404):**
```json
{
  "detail": "Dataset dataset-uuid not found"
}
```

---

### POST /dataset/{dataset_id}/caption

Caption images in a dataset using a specified backend.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `dataset_id` | string | Dataset identifier |

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `trigger_word` | string | Yes | Trigger word to prepend to captions |
| `backend` | string | Yes | Caption backend: `blip`, `gemini`, or `openai` |
| `model_name` | string | Yes | Model name for the backend |

**Backend Model Names:**
- `blip`: `Salesforce/blip-image-captioning-base` or `Salesforce/blip-image-captioning-large`
- `gemini`: `gemini-2.0-flash`, `gemini-2.5-flash`, etc.
- `openai`: `gpt-4o`, `gpt-4o-mini`, etc.

**Request Example:**
```json
{
  "trigger_word": "sks person",
  "backend": "gemini",
  "model_name": "gemini-2.0-flash"
}
```

**Response (200):**
```json
{
  "dataset_id": "dataset-uuid",
  "captioned_count": 15,
  "skipped_count": 5,
  "backend": "gemini"
}
```

**Response (404):**
```json
{
  "detail": "Dataset dataset-uuid not found"
}
```

---

## Error Responses

All endpoints may return standard error responses:

**400 - Bad Request:**
```json
{
  "detail": "Field 'base_model' must be 'sd15', 'sdxl', or 'flux', got 'invalid'"
}
```

**401 - Unauthorized:**
```json
{
  "detail": "Invalid API key"
}
```

**500 - Internal Server Error:**
```json
{
  "detail": "Internal server error"
}
```
