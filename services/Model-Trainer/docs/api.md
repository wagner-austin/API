# Model Trainer - API Reference

Complete API documentation for the model-trainer service.

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

## Training Run Endpoints

### POST /runs/train

Enqueue a model training job.

**Request Headers:**

| Header | Required | Description |
|--------|----------|-------------|
| `Content-Type` | Yes | Must be `application/json` |
| `X-Api-Key` | Conditional | Required if `SECURITY__API_KEY` is set |
| `X-Request-ID` | No | Correlation ID (auto-generated if omitted) |

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model_family` | string | No | `"gpt2"` | Model architecture (`gpt2`, `llama`, `qwen`, `char_lstm`) |
| `model_size` | string | No | `"small"` | Model size variant |
| `max_seq_len` | int | No | `512` | Maximum sequence length (>= 8) |
| `num_epochs` | int | No | `1` | Number of training epochs (>= 1) |
| `batch_size` | int | No | `4` | Training batch size (>= 1); see device defaults below |
| `learning_rate` | float | No | `0.0005` | Learning rate (>= 0.0) |
| `corpus_file_id` | string | Yes | - | File ID in data-bank for training corpus |
| `tokenizer_id` | string | No | `""` | Tokenizer ID to use |
| `user_id` | int | No | `0` | User ID for event attribution |
| `device` | string | No | `"auto"` | `"cpu"`, `"cuda"`, or `"auto"` (resolves to CUDA when available) |
| `precision` | string | No | `"auto"` | `"fp32"`, `"fp16"`, `"bf16"`, or `"auto"` (resolves based on device) |
| `data_num_workers` | int\|null | No | `null` | Optional DataLoader workers; default depends on device |
| `data_pin_memory` | bool\|null | No | `null` | Optional DataLoader pin_memory; default depends on device |

**Request Example:**
```json
{
  "model_family": "gpt2",
  "model_size": "small",
  "max_seq_len": 128,
  "num_epochs": 3,
  "batch_size": 8,
  "learning_rate": 0.0001,
  "corpus_file_id": "abc123",
  "tokenizer_id": "my-bpe-tokenizer",
  "user_id": 12345
}
```

**Response (200):**
```json
{
  "run_id": "gpt2-run-001",
  "job_id": "rq-job-uuid"
}
```

Device, Precision, and DataLoader defaults
- Device `"auto"` resolves once at job start to `"cuda"` when available; otherwise `"cpu"`.
- Precision `"auto"` resolves based on device:
  - CUDA: defaults to `"fp16"` (mixed precision with GradScaler for stability)
  - CPU: defaults to `"fp32"` (mixed precision not beneficial on CPU)
- Explicit precision values:
  - `"fp32"`: Full precision, works on any device
  - `"fp16"`: Mixed precision with loss scaling (CUDA only, raises error on CPU)
  - `"bf16"`: Mixed precision without scaling (CUDA only, raises error on CPU)
- Batch sizing (when you supply very small batches like 1–4):
  - gpt2 on CUDA defaults to 32
  - char_lstm on CUDA defaults to 64
  - other families on CUDA default to 16
  - CPU keeps your provided batch size unchanged
- DataLoader defaults when omitted:
  - CUDA: `data_num_workers = min(4, os.cpu_count())`, `data_pin_memory = true`
  - CPU: `data_num_workers = 0`, `data_pin_memory = false`

**Example - curl:**
```bash
curl -X POST http://localhost:8000/runs/train \
  -H "Content-Type: application/json" \
  -H "X-Api-Key: your-api-key" \
  -d '{
    "model_family": "gpt2",
    "corpus_file_id": "abc123",
    "num_epochs": 3
  }'
```

---

### GET /runs/{run_id}

Get training run status and heartbeat information.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `run_id` | string | Training run identifier |

**Response (200):**
```json
{
  "run_id": "gpt2-run-001",
  "status": "running",
  "last_heartbeat_ts": 1732680600.123,
  "message": "Training epoch 2/3"
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `run_id` | string | Training run identifier |
| `status` | string | One of: `queued`, `running`, `completed`, `failed` |
| `last_heartbeat_ts` | float\|null | Unix timestamp of last heartbeat |
| `message` | string\|null | Status message from worker |

---

### POST /runs/{run_id}/evaluate

Enqueue evaluation job for a completed training run.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `run_id` | string | Training run identifier |

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `split` | string | No | `"validation"` | Dataset split (`validation` or `test`) |
| `path_override` | string | No | null | Override path for evaluation data |

**Request Example:**
```json
{
  "split": "validation"
}
```

**Response (200):**
```json
{
  "run_id": "gpt2-run-001",
  "split": "validation",
  "status": "queued",
  "loss": null,
  "perplexity": null,
  "artifact_path": null
}
```

---

### GET /runs/{run_id}/eval

Get evaluation results for a training run.

**Response (200):**
```json
{
  "run_id": "gpt2-run-001",
  "split": "validation",
  "status": "completed",
  "loss": 3.21,
  "perplexity": 24.8,
  "artifact_path": "/data/artifacts/models/gpt2-run-001/eval"
}
```

---

### GET /runs/{run_id}/artifact

Get artifact pointer (storage location and file ID) for a completed run.

**Response (200):**
```json
{
  "storage": "data-bank",
  "file_id": "model-artifact-uuid"
}
```

---

### GET /runs/{run_id}/logs

Get training logs (tail).

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tail` | int | `200` | Number of lines to return from end |

**Response (200):** `text/plain`
```
[2024-11-27 10:30:00] Epoch 1/3, Step 100, Loss: 4.52
[2024-11-27 10:30:05] Epoch 1/3, Step 200, Loss: 4.21
...
```

**Response (404):**
```json
{
  "code": "NOT_FOUND",
  "message": "logs not found",
  "request_id": "uuid"
}
```

---

### GET /runs/{run_id}/logs/stream

Stream training logs via Server-Sent Events (SSE).

**Query Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tail` | int | `200` | Number of initial lines to return |
| `follow` | bool | `true` | Continue streaming new lines |

**Response:** `text/event-stream`
```
data: [2024-11-27 10:30:00] Epoch 1/3, Step 100, Loss: 4.52

data: [2024-11-27 10:30:05] Epoch 1/3, Step 200, Loss: 4.21

```

---

### POST /runs/{run_id}/cancel

Request cancellation of a running training job.

**Response (200):**
```json
{
  "status": "cancellation-requested"
}
```

---

## Inference Endpoints

### POST /runs/{run_id}/score

Score text using a trained model (calculate perplexity/loss).

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `text` | string | Conditional | - | Text to score (mutually exclusive with `path`) |
| `path` | string | Conditional | - | File path to score (mutually exclusive with `text`) |
| `detail_level` | string | No | `"summary"` | `"summary"` or `"per_char"` |
| `top_k` | int | No | null | Top-k tokens to return per position |
| `seed` | int | No | null | Random seed |

**Request Example:**
```json
{
  "text": "Hello, world!",
  "detail_level": "summary"
}
```

**Response (200):**
```json
{
  "request_id": "uuid",
  "status": "queued",
  "loss": null,
  "ppl": null,
  "per_position": null
}
```

---

### GET /runs/{run_id}/score/{request_id}

Get scoring result.

**Response (200):**
```json
{
  "request_id": "uuid",
  "status": "completed",
  "loss": 2.34,
  "ppl": 10.4,
  "per_position": null
}
```

---

### POST /runs/{run_id}/generate

Generate text using a trained model.

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `prompt_text` | string | Conditional | - | Prompt text (mutually exclusive with `prompt_path`) |
| `prompt_path` | string | Conditional | - | File path for prompt (mutually exclusive with `prompt_text`) |
| `max_new_tokens` | int | No | `64` | Maximum tokens to generate (1-1024) |
| `temperature` | float | No | `1.0` | Sampling temperature (0.0-2.0) |
| `top_k` | int | No | `50` | Top-k sampling |
| `top_p` | float | No | `1.0` | Nucleus sampling threshold (0.0-1.0) |
| `seed` | int | No | null | Random seed |

**Request Example:**
```json
{
  "prompt_text": "Once upon a time",
  "max_new_tokens": 100,
  "temperature": 0.8
}
```

**Response (200):**
```json
{
  "request_id": "uuid",
  "status": "queued",
  "outputs": null,
  "steps": null,
  "eos_terminated": null
}
```

---

### GET /runs/{run_id}/generate/{request_id}

Get generation result.

**Response (200):**
```json
{
  "request_id": "uuid",
  "status": "completed",
  "outputs": ["Once upon a time there was a..."],
  "steps": 100,
  "eos_terminated": false
}
```

---

## Chat Endpoints

Interactive chat with conversation memory. Sessions persist in Redis with a 1-hour TTL.

### POST /runs/{run_id}/chat

Start or continue a chat conversation.

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `message` | string | Yes | - | User message |
| `session_id` | string | No | null | Session ID (auto-generated if omitted) |
| `max_new_tokens` | int | No | `128` | Maximum tokens to generate (1-1024) |
| `temperature` | float | No | `0.8` | Sampling temperature (0.0-2.0) |
| `top_k` | int | No | `50` | Top-k sampling |
| `top_p` | float | No | `0.95` | Nucleus sampling threshold (0.0-1.0) |

**Request Example (new session):**
```json
{
  "message": "Hello, how are you?"
}
```

**Request Example (continue session):**
```json
{
  "message": "Tell me more about that.",
  "session_id": "abc123-uuid"
}
```

**Response (200):**
```json
{
  "session_id": "abc123-uuid",
  "request_id": "req-456-uuid",
  "status": "queued",
  "response": null
}
```

---

### GET /runs/{run_id}/chat/{session_id}/{request_id}

Poll for chat response.

**Response (200 - completed):**
```json
{
  "session_id": "abc123-uuid",
  "request_id": "req-456-uuid",
  "status": "completed",
  "response": "I'm doing well, thank you for asking!"
}
```

**Response (200 - still processing):**
```json
{
  "session_id": "abc123-uuid",
  "request_id": "req-456-uuid",
  "status": "running",
  "response": null
}
```

---

### GET /runs/{run_id}/chat/{session_id}

Get conversation history for a session.

**Response (200):**
```json
{
  "session_id": "abc123-uuid",
  "run_id": "gpt2-run-001",
  "messages": [
    {"role": "user", "content": "Hello, how are you?"},
    {"role": "assistant", "content": "I'm doing well, thank you for asking!"}
  ],
  "created_at": "2025-01-01T00:00:00Z"
}
```

---

### DELETE /runs/{run_id}/chat/{session_id}

Delete a chat session.

**Response (200):**
```json
{
  "status": "cancellation-requested"
}
```

---

## Tokenizer Endpoints

### POST /tokenizers/train

Enqueue a tokenizer training job.

**Request Headers:**

| Header | Required | Description |
|--------|----------|-------------|
| `Content-Type` | Yes | Must be `application/json` |
| `X-Api-Key` | Conditional | Required if `SECURITY__API_KEY` is set |

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `method` | string | Yes | Tokenizer method (`bpe` or `sentencepiece`) |
| `vocab_size` | int | Yes | Vocabulary size (>= 100) |
| `min_frequency` | int | Yes | Minimum token frequency (>= 1) |
| `corpus_file_id` | string | Yes | File ID in data-bank for training corpus |
| `holdout_fraction` | float | Yes | Fraction held out for validation (0.0-1.0) |
| `seed` | int | Yes | Random seed for reproducibility |

**Request Example:**
```json
{
  "method": "bpe",
  "vocab_size": 8000,
  "min_frequency": 2,
  "corpus_file_id": "corpus-abc123",
  "holdout_fraction": 0.1,
  "seed": 42
}
```

**Response (200):**
```json
{
  "tokenizer_id": "tok-uuid-abc123",
  "artifact_path": "/data/artifacts/tokenizers/tok-uuid-abc123",
  "coverage": null,
  "oov_rate": null
}
```

---

### GET /tokenizers/{tokenizer_id}

Get tokenizer status and statistics.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `tokenizer_id` | string | Tokenizer identifier |

**Response (200):**
```json
{
  "tokenizer_id": "my-bpe-tokenizer",
  "artifact_path": "/data/artifacts/tokenizers/my-bpe-tokenizer",
  "status": "completed",
  "coverage": 0.98,
  "oov_rate": 0.02,
  "token_count": 8000,
  "char_coverage": 0.9995
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `tokenizer_id` | string | Tokenizer identifier |
| `artifact_path` | string | Path to tokenizer artifacts |
| `status` | string | Training status (`queued`, `running`, `completed`, `failed`, `unknown`) |
| `coverage` | float\|null | Token coverage on holdout set |
| `oov_rate` | float\|null | Out-of-vocabulary rate |
| `token_count` | int\|null | Total tokens in vocabulary |
| `char_coverage` | float\|null | Character coverage |

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
| `INVALID_INPUT` | 422 | Invalid request body or parameters |
| `NOT_FOUND` | 404 | Resource not found |
| `UNAUTHORIZED` | 401 | Missing API key when required |
| `FORBIDDEN` | 403 | Invalid API key |
| `INTERNAL_ERROR` | 500 | Internal server error |

### Error Examples

**Invalid request body:**
```json
{
  "code": "INVALID_INPUT",
  "message": "Extra fields not allowed: unknown_field",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Missing required field:**
```json
{
  "code": "INVALID_INPUT",
  "message": "corpus_file_id is required",
  "request_id": "550e8400-e29b-41d4-a716-446655440001"
}
```

**Missing API key:**
```json
{
  "code": "UNAUTHORIZED",
  "message": "Missing or invalid API key",
  "request_id": "550e8400-e29b-41d4-a716-446655440002"
}
```

---

## Authentication

API key authentication is optional and controlled by environment variables.

**Enable authentication:**
```bash
SECURITY__API_KEY=your-secret-key
```

**Send API key:**
```bash
curl -H "X-Api-Key: your-secret-key" http://localhost:8000/runs/train ...
```

**Protected endpoints:**
- `POST /runs/train`
- `GET /runs/{run_id}`
- `POST /runs/{run_id}/evaluate`
- `GET /runs/{run_id}/eval`
- `GET /runs/{run_id}/artifact`
- `GET /runs/{run_id}/logs`
- `GET /runs/{run_id}/logs/stream`
- `POST /runs/{run_id}/cancel`
- `POST /runs/{run_id}/score`
- `GET /runs/{run_id}/score/{request_id}`
- `POST /runs/{run_id}/generate`
- `GET /runs/{run_id}/generate/{request_id}`
- `POST /runs/{run_id}/chat`
- `GET /runs/{run_id}/chat/{session_id}/{request_id}`
- `GET /runs/{run_id}/chat/{session_id}`
- `DELETE /runs/{run_id}/chat/{session_id}`
- `POST /tokenizers/train`
- `GET /tokenizers/{tokenizer_id}`

**Unprotected endpoints:**
- `GET /healthz`
- `GET /readyz`

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
- `application/json` (for all POST endpoints)

**Response content types:**
- `application/json` (default)
- `text/plain` (for `/runs/{run_id}/logs`)
- `text/event-stream` (for `/runs/{run_id}/logs/stream`)

---

## Event Publishing

Training publishes to Redis channel `trainer:events`:

- Job lifecycle (`platform_core.job_events`): `trainer.job.started.v1`, `trainer.job.progress.v1`, `trainer.job.completed.v1`, `trainer.job.failed.v1`
- Training metrics (`platform_core.trainer_metrics_events`): `trainer.metrics.config.v1`, `trainer.metrics.progress.v1`, `trainer.metrics.completed.v1`

Lifecycle events carry `job_id` (run_id), `user_id`, `queue`, `progress`, optional `message`, and optional `payload`. Metrics events carry training-specific fields (epochs, steps, loss/perplexity, artifact path) keyed by the same `job_id`/`user_id`.
