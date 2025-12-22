# Health Endpoints

Health and status endpoints for container orchestration and monitoring.

---

## GET /healthz

Liveness probe for container orchestration.

**Response (200):**
```json
{
  "status": "ok"
}
```

---

## GET /readyz

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

## GET /status

Comprehensive service status with dependency health, model info, and data counts.

**Response (200):**
```json
{
  "service": "covenant-radar-api",
  "version": "0.1.0",
  "dependencies": [
    {
      "name": "redis",
      "status": "ok",
      "message": null
    },
    {
      "name": "postgres",
      "status": "ok",
      "message": null
    }
  ],
  "model": {
    "model_id": "default",
    "model_path": "/data/models/active.ubj",
    "is_loaded": false
  },
  "data": {
    "deals": 5
  }
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `service` | string | Service name |
| `version` | string | Service version |
| `dependencies` | array | List of dependency health checks |
| `dependencies[].name` | string | Dependency name (`redis`, `postgres`) |
| `dependencies[].status` | string | `ok` or `error` |
| `dependencies[].message` | string\|null | Error message if status is `error` |
| `model.model_id` | string | Active model identifier |
| `model.model_path` | string | Path to active model file |
| `model.is_loaded` | boolean | Whether model is loaded in memory |
| `data.deals` | int | Total deal count in database |
