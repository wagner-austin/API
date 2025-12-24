# Covenant Radar API - API Reference

Complete API documentation for the covenant-radar-api service.

**Base URL:** `http://localhost:8007` (default)

---

## Endpoint Documentation

| Section | Description |
|---------|-------------|
| [Health Endpoints](api/health.md) | Liveness, readiness, and status probes |
| [Deal Endpoints](api/deals.md) | CRUD operations for loan deals |
| [Covenant Endpoints](api/covenants.md) | CRUD operations for loan covenants |
| [Measurement Endpoints](api/measurements.md) | Financial measurement management |
| [Evaluation Endpoints](api/evaluation.md) | Covenant compliance evaluation |
| [ML Endpoints](api/ml.md) | Model training, optimization, prediction, and explanation |
| [CLI Tools](api/cli.md) | Command-line tools for local development |

## Integrations

| Section | Description |
|---------|-------------|
| [Datadog](integrations/datadog.md) | APM tracing and custom metrics |
| [Streaming](integrations/streaming.md) | Kafka streaming for real-time inference |

---

## Quick Reference

### Health

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/healthz` | GET | Liveness probe |
| `/readyz` | GET | Readiness probe |
| `/status` | GET | Comprehensive service status |

### Deals

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/deals` | GET | List all deals |
| `/deals` | POST | Create a deal |
| `/deals/{deal_id}` | GET | Get a deal |
| `/deals/{deal_id}` | PUT | Update a deal |
| `/deals/{deal_id}` | DELETE | Delete a deal |

### Covenants

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/covenants` | POST | Create a covenant |
| `/covenants/by-deal/{deal_id}` | GET | List covenants by deal |
| `/covenants/{covenant_id}` | GET | Get a covenant |
| `/covenants/{covenant_id}` | DELETE | Delete a covenant |

### Measurements

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/measurements` | POST | Add measurements |
| `/measurements/by-deal/{deal_id}` | GET | List measurements by deal |
| `/measurements/by-deal/{deal_id}/period` | GET | List measurements by period |

### Evaluation

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/evaluate` | POST | Evaluate covenant compliance |

### ML

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ml/predict` | POST | Predict breach risk |
| `/ml/train` | POST | Train on internal data |
| `/ml/train-external` | POST | Train on external datasets |
| `/ml/optimize` | POST | Hyperparameter optimization |
| `/ml/explain` | POST | Feature importance explanation |
| `/ml/jobs/{job_id}` | GET | Get job status |
| `/ml/models/active` | GET | Get active model info |

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
| `INTERNAL_ERROR` | 500 | Internal server error |

### Error Examples

**Invalid request body:**
```json
{
  "code": "INVALID_INPUT",
  "message": "Missing required field 'deal_id'",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Resource not found:**
```json
{
  "code": "NOT_FOUND",
  "message": "Deal not found",
  "request_id": "550e8400-e29b-41d4-a716-446655440001"
}
```

---

## Request ID Tracing

All requests are assigned a unique `request_id` for tracing:

- **Provided:** Pass `X-Request-ID` header
- **Generated:** UUID v4 auto-generated if header omitted

The `request_id` appears in:
- All error responses
- Structured logs

When [Datadog integration](integrations/datadog.md) is enabled, logs also include `dd.trace_id` and `dd.span_id` for correlation with APM traces.

---

## Content Types

**Request content type:**
- `application/json` (for all POST/PUT endpoints)

**Response content types:**
- `application/json` (all endpoints)

---

## Domain Models

### Deal

| Field | Type | Description |
|-------|------|-------------|
| `id` | DealId | Unique identifier |
| `name` | string | Deal name |
| `borrower` | string | Borrower company name |
| `sector` | string | Industry sector |
| `region` | string | Geographic region |
| `commitment_amount_cents` | int | Commitment in cents |
| `currency` | string | Currency code |
| `maturity_date_iso` | string | Maturity date (YYYY-MM-DD) |

### Covenant

| Field | Type | Description |
|-------|------|-------------|
| `id` | CovenantId | Unique identifier |
| `deal_id` | DealId | Associated deal |
| `name` | string | Covenant name |
| `formula` | string | Calculation formula |
| `threshold_value_scaled` | int | Threshold (scaled integer) |
| `threshold_direction` | string | `<=` or `>=` |
| `frequency` | string | `QUARTERLY` or `ANNUAL` |

### Measurement

| Field | Type | Description |
|-------|------|-------------|
| `deal_id` | DealId | Associated deal |
| `period_start_iso` | string | Period start (YYYY-MM-DD) |
| `period_end_iso` | string | Period end (YYYY-MM-DD) |
| `metric_name` | string | Metric name |
| `metric_value_scaled` | int | Value (scaled integer) |

### CovenantResult

| Field | Type | Description |
|-------|------|-------------|
| `covenant_id` | CovenantId | Evaluated covenant |
| `period_start_iso` | string | Period start |
| `period_end_iso` | string | Period end |
| `calculated_value_scaled` | int | Calculated value |
| `status` | string | `OK`, `NEAR_BREACH`, `BREACH` |

---

## Scaled Integer Convention

All monetary and ratio values use scaled integers to avoid floating-point precision issues:

- **Monetary values**: Stored in cents (multiply by 100)
  - Example: $500,000,000 = `50000000000` cents
- **Ratios**: Scaled by 100 for two decimal places
  - Example: 4.5x leverage = `450` scaled

This convention ensures deterministic calculations and exact comparisons.
