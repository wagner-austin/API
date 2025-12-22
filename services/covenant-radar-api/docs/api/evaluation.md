# Evaluation Endpoints

Covenant compliance evaluation.

---

## POST /evaluate

Evaluate covenant compliance for a deal and period.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `deal_id` | string | Yes | Deal UUID |
| `period_start_iso` | string | Yes | Period start date (YYYY-MM-DD) |
| `period_end_iso` | string | Yes | Period end date (YYYY-MM-DD) |
| `tolerance_ratio_scaled` | int | Yes | Near-breach tolerance ratio (scaled) |

**Request Example:**
```json
{
  "deal_id": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d",
  "period_start_iso": "2024-01-01",
  "period_end_iso": "2024-03-31",
  "tolerance_ratio_scaled": 10
}
```

**Response (200):**
```json
[
  {
    "covenant_id": {"value": "c1d2e3f4-a5b6-4c7d-8e9f-0a1b2c3d4e5f"},
    "period_start_iso": "2024-01-01",
    "period_end_iso": "2024-03-31",
    "calculated_value_scaled": 333,
    "status": "OK"
  }
]
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `covenant_id` | object | Covenant identifier |
| `period_start_iso` | string | Period start date |
| `period_end_iso` | string | Period end date |
| `calculated_value_scaled` | int | Calculated metric value |
| `status` | string | `OK`, `NEAR_BREACH`, or `BREACH` |

**Status Definitions:**

| Status | Description |
|--------|-------------|
| `OK` | Covenant is in compliance |
| `NEAR_BREACH` | Within tolerance threshold of breach |
| `BREACH` | Covenant threshold exceeded |
