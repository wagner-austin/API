# Measurement Endpoints

Operations for managing financial measurements.

---

## GET /measurements/by-deal/{deal_id}

List all measurements for a specific deal.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `deal_id` | string | Deal UUID |

**Response (200):**
```json
[
  {
    "deal_id": {"value": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"},
    "period_start_iso": "2024-01-01",
    "period_end_iso": "2024-03-31",
    "metric_name": "total_debt",
    "metric_value_scaled": 1000000000
  }
]
```

---

## GET /measurements/by-deal/{deal_id}/period

List measurements for a deal within a specific period.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `deal_id` | string | Deal UUID |

**Query Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `period_start` | string | Yes | Period start date (YYYY-MM-DD) |
| `period_end` | string | Yes | Period end date (YYYY-MM-DD) |

**Response (200):**
```json
[
  {
    "deal_id": {"value": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"},
    "period_start_iso": "2024-01-01",
    "period_end_iso": "2024-03-31",
    "metric_name": "total_debt",
    "metric_value_scaled": 1000000000
  }
]
```

---

## POST /measurements

Add financial measurements for deals.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `measurements` | array | Yes | List of measurement objects |
| `measurements[].deal_id` | object | Yes | Deal identifier |
| `measurements[].deal_id.value` | string | Yes | Deal UUID |
| `measurements[].period_start_iso` | string | Yes | Period start date (YYYY-MM-DD) |
| `measurements[].period_end_iso` | string | Yes | Period end date (YYYY-MM-DD) |
| `measurements[].metric_name` | string | Yes | Metric name (e.g., `total_debt`, `ebitda`) |
| `measurements[].metric_value_scaled` | int | Yes | Metric value (scaled integer) |

**Request Example:**
```json
{
  "measurements": [
    {
      "deal_id": {"value": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"},
      "period_start_iso": "2024-01-01",
      "period_end_iso": "2024-03-31",
      "metric_name": "total_debt",
      "metric_value_scaled": 1000000000
    },
    {
      "deal_id": {"value": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"},
      "period_start_iso": "2024-01-01",
      "period_end_iso": "2024-03-31",
      "metric_name": "ebitda",
      "metric_value_scaled": 300000000
    }
  ]
}
```

**Response (201):**
```json
{
  "count": 2
}
```
