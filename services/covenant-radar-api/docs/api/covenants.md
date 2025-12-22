# Covenant Endpoints

CRUD operations for managing loan covenants.

---

## POST /covenants

Create a new covenant.

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | object | Yes | Covenant identifier object |
| `id.value` | string | Yes | UUID for the covenant |
| `deal_id` | object | Yes | Associated deal identifier |
| `deal_id.value` | string | Yes | Deal UUID |
| `name` | string | Yes | Covenant name |
| `formula` | string | Yes | Calculation formula (e.g., `total_debt / ebitda`) |
| `threshold_value_scaled` | int | Yes | Threshold value (scaled integer) |
| `threshold_direction` | string | Yes | `<=` or `>=` |
| `frequency` | string | Yes | `QUARTERLY` or `ANNUAL` |

**Request Example:**
```json
{
  "id": {"value": "c1d2e3f4-a5b6-4c7d-8e9f-0a1b2c3d4e5f"},
  "deal_id": {"value": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"},
  "name": "Max Leverage Ratio",
  "formula": "total_debt / ebitda",
  "threshold_value_scaled": 450,
  "threshold_direction": "<=",
  "frequency": "QUARTERLY"
}
```

**Response (201):**
```json
{
  "id": {"value": "c1d2e3f4-a5b6-4c7d-8e9f-0a1b2c3d4e5f"},
  "deal_id": {"value": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"},
  "name": "Max Leverage Ratio",
  "formula": "total_debt / ebitda",
  "threshold_value_scaled": 450,
  "threshold_direction": "<=",
  "frequency": "QUARTERLY"
}
```

---

## GET /covenants/by-deal/{deal_id}

List all covenants for a specific deal.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `deal_id` | string | Deal UUID |

**Response (200):**
```json
[
  {
    "id": {"value": "c1d2e3f4-a5b6-4c7d-8e9f-0a1b2c3d4e5f"},
    "deal_id": {"value": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"},
    "name": "Max Leverage Ratio",
    "formula": "total_debt / ebitda",
    "threshold_value_scaled": 450,
    "threshold_direction": "<=",
    "frequency": "QUARTERLY"
  }
]
```

---

## GET /covenants/{covenant_id}

Get a covenant by ID.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `covenant_id` | string | Covenant UUID |

**Response (200):**
```json
{
  "id": {"value": "c1d2e3f4-a5b6-4c7d-8e9f-0a1b2c3d4e5f"},
  "deal_id": {"value": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"},
  "name": "Max Leverage Ratio",
  "formula": "total_debt / ebitda",
  "threshold_value_scaled": 450,
  "threshold_direction": "<=",
  "frequency": "QUARTERLY"
}
```

---

## DELETE /covenants/{covenant_id}

Delete a covenant by ID.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `covenant_id` | string | Covenant UUID |

**Response (204):** No content
