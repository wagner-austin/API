# Deal Endpoints

CRUD operations for managing loan deals.

---

## GET /deals

List all deals.

**Response (200):**
```json
[
  {
    "id": {"value": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"},
    "name": "TechCorp Senior Credit Facility",
    "borrower": "TechCorp Inc",
    "sector": "Technology",
    "region": "North America",
    "commitment_amount_cents": 50000000000,
    "currency": "USD",
    "maturity_date_iso": "2027-12-31"
  }
]
```

---

## POST /deals

Create a new deal.

**Request Headers:**

| Header | Required | Description |
|--------|----------|-------------|
| `Content-Type` | Yes | Must be `application/json` |
| `X-Request-ID` | No | Correlation ID (auto-generated if omitted) |

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | object | Yes | Deal identifier object |
| `id.value` | string | Yes | UUID for the deal |
| `name` | string | Yes | Deal name |
| `borrower` | string | Yes | Borrower company name |
| `sector` | string | Yes | Industry sector |
| `region` | string | Yes | Geographic region |
| `commitment_amount_cents` | int | Yes | Commitment amount in cents |
| `currency` | string | Yes | Currency code (e.g., `USD`, `EUR`) |
| `maturity_date_iso` | string | Yes | Maturity date in ISO format (YYYY-MM-DD) |

**Request Example:**
```json
{
  "id": {"value": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"},
  "name": "Demo Leveraged Buyout",
  "borrower": "Demo Corp",
  "sector": "Manufacturing",
  "region": "North America",
  "commitment_amount_cents": 75000000000,
  "currency": "USD",
  "maturity_date_iso": "2029-06-30"
}
```

**Response (201):**
```json
{
  "id": {"value": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"},
  "name": "Demo Leveraged Buyout",
  "borrower": "Demo Corp",
  "sector": "Manufacturing",
  "region": "North America",
  "commitment_amount_cents": 75000000000,
  "currency": "USD",
  "maturity_date_iso": "2029-06-30"
}
```

**Example - curl:**
```bash
curl -X POST http://localhost:8007/deals \
  -H "Content-Type: application/json" \
  -d '{
    "id": {"value": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"},
    "name": "Demo Leveraged Buyout",
    "borrower": "Demo Corp",
    "sector": "Manufacturing",
    "region": "North America",
    "commitment_amount_cents": 75000000000,
    "currency": "USD",
    "maturity_date_iso": "2029-06-30"
  }'
```

---

## GET /deals/{deal_id}

Get a deal by ID.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `deal_id` | string | Deal UUID |

**Response (200):**
```json
{
  "id": {"value": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"},
  "name": "Demo Leveraged Buyout",
  "borrower": "Demo Corp",
  "sector": "Manufacturing",
  "region": "North America",
  "commitment_amount_cents": 75000000000,
  "currency": "USD",
  "maturity_date_iso": "2029-06-30"
}
```

**Response (404):**
```json
{
  "code": "NOT_FOUND",
  "message": "Deal not found",
  "request_id": "uuid"
}
```

---

## PUT /deals/{deal_id}

Update an existing deal.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `deal_id` | string | Deal UUID |

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Deal name |
| `borrower` | string | Yes | Borrower company name |
| `sector` | string | Yes | Industry sector |
| `region` | string | Yes | Geographic region |
| `commitment_amount_cents` | int | Yes | Commitment amount in cents |
| `currency` | string | Yes | Currency code |
| `maturity_date_iso` | string | Yes | Maturity date in ISO format |

**Request Example:**
```json
{
  "name": "Updated Deal Name",
  "borrower": "Demo Corp",
  "sector": "Manufacturing",
  "region": "North America",
  "commitment_amount_cents": 80000000000,
  "currency": "USD",
  "maturity_date_iso": "2030-06-30"
}
```

**Response (200):**
```json
{
  "id": {"value": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d"},
  "name": "Updated Deal Name",
  "borrower": "Demo Corp",
  "sector": "Manufacturing",
  "region": "North America",
  "commitment_amount_cents": 80000000000,
  "currency": "USD",
  "maturity_date_iso": "2030-06-30"
}
```

---

## DELETE /deals/{deal_id}

Delete a deal by ID.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `deal_id` | string | Deal UUID |

**Response (204):** No content
