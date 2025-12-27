# opportunity-radar-api - API Reference

Complete API documentation for the opportunity-radar-api service.

**Base URL:** `http://localhost:8000` (default)

---

## Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `KAGGLE_API_TOKEN` | Yes | - | Kaggle API token for API access |
| `PORT` | No | `8010` | Server port |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `LOG_FORMAT` | No | `json` | Log format: `json` or `text` |
| `GITHUB_TOKEN` | No | - | GitHub personal access token for codebase scanning |
| `GITHUB_REPO` | No | - | Repository to scan (format: `owner/repo`) |

### Codebase Scanning Modes

The service automatically detects how to scan the codebase:

| Mode | Condition | Description |
|------|-----------|-------------|
| **Local** | `GITHUB_TOKEN` not set | Scans local `libs/` and `services/` directories |
| **GitHub** | `GITHUB_TOKEN` + `GITHUB_REPO` set | Fetches `pyproject.toml` files via GitHub API |

GitHub mode is useful for Docker deployments where the local filesystem doesn't contain the monorepo.

---

## Health Endpoints

### GET /healthz

Liveness probe for container orchestration. Uses `platform_core.health.healthz()`.

**Response (200):**
```json
{
  "status": "ok"
}
```

---

## Codebase Endpoints

### GET /codebase/profile

Get the capability profile of the monorepo codebase.

**Response (200):**
```json
{
  "capabilities": [
    {
      "name": "xgboost_tabular",
      "strength": "strong",
      "tags": ["tabular", "classification", "regression", "xgboost"],
      "description": "XGBoost gradient boosting for tabular data"
    },
    {
      "name": "huggingface_transformers",
      "strength": "strong",
      "tags": ["nlp", "transformers", "huggingface", "text-classification", "text-generation", "llm"],
      "description": "Hugging Face Transformers for NLP and LLMs"
    },
    {
      "name": "torchvision_cv",
      "strength": "strong",
      "tags": ["computer-vision", "image", "pytorch", "image-classification"],
      "description": "TorchVision for computer vision tasks"
    },
    {
      "name": "language_identification",
      "strength": "moderate",
      "tags": ["nlp", "language-detection", "multilingual"],
      "description": "FastText for language identification"
    }
  ],
  "technologies": ["python"],
  "frameworks": ["fastapi", "flask"],
  "ml_backends": ["xgboost", "lightgbm", "pytorch", "sklearn", "transformers", "torchvision"],
  "data_formats": ["csv", "parquet", "excel"],
  "task_types": ["binary_classification", "image_classification", "multiclass_classification", "regression", "text_classification", "text_generation"]
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `capabilities` | array | List of detected capabilities |
| `capabilities[].name` | string | Capability identifier |
| `capabilities[].strength` | string | `strong`, `moderate`, or `basic` |
| `capabilities[].tags` | string[] | Tags for matching |
| `capabilities[].description` | string | Human-readable description |
| `technologies` | string[] | Programming languages |
| `frameworks` | string[] | Web/API frameworks |
| `ml_backends` | string[] | ML libraries |
| `data_formats` | string[] | Supported data formats |
| `task_types` | string[] | ML task types |

---

### GET /codebase/libs

List all libraries in the monorepo.

**Response (200):**
```json
[
  {
    "name": "platform-core",
    "path": "/path/to/libs/platform_core",
    "dependencies": ["httpx", "structlog"]
  },
  {
    "name": "platform-kaggle",
    "path": "/path/to/libs/platform_kaggle",
    "dependencies": ["kaggle", "platform-core", "platform-codebase"]
  }
]
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Library name from pyproject.toml |
| `path` | string | Absolute path to library |
| `dependencies` | string[] | Direct dependencies |

---

### GET /codebase/services

List all services in the monorepo.

**Response (200):**
```json
[
  {
    "name": "opportunity-radar-api",
    "path": "/path/to/services/opportunity-radar-api",
    "dependencies": ["fastapi", "platform-kaggle", "platform-devpost"],
    "has_rules_files": false
  },
  {
    "name": "turkic-api",
    "path": "/path/to/services/turkic-api",
    "dependencies": ["fastapi", "fasttext"],
    "has_rules_files": true
  }
]
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Service name from pyproject.toml |
| `path` | string | Absolute path to service |
| `dependencies` | string[] | Direct dependencies |
| `has_rules_files` | bool | Whether service has `.rules` files |

---

## Kaggle Endpoints

### GET /kaggle/competitions

Find Kaggle competitions matching criteria.

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `tags` | string[] | No | `[]` | Tags to include (repeatable, e.g., `?tags=tabular&tags=nlp`) |
| `exclude` | string[] | No | `[]` | Tags to exclude (repeatable) |
| `min_score` | float | No | `0.0` | Minimum match score (0.0-1.0) |
| `match_codebase` | bool | No | `true` | Score against codebase capabilities |
| `active_only` | bool | No | `true` | Exclude competitions past their deadline |
| `fetch_descriptions` | bool | No | `true` | Fetch full descriptions for better matching |

**Response (200) - With codebase matching:**
```json
[
  {
    "competition": {
      "ref": "amex-default-prediction",
      "title": "American Express Default Prediction",
      "category": "Featured",
      "reward": "$100,000",
      "deadline": "2024-08-01",
      "team_count": 5000,
      "tags": ["tabular", "classification", "finance"],
      "description": "Predict credit default...",
      "url": "https://www.kaggle.com/competitions/amex-default-prediction"
    },
    "match_score": 0.85,
    "matched_capabilities": ["xgboost", "classification", "tabular"],
    "missing_capabilities": [],
    "recommendation": "strong_fit"
  }
]
```

**Response (200) - Without codebase matching (`match_codebase=false`):**
```json
[
  {
    "ref": "amex-default-prediction",
    "title": "American Express Default Prediction",
    "category": "Featured",
    "reward": "$100,000",
    "deadline": "2024-08-01",
    "team_count": 5000,
    "tags": ["tabular", "classification", "finance"],
    "description": "Predict credit default...",
    "url": "https://www.kaggle.com/competitions/amex-default-prediction"
  }
]
```

**Recommendation Values:**

| Value | Match Score | Description |
|-------|-------------|-------------|
| `strong_fit` | >= 0.7 | Excellent match with codebase capabilities |
| `good_fit` | >= 0.4 | Good match, some capabilities align |
| `stretch` | >= 0.2 | Partial match, learning opportunity |
| `new_territory` | < 0.2 | Low match, significant new skills needed |

**Example - curl:**
```bash
# Find tabular competitions with 30%+ match score
curl "http://localhost:8000/kaggle/competitions?tags=tabular&min_score=0.3"

# Find NLP competitions, exclude image-related
curl "http://localhost:8000/kaggle/competitions?tags=nlp&exclude=image&exclude=computer-vision"

# Get all competitions without scoring
curl "http://localhost:8000/kaggle/competitions?match_codebase=false"

# Include expired competitions (for historical analysis)
curl "http://localhost:8000/kaggle/competitions?active_only=false"

# Skip description fetching for faster responses (less accurate matching)
curl "http://localhost:8000/kaggle/competitions?fetch_descriptions=false"
```

---

### GET /kaggle/competitions/{ref}

Get a specific competition by reference slug.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `ref` | string | Competition reference slug (e.g., `titanic`) |

**Response (200):**
```json
{
  "ref": "titanic",
  "title": "Titanic - Machine Learning from Disaster",
  "category": "Getting Started",
  "reward": "Knowledge",
  "deadline": "2030-01-01",
  "team_count": 15000,
  "tags": ["tabular", "classification", "beginner"],
  "description": "Start here! Predict survival on the Titanic",
  "url": "https://www.kaggle.com/competitions/titanic"
}
```

**Response (404):**
```json
{
  "detail": "Competition titanic-invalid not found"
}
```

---

## Devpost Endpoints

### GET /devpost/hackathons

Find Devpost hackathons matching criteria.

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `themes` | string[] | No | `[]` | Theme names to include (repeatable) |
| `exclude` | string[] | No | `[]` | Theme names to exclude (repeatable) |
| `states` | string[] | No | `["open"]` | Allowed states (repeatable) |
| `min_score` | float | No | `0.0` | Minimum match score (0.0-1.0) |
| `match_codebase` | bool | No | `true` | Score against codebase capabilities |
| `featured_only` | bool | No | `false` | Only return featured hackathons |

**Valid States:**

| State | Description |
|-------|-------------|
| `open` | Currently accepting submissions |
| `upcoming` | Not yet started |
| `ended` | Submission period closed |
| `submissions` | In submission review period |

**Response (200) - With codebase matching:**
```json
[
  {
    "hackathon": {
      "id": 12345,
      "title": "AI for Good Hackathon",
      "url": "https://devpost.com/hackathons/ai-for-good",
      "open_state": "open",
      "time_left_to_submission": "5 days",
      "themes": [
        {"id": 1, "name": "Machine Learning"},
        {"id": 2, "name": "Social Impact"}
      ],
      "prize_amount": "$10,000",
      "featured": true
    },
    "match_score": 0.75,
    "matched_capabilities": ["python", "fastapi", "ml"],
    "missing_capabilities": ["mobile"],
    "recommendation": "good_fit"
  }
]
```

**Response (200) - Without codebase matching (`match_codebase=false`):**
```json
[
  {
    "id": 12345,
    "title": "AI for Good Hackathon",
    "url": "https://devpost.com/hackathons/ai-for-good",
    "open_state": "open",
    "time_left_to_submission": "5 days",
    "themes": [
      {"id": 1, "name": "Machine Learning"},
      {"id": 2, "name": "Social Impact"}
    ],
    "prize_amount": "$10,000",
    "featured": true
  }
]
```

**Example - curl:**
```bash
# Find open AI hackathons
curl "http://localhost:8000/devpost/hackathons?themes=AI&themes=Machine%20Learning&states=open"

# Find upcoming and open hackathons
curl "http://localhost:8000/devpost/hackathons?states=open&states=upcoming"

# Find featured hackathons only
curl "http://localhost:8000/devpost/hackathons?featured_only=true"

# Get all hackathons without scoring
curl "http://localhost:8000/devpost/hackathons?match_codebase=false"
```

---

### GET /devpost/hackathons/{hackathon_id}

Get a specific hackathon by ID.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `hackathon_id` | int | Hackathon identifier |

**Response (200):**
```json
{
  "id": 12345,
  "title": "AI for Good Hackathon",
  "url": "https://devpost.com/hackathons/ai-for-good",
  "open_state": "open",
  "time_left_to_submission": "5 days",
  "themes": [
    {"id": 1, "name": "Machine Learning"},
    {"id": 2, "name": "Social Impact"}
  ],
  "prize_amount": "$10,000",
  "featured": true
}
```

**Response (404):**
```json
{
  "detail": "Hackathon 99999 not found"
}
```

---

## Error Handling

All errors return JSON with FastAPI's standard format:

**Validation Error (422):**
```json
{
  "detail": [
    {
      "type": "value_error",
      "loc": ["query", "min_score"],
      "msg": "Input should be less than or equal to 1",
      "input": "1.5"
    }
  ]
}
```

**Not Found (404):**
```json
{
  "detail": "Competition xyz not found"
}
```

**Internal Error (500):**
```json
{
  "code": "INTERNAL_ERROR",
  "message": "An unexpected error occurred",
  "request_id": "uuid"
}
```

---

## Content Types

**Response content type:**
- `application/json` (all endpoints)

---

## Rate Limiting

This service does not implement rate limiting directly. Rate limits are determined by the underlying Kaggle and Devpost APIs:

- **Kaggle API**: Subject to Kaggle's rate limits (typically 100 requests/day for list operations)
- **Devpost API**: Subject to Devpost's rate limits

---

## Caching

Currently, no caching is implemented. Each request fetches fresh data from the underlying APIs.

Future enhancements may include:
- In-memory caching with TTL
- Redis-backed caching for distributed deployments

---

## Usage Patterns

### Finding Competitions That Fit Your Codebase

The default behavior scores competitions against your codebase capabilities:

```bash
# Find tabular competitions with strong alignment (70%+ match)
curl "http://localhost:8000/kaggle/competitions?tags=tabular&min_score=0.7"
```

### Finding Learning Opportunities

To discover competitions that would help you grow and learn new skills, use a low or zero `min_score`:

```bash
# Find ALL tabular competitions, including ones requiring new skills
curl "http://localhost:8000/kaggle/competitions?tags=tabular&min_score=0.0"
```

The response includes a `recommendation` field that categorizes opportunities:

| Recommendation | Match Score | What It Means |
|----------------|-------------|---------------|
| `strong_fit` | >= 70% | Excellent match - you have the capabilities |
| `good_fit` | >= 40% | Good match - some capabilities align |
| `stretch` | >= 20% | Partial match - a learning opportunity |
| `new_territory` | < 20% | Low match - significant new skills needed |

**The `stretch` and `new_territory` categories are valuable for growth** - they identify competitions that would push you to learn new technologies and techniques.

### Pure Discovery Mode

To browse all available opportunities without scoring against your codebase:

```bash
# List all competitions without capability matching
curl "http://localhost:8000/kaggle/competitions?match_codebase=false"

# List all hackathons without capability matching
curl "http://localhost:8000/devpost/hackathons?match_codebase=false"
```

This returns raw competition/hackathon data without `match_score`, `matched_capabilities`, or `recommendation` fields - useful for general exploration.

### Combining Filters for Targeted Discovery

```bash
# Find NLP competitions - filter client-side for learning opportunities
curl "http://localhost:8000/kaggle/competitions?tags=nlp&min_score=0.0"
# Then filter response for recommendation == "stretch" or "new_territory"

# Find open hackathons in AI themes, see all skill levels
curl "http://localhost:8000/devpost/hackathons?themes=AI&themes=Machine%20Learning&states=open&min_score=0.0"

# Exclude computer-vision to focus on non-image competitions
curl "http://localhost:8000/kaggle/competitions?tags=tabular&exclude=computer-vision"
```

---

## Authentication

This service does not require authentication. All endpoints are publicly accessible.

For production deployments, consider adding:
- API key authentication via `platform_core.security`
- Rate limiting per client
