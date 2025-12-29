# GitHub Stats API - API Reference

Complete API documentation for the github-stats-api service.

**Base URLs:**
- Local (poetry): `http://localhost:8000`
- Docker Compose: `http://localhost:8009`
- Via Traefik: `http://localhost/github-stats`
- Railway: `https://<your-app>.railway.app`

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

Readiness probe. Returns 200 if service is ready to accept requests.

**Response (200):**
```json
{
  "status": "ready"
}
```

---

### GET /health

General health check endpoint.

**Response (200):**
```json
{
  "status": "ok"
}
```

---

## Stats Card Endpoints

### GET /api

Generate a GitHub user statistics SVG card.

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `username` | string | Yes | - | GitHub username (max 39 chars, alphanumeric + hyphens) |
| `theme` | string | No | `"default"` | Color theme (see [Themes](#themes)) |
| `hide_border` | string | No | `"false"` | Hide card border (`true`, `false`, `1`, `0`, `yes`, `no`) |
| `show_icons` | string | No | `"true"` | Show stat icons (`true`, `false`, `1`, `0`, `yes`, `no`) |
| `include_all_commits` | string | No | `"false"` | Include private commits (`true`, `false`, `1`, `0`, `yes`, `no`) |
| `hide` | string | No | - | Comma-separated stats to hide (see [Hideable Stats](#hideable-stats)) |
| `disable_animations` | string | No | `"false"` | Disable CSS animations (`true`, `false`, `1`, `0`, `yes`, `no`) |

**Response (200):** `image/svg+xml`

Returns an SVG image with the user's GitHub statistics.

**Response Headers:**

| Header | Value |
|--------|-------|
| `Content-Type` | `image/svg+xml` |
| `Cache-Control` | `max-age={CACHE_TTL_SECONDS}, s-maxage={CACHE_TTL_SECONDS}` |

**Example Request:**
```
GET /api?username=wagner-austin&theme=dracula&hide_border=true&show_icons=true
```

**Example - curl:**
```bash
curl "http://localhost:8000/api?username=wagner-austin&theme=dracula"
```

**Stats Card Content:**

The stats card displays:
- User name and GitHub rank (S+, S, A+, A, B+, B, C)
- Total Stars
- Total Commits
- Total PRs
- Total Issues
- Contributed to (repositories)

---

### GET /api/top-langs

Generate a GitHub user's top programming languages SVG card.

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `username` | string | Yes | - | GitHub username (max 39 chars, alphanumeric + hyphens) |
| `theme` | string | No | `"default"` | Color theme (see [Themes](#themes)) |
| `hide_border` | string | No | `"false"` | Hide card border (`true`, `false`, `1`, `0`, `yes`, `no`) |
| `layout` | string | No | `"default"` | Layout style (see [Layouts](#layouts)) |
| `langs_count` | string | No | `"8"` | Number of languages to show (1-20) |
| `hide` | string | No | - | Comma-separated languages to hide (case-sensitive) |
| `disable_animations` | string | No | `"false"` | Disable CSS animations (`true`, `false`, `1`, `0`, `yes`, `no`) |

**Response (200):** `image/svg+xml`

Returns an SVG image with the user's top programming languages.

**Response Headers:**

| Header | Value |
|--------|-------|
| `Content-Type` | `image/svg+xml` |
| `Cache-Control` | `max-age={CACHE_TTL_SECONDS}, s-maxage={CACHE_TTL_SECONDS}` |

**Example Request:**
```
GET /api/top-langs?username=wagner-austin&theme=dracula&layout=compact&langs_count=8
```

**Example - curl:**
```bash
curl "http://localhost:8000/api/top-langs?username=wagner-austin&layout=compact"
```

**Languages Card Content:**

The languages card displays:
- Language names with official GitHub colors
- Percentage of code in each language
- Visual bar or chart representation (based on layout)

---

### GET /api/capabilities

Generate a GitHub repository's codebase capabilities SVG card.

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `repo` | string | Yes | - | GitHub repository in `owner/repo` format |
| `theme` | string | No | `"default"` | Color theme (see [Themes](#themes)) |
| `hide_border` | string | No | `"false"` | Hide card border (`true`, `false`, `1`, `0`, `yes`, `no`) |
| `disable_animations` | string | No | `"false"` | Disable CSS animations (`true`, `false`, `1`, `0`, `yes`, `no`) |

**Response (200):** `image/svg+xml`

Returns an SVG image with the repository's detected capabilities.

**Response Headers:**

| Header | Value |
|--------|-------|
| `Content-Type` | `image/svg+xml` |
| `Cache-Control` | `max-age={CACHE_TTL_SECONDS}, s-maxage={CACHE_TTL_SECONDS}` |

**Example Request:**
```
GET /api/capabilities?repo=wagner-austin/model-trainer&theme=dracula
```

**Example - curl:**
```bash
curl "http://localhost:8000/api/capabilities?repo=wagner-austin/model-trainer&theme=dracula"
```

**Capabilities Card Content:**

The capabilities card displays:
- Detected ML/AI capabilities with strength indicators (strong, moderate, basic)
- ML backends (XGBoost, PyTorch, etc.)
- Frameworks (FastAPI, Flask, etc.)
- Supported data formats (CSV, Parquet, etc.)
- Task types (binary classification, image classification, etc.)

---

## Reference

### Themes

Available color themes:

| Theme | Description | Visual Effects |
|-------|-------------|----------------|
| `default` | Light theme with blue accents | None |
| `dark` | Dark background with light text | None |
| `dracula` | Dracula color scheme | None |
| `github_dark` | GitHub's dark mode colors | None |
| `transparent` | Transparent background | None |
| `cyberpunk` | Cyan/magenta neon sci-fi | Gradient, pulsing glow, twinkling sparkles |
| `synthwave` | Pink/blue 80s retro | Gradient, pulsing glow, twinkling sparkles |
| `neon` | Bright green/red neon | Gradient, pulsing glow, twinkling sparkles |
| `aurora` | Green/teal northern lights | Gradient, pulsing glow, twinkling sparkles |
| `radical` | Pink/yellow bold colors | Gradient, pulsing glow, twinkling sparkles |

### Visual Effects (Premium Themes)

The premium themes (cyberpunk, synthwave, neon, aurora, radical) include:

- **Gradient backgrounds**: Linear color transitions instead of flat colors
- **Pulsing glow**: Title text pulses with a neon glow effect (2s infinite animation)
- **Twinkling sparkles**: Decorative stars that fade in/out at different speeds (1.5-2.2s infinite animations)

These animations are CSS-based and render in GitHub READMEs.

### Layouts

Available layout styles for language cards:

| Layout | Description |
|--------|-------------|
| `default` | Vertical list with progress bars |
| `compact` | Horizontal compact bar |
| `donut` | Circular donut chart |
| `pie` | Pie chart visualization |

### Hideable Stats

Stats that can be hidden on the stats card:

| Value | Description |
|-------|-------------|
| `stars` | Total stars received |
| `commits` | Total commits |
| `prs` | Total pull requests |
| `issues` | Total issues |
| `contribs` | Repositories contributed to |

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
| `INVALID_INPUT` | 400 | Invalid query parameters |
| `NOT_FOUND` | 404 | GitHub user not found |
| `EXTERNAL_SERVICE_ERROR` | 502 | GitHub API error |

### Error Examples

**Missing required parameter:**
```json
{
  "code": "INVALID_INPUT",
  "message": "username is required",
  "request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Invalid username format:**
```json
{
  "code": "INVALID_INPUT",
  "message": "username contains invalid character '_'",
  "request_id": "550e8400-e29b-41d4-a716-446655440001"
}
```

**User not found:**
```json
{
  "code": "NOT_FOUND",
  "message": "GitHub user 'nonexistent-user-12345' not found",
  "request_id": "550e8400-e29b-41d4-a716-446655440002"
}
```

**Invalid theme:**
```json
{
  "code": "INVALID_INPUT",
  "message": "theme must be one of: aurora, cyberpunk, dark, default, dracula, github_dark, neon, radical, synthwave, transparent",
  "request_id": "550e8400-e29b-41d4-a716-446655440003"
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

---

## Content Types

**Response content types:**
- `image/svg+xml` (for card endpoints)
- `application/json` (for health and error responses)

---

## Caching

Responses include cache headers to reduce GitHub API load:

```
Cache-Control: max-age=1800, s-maxage=1800
```

Default TTL is 30 minutes (1800 seconds). Configure via `CACHE_TTL_SECONDS` environment variable.

---

## Rate Limiting

This service uses the GitHub GraphQL API, which has rate limits:
- **Authenticated:** 5,000 points per hour
- Each query consumes approximately 1-2 points

The caching mechanism helps minimize API calls. Consider increasing `CACHE_TTL_SECONDS` if you experience rate limiting.

---

## GitHub Username Validation

Usernames are validated according to GitHub's rules:

- Maximum 39 characters
- Alphanumeric characters and hyphens only
- Cannot start or end with a hyphen
- Cannot contain consecutive hyphens

---

## Ranking Algorithm

User rank is calculated based on activity:

```
score = commits * 1 + prs * 2 + issues * 1 + stars * 4
percentile = 100 - (log10(score + 1) * 15)
```

Rank thresholds:
| Rank | Percentile |
|------|------------|
| S+ | Top 1% |
| S | Top 12.5% |
| A+ | Top 25% |
| A | Top 37.5% |
| B+ | Top 50% |
| B | Top 62.5% |
| C | Rest |
