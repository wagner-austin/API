# AI Instructions for Opportunity Radar

## When to Use This

Use these instructions when the user asks about:
- Kaggle competitions (finding, matching, analyzing)
- Devpost hackathons (finding, matching, analyzing)
- What competitions/hackathons fit their codebase
- Opportunities for ML, data science, or hackathon projects

## Production API

**Base URL:** `https://opportunity-radar-api-production.up.railway.app`

## Exactly What To Do

### 1. Find Kaggle Competitions Matching Codebase

```bash
curl "https://opportunity-radar-api-production.up.railway.app/kaggle/competitions"
```

Response includes `match_score`, `matched_capabilities`, and `recommendation` (strong_fit/good_fit/stretch/new_territory).

### 2. Find Devpost Hackathons

```bash
curl "https://opportunity-radar-api-production.up.railway.app/devpost/hackathons"
```

### 3. Get Codebase Capabilities

```bash
curl "https://opportunity-radar-api-production.up.railway.app/codebase/profile"
```

Shows what ML backends, frameworks, and capabilities the monorepo has.

### 4. Get Specific Competition Details

```bash
curl "https://opportunity-radar-api-production.up.railway.app/kaggle/competitions/{ref}"
```

Example: `/kaggle/competitions/titanic`

### 5. Get Specific Hackathon Details

For Devpost hackathons, the API returns basic info. For **full details** (requirements, prizes, judging criteria), use WebFetch on the hackathon URL directly:

```
WebFetch: https://devpost.com/hackathons/{slug}
Prompt: "Extract requirements, prizes, judging criteria, and submission deadline"
```

## Query Parameters

### Kaggle Competitions
| Param | Default | Description |
|-------|---------|-------------|
| `tags` | `[]` | Filter by tags (repeatable) |
| `exclude` | `[]` | Exclude tags (repeatable) |
| `min_score` | `0.0` | Minimum match score 0.0-1.0 |
| `match_codebase` | `true` | Score against codebase |
| `active_only` | `true` | Only active competitions |

### Devpost Hackathons
| Param | Default | Description |
|-------|---------|-------------|
| `themes` | `[]` | Filter by themes (repeatable) |
| `exclude` | `[]` | Exclude themes (repeatable) |
| `states` | `["open"]` | States: open, upcoming, ended |
| `min_score` | `0.0` | Minimum match score |
| `featured_only` | `false` | Only featured hackathons |

## Example Workflows

### "What hackathons should I do?"

1. Hit `/devpost/hackathons` to get matches
2. Sort by `match_score` descending
3. For top 3-5, use WebFetch on their URLs to get full details
4. Present recommendations with fit analysis

### "Find ML competitions for my skills"

1. Hit `/kaggle/competitions?tags=tabular&tags=classification`
2. Filter response by `recommendation == "strong_fit"` or `"good_fit"`
3. Present with deadlines and prize info

### "What can my codebase do?"

1. Hit `/codebase/profile`
2. Summarize capabilities, ML backends, and task types

## Do NOT

- Do NOT import platform_kaggle or platform_devpost directly - use the API
- Do NOT run scripts/find_competitions.py - use the API
- Do NOT scrape Kaggle directly - use the API (it handles auth)
- Do NOT guess at codebase capabilities - hit /codebase/profile

## API Availability

The API is deployed on Railway. If it returns errors:
1. Check if Railway is up
2. The API requires KAGGLE_API_TOKEN to be set on the server
3. Devpost endpoints work without auth
