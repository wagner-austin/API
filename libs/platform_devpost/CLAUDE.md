# AI Instructions for platform_devpost

## What This Library Does

- Fetches hackathons from Devpost's internal API
- Filters by themes, states, and interests
- Matches hackathons against codebase capabilities
- Scores fit (strong_fit, good_fit, stretch, new_territory)

## Do NOT Use This Directly

**Use the opportunity-radar-api instead:**

```bash
# Find hackathons matching codebase
curl "https://opportunity-radar-api-production.up.railway.app/devpost/hackathons"

# Filter by themes
curl "https://opportunity-radar-api-production.up.railway.app/devpost/hackathons?themes=AI&themes=Machine%20Learning"

# Filter by state
curl "https://opportunity-radar-api-production.up.railway.app/devpost/hackathons?states=open&states=upcoming"

# Featured only
curl "https://opportunity-radar-api-production.up.railway.app/devpost/hackathons?featured_only=true"

# Without codebase matching
curl "https://opportunity-radar-api-production.up.railway.app/devpost/hackathons?match_codebase=false"
```

## Getting Full Hackathon Details

The API returns basic hackathon info. For **full details** (requirements, prizes, judging criteria, submission rules), use WebFetch on the hackathon URL:

```
WebFetch: https://devpost.com/hackathons/{slug}
Prompt: "Extract requirements, prizes, judging criteria, timeline, and submission rules"
```

## When To Use This Library Directly

Only use this library directly when:
1. Writing tests for opportunity-radar-api
2. Extending hackathon matching logic
3. The API is unavailable

## Key Functions (for lib development only)

```python
from platform_devpost import (
    find_hackathons,      # Main entry point
    get_codebase_profile, # Get detected capabilities
    filter_hackathons,    # Apply interest filter
    match_hackathon,      # Score single hackathon
    match_hackathons,     # Score multiple hackathons
)
```

## Types

- `Hackathon`: id, title, url, open_state, time_left_to_submission, themes, prize_amount, etc.
- `HackathonMatch`: hackathon, match_score, matched_capabilities, missing_capabilities, recommendation
- `InterestFilter`: themes, exclude_themes, states
- `HackathonState`: "open" | "upcoming" | "ended" | "submissions"

## Testing Fakes

```python
from platform_devpost import (
    FakeDevpostClient,
    FakeDevpostApi,
    make_fake_hackathon,
    make_fake_theme,
    make_interest_filter,
    hooks,
    reset_hooks,
)

# Override client for testing
hooks.devpost_client = lambda: FakeDevpostClient(hackathons=(...))
```
