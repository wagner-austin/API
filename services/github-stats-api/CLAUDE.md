# AI Instructions for GitHub Stats API

## When to Use This

Use these instructions when the user asks about:
- GitHub profile stats cards (commits, stars, PRs, issues, rank)
- Top programming languages visualization
- Codebase capabilities detection and display
- Generating SVG cards for GitHub READMEs

## Production API

**Base URL:** `https://github-stats-api-production-5042.up.railway.app`

## Exactly What To Do

### 1. Generate User Stats Card

```bash
curl "https://github-stats-api-production-5042.up.railway.app/api?username=wagner-austin&theme=dracula"
```

Returns an SVG image with user's GitHub statistics (commits, stars, PRs, issues, rank).

### 2. Generate Top Languages Card

```bash
curl "https://github-stats-api-production-5042.up.railway.app/api/top-langs?username=wagner-austin&layout=compact&theme=dracula"
```

Returns an SVG image with user's most-used programming languages.

### 3. Generate Codebase Capabilities Card

```bash
curl "https://github-stats-api-production-5042.up.railway.app/api/capabilities?repo=wagner-austin/model-trainer&theme=dracula"
```

Returns an SVG image showing detected ML/AI capabilities, backends, frameworks, and task types.

## Query Parameters

### Stats Card (/api)
| Param | Default | Description |
|-------|---------|-------------|
| `username` | required | GitHub username |
| `theme` | `default` | Color theme (default, dark, dracula, github_dark, transparent) |
| `hide_border` | `false` | Hide card border |
| `show_icons` | `true` | Show stat icons |
| `include_all_commits` | `false` | Include all commits, not just current year |
| `hide` | `""` | Comma-separated stats to hide (stars, commits, prs, issues, contribs) |
| `disable_animations` | `false` | Disable CSS animations |

### Languages Card (/api/top-langs)
| Param | Default | Description |
|-------|---------|-------------|
| `username` | required | GitHub username |
| `theme` | `default` | Color theme |
| `hide_border` | `false` | Hide card border |
| `layout` | `default` | Layout style (default, compact, donut, pie) |
| `langs_count` | `8` | Number of languages (1-20) |
| `hide` | `""` | Comma-separated languages to hide |
| `disable_animations` | `false` | Disable CSS animations |

### Capabilities Card (/api/capabilities)
| Param | Default | Description |
|-------|---------|-------------|
| `repo` | required | GitHub repo in owner/repo format |
| `theme` | `default` | Color theme |
| `hide_border` | `false` | Hide card border |
| `disable_animations` | `false` | Disable CSS animations |

## Embedding in GitHub README

### Stats Card
```markdown
![GitHub Stats](https://github-stats-api-production-5042.up.railway.app/api?username=wagner-austin&theme=dracula)
```

### Languages Card
```markdown
![Top Languages](https://github-stats-api-production-5042.up.railway.app/api/top-langs?username=wagner-austin&layout=compact&theme=dracula)
```

### Capabilities Card
```markdown
![Codebase Capabilities](https://github-stats-api-production-5042.up.railway.app/api/capabilities?repo=wagner-austin/model-trainer&theme=dracula)
```

## Available Themes

| Theme | Description |
|-------|-------------|
| `default` | Light theme with blue accents |
| `dark` | Dark background with light text |
| `dracula` | Dracula color scheme |
| `github_dark` | GitHub's dark mode colors |
| `transparent` | Transparent background |

## Available Layouts (Languages Card)

| Layout | Description |
|--------|-------------|
| `default` | Vertical list with progress bars |
| `compact` | Horizontal compact bar |
| `donut` | Circular donut chart |
| `pie` | Pie chart visualization |

## Do NOT

- Do NOT use github-readme-stats.vercel.app for users in this codebase - use this API
- Do NOT hardcode stats - always use the live API
- Do NOT disable animations unless specifically requested

## API Availability

The API is deployed on Railway. If it returns errors:
1. Check if Railway is up
2. The API requires GITHUB_TOKEN to be set on the server
3. Check the /healthz endpoint for service status
