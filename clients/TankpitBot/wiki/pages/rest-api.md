---
title: REST API
tags: [api, rest, endpoints]
related:
  - "[[connection-protocol]]"
source_paths:
  - https://tankpit.com/api (official docs)
  - 2026-06-19 live endpoint testing
fact_checked: "2026-06-19"
confidence: high
verified: 2026-06-19 (every endpoint hit and response structure confirmed)
hubs: [js-client]
---

# REST API

TankPit has an official public REST API at `tankpit.com/api`. Rate limited to 1 request/second (HTTP 503 on exceeded).

## Endpoints

### GET /api/active_games

Returns array of currently active game sessions.

```json
[
  {
    "map": "Desert",
    "name": "World (Desert)",
    "game_id": 5,
    "playing_tanks_count": 1,
    "waiting_tanks_count": 0,
    "playing_tanks": [
      {
        "tank_id": 105117,
        "name": "Triumvirate",
        "awards": [3,3,0,1,0,0,0,0,0],
        "rank": "lieutenant",
        "color": "blue"
      }
    ]
  },
  {
    "map": "Rocks and Swamp",
    "name": "Practice",
    "game_id": 1,
    "playing_tanks_count": 0,
    "waiting_tanks_count": 0
  }
]
```

The `playing_tanks` array is present only when `playing_tanks_count > 0`. Awards array is 9 elements (decoration slots 0-8, values 0-3).

### GET /api/tank?tank_id={id}

Returns tank profile.

```json
{
  "tank_id": 105117,
  "name": "Triumvirate",
  "awards": [3,3,0,1,0,0,0,0,0],
  "main_color": "blue",
  "ping": "97ms",
  "favorite_map": "Desert",
  "bf_tank_name": "TripLeCrowN",
  "bf_tank_color": 1,
  "map_data": {"World": {}},
  "bulletin_board_posts": [24905, 24902, 24886, 24885, 24880, 24871]
}
```

`bf_tank_name` / `bf_tank_color` are the player's original Battlefield (bonus.com) tank from the 1997 game.

### GET /api/find_tank?name={name}

Search tanks by name. Returns array of matches.

```json
[
  {
    "tank_id": 105177,
    "name": "BoMbIrAn",
    "awards": [3,3,0,2,0,0,0,0,0],
    "main_color": "purple"
  }
]
```

### GET /api/leaderboards

Returns array of available leaderboard identifiers.

```json
["2026","2025","2024","2023","2022","2021","2020","2019","2018","2017","2013-2016","2012","overall"]
```

### GET /api/leaderboards?leaderboard={id}&page={page}

Paginated leaderboard results. Optional filters: `color`, `rank`, `search`.

```json
{
  "leaderboard": "2026",
  "page": 1,
  "total_pages": 32,
  "results": [
    {
      "name": "BoMbIrAn",
      "tank_id": 105177,
      "awards": [3,3,0,2,0,0,0,0,0],
      "rank": "general",
      "color": "purple",
      "placing": 1
    }
  ]
}
```

### GET /api/upcoming_tournaments

Returns array of upcoming tournaments. Empty if none scheduled.

### GET /api/finished_tournaments

Returns array of finished tournaments.

```json
[
  {
    "tournament_id": 2591,
    "start_time_utc": "2025-01-31 03:00:00",
    "end_time_utc": "2025-01-31 04:00:00",
    "map": "Deep Six"
  }
]
```

### GET /api/last_finished_tournament

Returns the most recently finished tournament. Empty if none.

### GET /api/tournament_results?tournament_id={id}

Returns results for a specific tournament.

### GET /api/bb?year={year}&month={month}&day={day}

Returns bulletin board posts for a specific date.

### GET /api/bb/post?post_id={id}

Returns a single bulletin board post.

```json
{
  "year": 2026,
  "month": 3,
  "day": 17,
  "section": "General",
  "message": "{tank:88595}, \r\nI had aspirations...",
  "tank_id": 105117,
  "tank_name": "Triumvirate",
  "awards": [3,3,0,1,0,0,0,0,0]
}
```

Tank mentions use `{tank:id}` format in message text.

## Web Resources

### Static Files

- CSS: `/content/style-{hash}.css` (82KB)
- Site JS: `/content/tp-{hash}.js` (12KB) — calendar, alerts, tank drag-and-drop, login UI
- Game JS: `/content/tpclient-{hash}.js` — the game client (only loaded on authenticated play page)
- Images: `/images/` — sprites, maps, menu, awards, guide

### Pages

| Path | Description |
|------|-------------|
| `/` | Home page |
| `/play` | Game client (requires auth, redirects to `/before-playing` if unauthenticated) |
| `/before-playing` | Guest tank name entry |
| `/leaderboards` | Paginated leaderboard with year/color/rank filters |
| `/tournaments` | Calendar + recent winners |
| `/bulletin-board` | Community message board |
| `/help` | FAQ with expandable answers |
| `/about` | Game history (remake of bonus.com Battlefield, 1997) |
| `/downloads` | Sprite packs (sprites.zip), Discord link |
| `/old-news` | News archive 2017-2023 |
| `/map-info?map_id={id}` | Per-map description page |
| `/tanks/profile?tank_id={id}` | Tank profile (awards, ping, stats, posts) |
| `/account/forgot-password` | Password recovery |
| `/contact` | Contact form |

### Authentication

Login uses a CAPTCHA-protected form with:
- `captcha_code` — per-page-load CAPTCHA token (`window.captcha_code`)
- `token` — CSRF token
- `redirect` — post-login redirect path

## FAQ Game Facts (from /help)

- **Bots rank up to Sergeant**: "We thought it would be interesting to allow the in game bots (eg. orange-1) to be promoted up to the rank of sergeant. If the bots reach the rank of sergeant, they become significantly smarter and harder to deactivate."
- **Disconnect protection**: Disconnected tanks become "uncontrollable" for 1 minute (can't be attacked). After that, armor shields enable and enemies can resume attacking.
- **Auto-mine under obstacles**: "The game automatically plants a mine under an obstacle you move if it detects that you might be inside a base."
- **Anti-cheat on promotions**: Kills between related accounts/IPs don't count for promotions (tournaments only currently).
- **Ping**: "A great ping is under 100ms while a poor ping is over 300ms."
- **Game origin**: Remake of "Battlefield" from bonus.com (1997), moved to playbattlefield.com, shut down 2008. TankPit started 2012.
