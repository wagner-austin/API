---
title: Game Modes
tags: [js-client, game-mechanics, modes]
related:
  - "[[connection-protocol]]"
  - "[[client-constants]]"
  - "[[game-rules]]"
  - "[[tournament-strategy]]"
source_paths:
  - tpclient.js lines 207 (Jf class)
  - line 217 (mode parsing)
  - lines 265-266 (tournament guide); docs/sources/sigmas-tankpit-guide-v3.4.pdf for the tournament capacity ladder
fact_checked: "2026-07-06"
confidence: high
verified: 2026-06-19 (mode encoding/values traced from JS lobby handler); tournament capacity ladder guide-sourced, not verified against a tournament capture
hubs: [js-client]
---

# Game Modes

How game modes (practice, normal, tournament) are encoded and how they affect gameplay. Extracted from the lobby parsing code and the How To Play guide.

## Mode Encoding

Game mode is parsed from the server's game list message (code 43, `+` message):

```javascript
// Line 217, inside case 43:
switch (u[5]) {        // mode string from server
  case "e":
  case "t": ua = 5;    // Tournament
    break;
  case "p": ua = 6;    // Practice
    break;
  case "n": ua = 7;    // Normal
    break;
  default:
    x("Invalid game mode: " + ua);
    break;
}
```

The mode is stored as the `j` field on the Jf (game info) object.

## Mode Values

| Value | String | Mode | Description |
|-------|--------|------|-------------|
| 5 | "e" or "t" | Tournament | Competitive timed matches |
| 6 | "p" | Practice | Training fields |
| 7 | "n" | Normal | Standard gameplay |

## Practice Mode Differences

The JS client checks `6 === this.fa.j` (is practice) in several places:

1. **Top 10 disabled** (line 71, `fe` function):
   ```
   "This is a practice field\nTop 10 is not registered here\n
   If you feel you are skilled enough\ngo and try a real field"
   ```

2. **Promotion display** (line 60, Vd function):
   When rank is 1 (Private) and mode is practice:
   ```
   " * This is a practice field."
   ```
   (No promotion requirements shown)

3. **Deactivation bar** (line 61): When mode is 5 (not practice, i.e. tournament):
   `Dc(this.v, 0)` — reset promo state display

4. **Tips hidden** (line 75, key 22 handler):
   ```
   6 !== this.fa.j ? (this.Ba = !this.Ba) ? "Hide tips" : "Show tips"
                   : "This is a practice field."
   ```
   Practice fields don't support tip toggling.

## Tournament Mode Differences

From the How To Play guide (xi class, line 265):

1. **Equipment capacity**: Recruits can hold 60 of each equipment type instead of 20
2. **Elimination**: Tanks deactivated too many times are eliminated from the tournament
3. **Ranking**: Everyone starts as Recruit with a set time to compete
4. **Awards**: Top 3 get Bronze, Silver, Golden Cup decorations
5. **Free kills banned**: Intentionally allowing yourself to be deactivated is prohibited

### Tournament capacity ladder

Regular mode grows capacity by **+5 per rank** from a Recruit floor of 20 (see [[game-rules]]). Tournament mode grows capacity by **+8 per rank** from a Recruit floor of 60. Full progression:[^t1]

| Rank | Regular | Tournament |
|---|---:|---:|
| Recruit | 20 | 60 |
| Private | 25 | 68 |
| Corporal | 30 | 76 |
| Sergeant | 35 | 84 |
| Lieutenant | 40 | 92 |
| Captain | 45 | 100 |
| Major | 50 | 108 |
| Colonel | 55 | 116 |
| General | 60 | 124 |

**Points required for promotion and kill requirements are the same as the main map's** — only the carry cap changes.[^t1]

### Elimination flag

The elimination flag is transmitted in the TankExit (V.`)`) message:
```
a[4] = was_eliminated    — 1=eliminated from tournament
```

## Game Info Structure (Jf class, line 207)

```javascript
function Jf(a, b, c, d, e, f, g) {
  this.id = a;       // game_id
  this.name = b;     // display name
  this.o = c;        // field_id (map image index)
  this.l = d;        // flags array (tile theme indices per category)
  this.i = e;        // initial team (-1 = no preference)
  this.h = -1;       // selected team (set on join)
  this.j = f;        // game mode (5/6/7)
  this.m = g;        // series year (leaderboard year)
}
```

## Field Images

The field_id (`this.o`) determines which map image to load:

```javascript
// Line 98:
wd = Fe(D + "maps/field" + (10 > b ? "0" : "") + (b + "_r.gif"), "Map", ...);
```

Format: `/images/maps/field{XX}_r.gif` where XX is zero-padded field_id.

## Series/Leaderboard Year

The `this.m` field stores the leaderboard series year. The settings panel (Pi class, line 295) lets players choose between the current year's leaderboard or "Overall":

```javascript
// Line 297:
tankpit.series_id === a.l
  ? "Your tank was created during the current series..."
  : "Your tank was created during the {year} series. That series is now closed..."
```

The Overall leaderboard is enabled via the `O` settings command (Ja class).

## Lobby Statistics

When a game is selected, the server sends game info (code 61, `=` message):

```javascript
// Line 219:
l.da = u[1];    // start date
l.ea = u[2];    // name
l.P = parseInt(u[3], 10);   // rank
l.ca = parseInt(u[4], 10);  // orange count
l.ba = parseInt(u[5], 10);  // purple count
l.$ = parseInt(u[6], 10);   // blue count
l.aa = parseInt(u[7], 10);  // red count (note: Hi = max_rank parsed but not displayed)
```

This populates the lobby stats panel showing active forces per team.

[^t1]: Sigma's TankPit Tournament Guide v3.4, 16-Jan-2015 (`docs/sources/sigmas-tankpit-guide-v3.4.pdf`), §"Initial equipment fill" — full per-rank capacity table and the "points/kill requirements unchanged from main map" claim. Tournament mode is not exercised by this project; ladder is documented for preservation, not verified against tournament captures.
