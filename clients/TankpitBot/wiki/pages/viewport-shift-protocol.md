---
title: Viewport Shift Protocol
tags: [protocol, viewport, wire, js-client]
related: [[viewport-frame]], [[viewport-update-algorithm]], [[v-table-complete]], [[js-source-map]], [[executor-rejection-loops]]
sources: [see footnotes]
fact_checked: 2026-07-17
confidence: high
---

# Viewport Shift Protocol

The complete client↔server contract for changing the visible viewport. Documents the three client-initiated commands that request a shift, the server response, and how the current bot's world-model relates to what the game actually supports.

## Three shift triggers

The server sends a fresh `0x5A ViewportUpdate` (see [[viewport-update-algorithm]]) whenever the viewport changes. The change can be triggered by any of:

1. **Teleport landing** — `0x74`-family teleport command lands the tank, server recenters the viewport on the new position and sends `0x5A`.
2. **Client scope command** — client sends `Rb` (`"Z"`) or `Sb` (`"z"`) requesting a shift; server confirms with `0x5A`.
3. **Server-side auto-shift** — when autoscroll is enabled (via `Ia` text control), the server sends `0x5A` **only when the tank walks onto a viewport-edge tile** (not on every walk step). The shift then **recenters the viewport on the tank** — the tank's post-shift position is at (or near) the center of the new 16×16 frame.[^user] Behaviour paired 1:1 with `0x3D MovementResponse` in the corpus.[^5]

## The three client-side commands

Two are binary game commands (encoded through `K` base class → framed + XOR'd), one is a plain text control command (encoded through `va` base class → no XOR).[^1]

### `Ia` — Autoscroll setting (text)

Persistent server-side setting toggled by the client. Text encoding: `"A" + Number(bool)`.[^2]

| Wire form | Meaning |
|---|---|
| `"A1"` | Autoscroll **ON** — server auto-sends `0x5A` on walk edge-crossings |
| `"A0"` | Autoscroll **OFF** — `0x5A` only fires on teleport / explicit scope command |

### `Rb` — Scope extend (binary, 3 bytes)

Client-initiated shift in a compass direction.[^3]

```
[len=3, 'Z'=0x5A, direction:1]
```

Direction byte is a menu index 0-8: `0=Center, 1=W, 2=E, 3=S, 4=N, 5=NE, 6=SE, 7=SW, 8=NW`.[^4] Same byte on the wire is `'Z' = 0x5A` — the ClientCommand `Z` and the ServerMessage `Z` (`0x5A ViewportUpdate`) share the character, distinguished only by direction (sent vs received).

### `Sb` — Scope move (binary, 4 bytes)

Client-initiated shift to a specific tile.

```
[len=4, 'z'=0x7A, x:1, y:1]
```

Lowercase `z` — different code from `Rb`.

## Client state machine

State 13 = "Scope change pending".[^6] Entered when the client sends `Rb` or `Sb`. The `0x5A` handler clears the `a.Ja` scope-change-pending flag on arrival (tpclient.pretty.js:4603).

## Empirical validation

Corpus: `runs/sniff/latest.capture_session.json` (2026-07-10, 421.8 s human-driven session).

| Event | Count |
|---|---|
| Received `0x5A ViewportUpdate` | 22 |
| Received `0x3D MovementResponse` | 22 (1:1 with 0x5A) |
| Received `0x47 Movement` (walk broadcast) | 42 |
| Sent teleport | 4 |
| Game log "Extend view {direction}" | 8 |

Every "Extend view" game-log line is followed by a `0x5A` 0–2 s later — proving `Rb → 0x5A` round-trip. The 22 − 4 − 8 = ~10 remaining `0x5A` messages coincide with walk broadcasts, evidencing server-side auto-shift under autoscroll=on. Note that `0x5A` count (22) is far less than `0x47` walk-broadcast count (42) — most walks do not trigger a shift, only edge-arriving walks do, consistent with the "auto-shift on edge-tile arrival" rule above.

## Bot's current state and the gap

The bot never sends `Ia`, `Rb`, or `Sb`. The sniffer correctly captures inbound `0x5A` and updates the viewport origin (see `src/tankpit_bot/sniffer/viewport.py::update_viewport_origin`), but with autoscroll off and no scope commands, `0x5A` only fires on teleport — so the bot's world-model assumption "viewport is fixed until teleport" holds.[^7]

This is a **bot configuration choice, not a game limit**. Turning it back on requires either sending `Ia("A1")` once (server takes over) or dispatching `Rb`/`Sb` when the planner wants to see off-viewport space. Consequences for existing bot logic in [[executor-rejection-loops]] and [[viewport-frame]].

## Latent doc bug (fixed 2026-07-18)

`src/tankpit_bot/protocol/commands.py:95-96` labelled `PLAIN_AUTOSCROLL_ON = b"A0"` and `PLAIN_AUTOSCROLL_OFF = b"A1"` — inverted from the JS truth (`Number(true) == 1`). Same inversion in `docs/protocol-discovery.md:435-436`. Constants were unused in `src/` (grep 2026-07-17), so no live misfire occurred. Both sites corrected 2026-07-18: `"A1"` = ON, `"A0"` = OFF.

[^1]: `Ia` inherits from `va` at tpclient.pretty.js:240; `Rb` and `Sb` inherit from `K` at tpclient.pretty.js:766/780. Both send paths are traced in [[js-source-map]] §"Client Command Classes".
[^2]: `Ia.prototype.toString = function() { return this.code + Number(this.h) };` at tpclient.pretty.js:241-243. `this.h` is the bool. `ke(a, b)` at line 5124 sends `new Ia(b)` when the setting toggles.
[^3]: `Rb.prototype.h` at tpclient.pretty.js:767-773: `a[0]=3, a[1]='Z'.charCodeAt(0), a[2]=direction`.
[^4]: Menu-index ordering from tpclient.pretty.js:6998 hotkey menu: "Scope Center;Scope W;Scope E;Scope S;Scope N;Scope NE;Scope SE;Scope SW;Scope NW". Client emits `new Rb(this.Y)` at line 1661 with `this.Y` being the raw menu index 0-8.
[^5]: `runs/sniff/latest.capture_session.json` — 22 `0x5A` paired 1:1 in time with 22 `0x3D MovementResponse` events, 8 game-log "Extend view {NE|E|SE|W|N}" events, 4 sent teleports.
[^6]: State 13 documented in [[js-source-map]] §"State Machine (s field)". Handler at tpclient.pretty.js:1648-1662 dispatches `Rb` (line 1661) or `Sb` (line 1654) based on the pending scope Y and the `ga` flag.
[^7]: `src/tankpit_bot/state/scan_coverage.py:29` — "the bot teleports the viewport is fixed until the next teleport"; `src/tankpit_bot/bot/ai/hunt_mode.py:52-53` — "viewport shifting is OFF, so walking to an edge reveals no new [tiles]".
[^user]: user (Austin), 2026-07-17 — "auto shift doesnt center on bot every walk. it only shifts when the bot walks to a tile on the edge of the viewport. and then it recenters on bot btw." Reaffirmed: "when you walk to the edge, with auto scroll on, it will center the viewport on the bot."
