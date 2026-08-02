---
title: Viewport Shift Protocol
tags: [protocol, viewport, wire, js-client]
related:
  - "[[viewport-frame]]"
  - "[[viewport-update-algorithm]]"
  - "[[v-table-complete]]"
  - "[[js-source-map]]"
  - "[[executor-rejection-loops]]"
source_paths:
  - "runs/bot"
  - "runs/sniff"
fact_checked: "2026-08-01"
confidence: high
hubs: [protocol]
---

# Viewport Shift Protocol

The complete client↔server contract for changing the visible viewport. Documents the three client-initiated commands that request a shift, the server response, and how the current bot's world-model relates to what the game actually supports.[^1]

## Three shift triggers

The server sends a fresh `0x5A ViewportUpdate` (see [[viewport-update-algorithm]]) whenever the viewport changes. The change can be triggered by any of:

1. **Teleport landing** — `0x74`-family teleport command lands the tank, server recenters the viewport on the new position and sends `0x5A`.
2. **Client scope command** — client sends `Rb` (`"Z"`) or `Sb` (`"z"`) requesting a shift; server confirms with `0x5A`.
3. **Server-side auto-shift** — when autoscroll is enabled (via `Ia` text control), the server sends `0x5A` **only when the tank walks onto a viewport-edge tile** (not on every walk step). The shift then **recenters the viewport on the tank** at exactly `(x−8, y−8)`.[^user] **Measured live 2026-07-25 as a controlled OFF/ON pair** (viewport probe runs 20260725-190352 and -192738):[^8]
   - **ON**: the tank walked east from window `(138,116)`; the step onto the edge column 153 (`138+15`) delivered — in the same wire tick as its `0x47` echo — a fresh `0x5A window=(145,116)` with `145 = 153−8`, no teleport involved.
   - **OFF (control)**: the tank walked onto the edge column 168 of window `(153,121)` and **no `0x5A` ever came** — the window is static; only a teleport recenters it.

   User corroboration (verbatim, 2026-07-24): *"autoscroll makes it so that when you get to the edge of the viewport it re centers on you. otherwise the viewport is fixed and only centers on teleport."*[^user] **The setting is SERVER-PERSISTED per account** — run 20260725-192738 opened in exactly the OFF state run 20260725-190352's restore press had left (fresh browser + fresh login between them).[^8] The mystery of runs opening ON that day traces to the 2026-07-24 key probe's "restore" press, which — under the then-inverted constant understanding — actually sent the ON toggle and left the account ON for a day.[^9]

## The three client-side commands

Two are binary game commands (encoded through `K` base class → framed + XOR'd), one is a plain text control command (encoded through `va` base class → no XOR).[^1]

### `Ia` — Autoscroll setting (text)

Persistent server-side setting toggled by the client. Text encoding: `"A" + Number(bool)`.[^2]

| Wire form | Meaning |
|---|---|
| `"A1"` | Autoscroll **ON** — server auto-sends `0x5A` on walk edge-crossings |
| `"A0"` | Autoscroll **OFF** — `0x5A` only fires on teleport / explicit scope command |

The server **acks the toggle with the same plaintext two-byte echo, un-XORed** — raw `41 30`/`41 31` on the wire (and `43 30`/`43 31` for the chat toggle).[^9] The 0x41/0x43 letters are overloaded with XOR-encoded binary frames (Deactivation, CacheUpdate), so the ack must be discriminated **before** any XOR decode: exactly two raw bytes with an ASCII `0`/`1` flag (`protocol.decoders.text.try_decode_plaintext_ack`). The first decoder read the flag byte *after* XOR corruption and tested `== 1`, so a true ON ack decoded as False — both 2026-07-25 viewport-probe aborts were this bug, not game behavior (fixed in the same day's `protocol:` commit).

### `Rb` — Scope extend (binary, 3 bytes)

Client-initiated shift in a compass direction.[^3]

```
[len=3, 'Z'=0x5A, direction:1]
```

**Direction byte is the compass CLOCKWISE FROM NORTH** (wire-measured 2026-08-01[^11]): `0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW`, plus `8=Center` (recenter on the tank — user-confirmed option 2026-08-01; the byte is the inferred remaining value, unobserved on the wire). The earlier menu-index reading (`0=Center, 1=W, ...` from the tpclient hotkey-menu string order[^4]) is **FALSIFIED** for the wire byte: the 2026-07-10 capture's 8 sent `Rb` frames pair with game-log "Extend view {dir}" lines as `0=N, 1=NE, 2=E, 3=SE, 6=W` — and the JS key handlers (`End`/`Home` → 5/7) independently corroborate the clockwise table. Same byte on the wire is `'Z' = 0x5A` — the ClientCommand `Z` and the ServerMessage `Z` (`0x5A ViewportUpdate`) share the character, distinguished only by direction (sent vs received).

#### The ANCHOR law (measured 2026-08-01, zero free parameters)

The server does NOT stride the window a fixed distance — it **anchors it to the tank so the full 16×16 extends in the requested direction** (user corroboration, verbatim, 2026-08-01: *"theres 8 caridnal movements which shift it all the way in that direction so the bot is at the very edge. eg if you used shift east the viewport would shift east all the way so the bot ... is on the western edge of the viewport"*). Per axis:

| Direction component | New origin axis |
|---|---|
| eastward (NE/E/SE) | `left = tank_x` (tank pinned to the WEST edge) |
| westward (SW/W/NW) | `left = tank_x − 15` (tank on the EAST edge) |
| southward (SE/S/SW) | `top = tank_y` (tank on the NORTH edge) |
| northward (NW/N/NE) | `top = tank_y − 15` (tank on the SOUTH edge) |
| unnamed axis | unchanged |

All 8 capture events fit exactly (walk-corrected self positions from the 0x47 path echoes)[^11]:

| Extend | Self | Window before | Window after (0x5A) | Anchor check |
|---|---|---|---|---|
| NE | (129,172) | (114,163) | (129,157) | (129, 172−15) ✓ |
| E | (144,164) | (129,157) | (144,157) | (144, top kept) ✓ |
| E | (160,168) | (145,159) | (160,159) | (160, top kept) ✓ |
| SE | (170,174) | (160,159) | (170,174) | (170, 174) ✓ |
| W | (188,233) | (188,221) | (173,221) | (188−15, top kept) ✓ |
| SE | (188,233) | (173,221) | (188,233) | (188, 233) ✓ |
| E | (96,207) | (81,201) | (96,201) | (96, top kept) ✓ |
| N | (113,34) | (110,33) | (110,19) | (left kept, 34−15) ✓ |

Consequences: a single pan shows at most 15 tiles beyond the tank in the panned direction; **repeated pans do not scroll across the map** (the anchor re-derives the same tank-pinned window), so the reachable view is the 31×31 area centered on the tank. Confirm lag measured 50 ms–1.5 s; every `Rb` is answered, even when the patch enumerates nothing.[^11]

### `Sb` — Scope move (binary, 4 bytes)

Client-initiated shift to a specific tile.[^1]

```
[len=4, 'z'=0x7A, x:1, y:1]
```

Lowercase `z` — different code from `Rb`.[^1]

## Client state machine

State 13 = "Scope change pending".[^6] Entered when the client sends `Rb` or `Sb`. The `0x5A` handler clears the `a.Ja` scope-change-pending flag on arrival (tpclient.pretty.js:4603).

## Empirical validation

Corpus: `runs/sniff/latest.capture_session.json` (2026-07-10, 421.8 s human-driven session).[^5]

| Event | Count |
|---|---|
| Received `0x5A ViewportUpdate` | 22 |
| Received `0x3D MovementResponse` | 22 (1:1 with 0x5A) |
| Received `0x47 Movement` (walk broadcast) | 42 |
| Sent teleport | 4 |
| Game log "Extend view {direction}" | 8 |

Every "Extend view" game-log line is followed by a `0x5A` 0–2 s later — proving `Rb → 0x5A` round-trip.[^5] The 22 − 4 − 8 = ~10 remaining `0x5A` messages coincide with walk broadcasts, evidencing server-side auto-shift under autoscroll=on. Note that `0x5A` count (22) is far less than `0x47` walk-broadcast count (42) — most walks do not trigger a shift, only edge-arriving walks do, consistent with the "auto-shift on edge-tile arrival" rule above.

## Bot's current state (updated 2026-08-01)

**The bot now sends `Rb`** — the scope-shift capability landed end to end: `make_scope_shift_command` → executor → `bot_dispatch.scope_shift` (`build_scope_command`, fire-and-forget like chat; the sniffer's 0x5A ingestion updates the origin when the confirm lands). First doctrine consumer is the **ferry scope scout** (`bot/ai/scope_scout.py`): when the larder declines a water-locked container `no_landing` and no fresh ferry belief exists, one FREE pan at the goal's water runs before any discovery teleport — a revealed ferry arrives as a 0x5A terrain-5 patch and the next larder tick hops `ferry_served` ([[ferry-mechanics]]). One pan per 30 s cooldown (`SCOPE_SCOUT_COOLDOWN_MS` — a no-ferry pan leaves no negative belief, so the latch is what stops a re-fire loop); never during a held combat lock. `Ia` is sent only by the session-start OFF dance; `Sb` remains unsent. The sim enforces the anchor law (`sim/viewport_window.py::apply_scope_shift`, all 8 measured rows pinned in `tests/sim/test_scope.py`) and answers every client `Rb` with the shifted 0x5A.[^7]

### Autoscroll doctrine addendum (user rulings 2026-08-01, NOT yet implemented)

User, verbatim: *"tbh when i use ferries i use auto scroll on so i can ride it across multiple viewports. also when i forage with no extra radars, i use auto scroll on."* Two standing conditions where the human practice turns `Ia("A1")` ON: (1) **ferry rides beyond one window** — with autoscroll OFF the ride's legs stop at the stored window edge and the next click cannot progress; with ON each edge arrival recenters (one tick) and the ride continues; (2) **foraging with zero extra radars** — edge-walk recenters expose fresh ground to the free radar. Also reconfirmed: *"if you walk to an edge or to a corner tile, with autoscroll on, itll center on you. whcih takes one tivk ofc"* (matches the measured 2026-07-25 edge-arrival law). The bot still pins autoscroll OFF for the whole session; dynamic toggling (ON while riding/radar-broke foraging, OFF otherwise) is the queued next build.

This is a **bot configuration choice, not a game limit** — and a deliberate one (user, verbatim, 2026-07-24: *"i usually run the bot with autoscroll off. it was too complicated too implement proper viewport awareness for the bot."*). The fixed-viewport + teleport-recenter model IS the intended operating mode; do not "restore" autoscroll to on for the account. Turning it back on requires either sending `Ia("A1")` once (server takes over) or dispatching `Rb`/`Sb` when the planner wants to see off-viewport space. Consequences for existing bot logic in [[executor-rejection-loops]] and [[viewport-frame]].

## Latent doc bug (fixed 2026-07-18)

`src/tankpit_bot/protocol/commands.py:95-96` labelled `PLAIN_AUTOSCROLL_ON = b"A0"` and `PLAIN_AUTOSCROLL_OFF = b"A1"` — inverted from the JS truth (`Number(true) == 1`, see [^2]). Same inversion in `docs/protocol-discovery.md:435-436`. Constants were unused in `src/` (grep 2026-07-17), so no live misfire occurred. Both sites corrected 2026-07-18: `"A1"` = ON, `"A0"` = OFF.

[^1]: `Ia` inherits from `va` at tpclient.pretty.js:240; `Rb` and `Sb` inherit from `K` at tpclient.pretty.js:766/780. Both send paths are traced in [[js-source-map]] §"Client Command Classes".
[^2]: `Ia.prototype.toString = function() { return this.code + Number(this.h) };` at tpclient.pretty.js:241-243. `this.h` is the bool. `ke(a, b)` at line 5124 sends `new Ia(b)` when the setting toggles.
[^3]: `Rb.prototype.h` at tpclient.pretty.js:767-773: `a[0]=3, a[1]='Z'.charCodeAt(0), a[2]=direction`.
[^4]: Menu-index ordering from tpclient.pretty.js:6998 hotkey menu: "Scope Center;Scope W;Scope E;Scope S;Scope N;Scope NE;Scope SE;Scope SW;Scope NW". Client emits `new Rb(this.Y)` at line 1661 with `this.Y` being the raw menu index 0-8.
[^5]: `runs/sniff/latest.capture_session.json` — 22 `0x5A` paired 1:1 in time with 22 `0x3D MovementResponse` events, 8 game-log "Extend view {NE|E|SE|W|N}" events, 4 sent teleports.
[^6]: State 13 documented in [[js-source-map]] §"State Machine (s field)". Handler at tpclient.pretty.js:1648-1662 dispatches `Rb` (line 1661) or `Sb` (line 1654) based on the pending scope Y and the `ga` flag.
[^7]: `src/tankpit_bot/state/scan_coverage.py:29` — "the bot teleports the viewport is fixed until the next teleport"; `src/tankpit_bot/bot/ai/hunt_mode.py:52-53` — "viewport shifting is OFF, so walking to an edge reveals no new [tiles]".
[^user]: user (Austin), 2026-07-17 — "auto shift doesnt center on bot every walk. it only shifts when the bot walks to a tile on the edge of the viewport. and then it recenters on bot btw." Reaffirmed: "when you walk to the edge, with auto scroll on, it will center the viewport on the bot."
[^8]: Viewport probe captures `runs/probe/viewport-20260725-190352.capture_session.json` (OFF-phase edge walk to column 168, out-of-window rejects at 169+, ON-phase degenerate) and `runs/probe/viewport-20260725-192738.capture_session.json` (ON-phase edge arrival at column 153 → same-tick `0x5A (145,116)`); offline pairing via `analysis_scripts/analyze_viewport_probe.py`.
[^9]: Raw short frames extracted from `key_probe.capture_session.json` (2026-07-24): autoscroll ack `4130` = ASCII `"A0"`, chat ack `4331` = ASCII `"C1"` — pre-XOR bytes, matching the plaintext `Ia`/`Ka` command encodings byte-for-byte. The inverted-restore inference: the key probe's restore press, decoded through the then-broken `== 1` test, reported OFF while actually leaving ON — consistent with viewport-probe runs -175629 through -190352 all opening ON and run -192738 (after -190352's verified OFF restore) opening OFF.
[^10]: user (Austin), 2026-07-25, rejecting the walking-traversal suggestion this page briefly carried. Step latency from the run -192738 timeline: single-tile walk commands echoed their 0x47 ~1.7-2.0 s apart.
[^11]: Mined 2026-08-01 from `runs/sniff/sniff-20260710-202821.capture_session.json` (the 421.8 s human session this page's corpus counts come from — the `latest.*` alias has since been overwritten): all 8 sent short frames XOR-decode to `03 5a <dir>` within 31 ms of their "Extend view" game-log line; window origins from the paired 0x5A, self positions walk-corrected through the 0x47 path echoes. Scratchpad miner `mine_scope_shift.py`; the law + direction table are pinned executable in `tests/sim/test_scope.py::test_anchor_law_reproduces_every_measured_shift`.

## Acceptance boundary (measured 2026-07-25)

The viewport probe paired every sent move with its response across both autoscroll states:[^8]

| Move target | Server response |
|---|---|
| Inside the current `0x5A` window | **Accepted**: `0x47` echo with a server-computed path — the server pathfinds around rock/water, up to 15 tiles observed (`path='nwwwsssswwwwnnn'`), so an accepted move is a path, not a straight line |
| Outside the current window (by even 1 column) | **Rejected**: `0x52 Supervisor err=0` (`CANT_DO`) — observed at exactly the boundary: target `x=168` (edge column) accepted, `x=169` rejected, window `(153,121)` |
| Unreachable / no path (e.g. water target) | **Rejected**: `0x52 err=1` (`CANT_GO`) |
| Re-sent while already walking to it | **Rejected**: `0x52 err=6` (`ALREADY_THERE`) |

The bound is **the window, not a self-centered radius** — the earlier "Chebyshev ≤ 8" reading from the teleport probe was the same law seen from a freshly centered window (where window edge and self ± 8 coincide).[^8]

**Walking is NOT a travel mechanism** (user ruling, verbatim, 2026-07-25: *"walking is too slow... we teleport for a reason. we walk for equipment and fuel pickups in the same viewport. but no we're not walking across the map or to enemies."*).[^10] The wire supports window-following traversal under autoscroll ON, but each step costs a full server round-trip (~2 s/tile measured in the probe runs[^8]) — crossing the map would take minutes where a teleport is instant. The bot contract stays: **teleport to travel, walk only for in-viewport pickups.** The measured laws above matter for correctness (what the server accepts and when the window moves), not as a traversal strategy.

## Sim as-built (2026-07-25, scope shifts added 2026-08-01)

The fake server enforces these laws (`sim/viewport_window.py`): the client holds a stored window set at join, teleport landings, and now client `Rb` scope commands (`apply_scope_shift` — the measured anchor law, direction 8 recentering like a landing); walking never recenters it and never re-emits 0x5A (the dynamic-layer patch refresh is event-driven — ferry/block changes); client moves and pickups outside the window reject with 0x52 code 0; extra radar covers exactly the stored 16×16 window and free radar clips to it; visibility (0x58 exits / 0x3D entries) runs on the window, including tanks a pan reveals. Two fidelity bugs the first ferry-scenario soak exposed (both fixed 2026-08-01): the sim sent `teleport_landed` BEFORE the 0x3D position update — reversed from the real wire order — which made every exact landing read as a displacement and spuriously consume ferry beliefs; and 0x5A patch entities were not sorted in patch-linear order, so a ferry riding WEST (fresh patch earlier in the walk than its revert) crashed the skip-RLE encoder. The pre-2026-07-25 sim recentered on every walk — autoscroll-ON behavior the bot never plays under.[^8]

## Rest-state law (corpus-swept 2026-07-22)

3,387 bot-session samples pairing every 0x5A origin with the self
tank's wire position: **at rest the tank sits at exactly window
offset (8, 8)** — the modal bin holds ~10× any other. The wide
dispersion (offsets across and beyond the window) decodes as CLIENT
ANIMATION LAG: server movement is instant, so the wire position
leads the on-screen walk, and the camera follows the animation. The
sim's centered-window model is therefore the rest-state truth, and
since the bot only issues actions from rest, centered is behaviorally
exact for bot play ([[physics-module-roadmap]]).
