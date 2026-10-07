---
title: Sim Web Client (the browser plays the sim server)
tags: [sim, architecture, multiplayer, protocol, typescript]
related:
  - "[[sim-network-server]]"
  - "[[sim-renderer]]"
  - "[[terrain-system]]"
  - "[[viewport-shift-protocol]]"
  - "[[movable-blocks]]"
source_paths:
  - "web/src/wire.ts"
  - "web/src/lobby.ts"
  - "web/src/envelope.ts"
  - "web/src/tile_messages.ts"
  - "web/src/commands.ts"
  - "web/src/field.ts"
  - "web/src/world_view.ts"
  - "web/src/wire_client.ts"
  - "web/src/play.ts"
  - "web/play.html"
  - "web/tests/sim_session.spec.ts"
  - "web/tests/envelope.spec.ts"
  - "web/tests/wire_client.spec.ts"
  - "src/tankpit_bot/sim/net_web.py"
  - "tests/sim/test_net_web.py"
  - "tests/sim/test_web_session.py"
source_git_blobs:
  "web/src/wire.ts": "ca639efc9969d803e417b17ee87bcda2ac63b3f7"
  "web/src/lobby.ts": "b80a213c2c6d2ecd47973b65f6291a237e068b4c"
  "web/src/envelope.ts": "c58958f61fcdf7a2825eb9ab8e97a45474b3a206"
  "web/src/tile_messages.ts": "d419e2fa258d1121974d22fede705d296d09cdb8"
  "web/src/commands.ts": "767f79331007edb3398b84ad4509d95cce17ceb7"
  "web/src/field.ts": "8bf5aaa457a82290085a0b118f56b5bbc716377d"
  "web/src/world_view.ts": "daa281b1c2af18ca0e4a0cf33b7e4c2b2b8c6479"
  "web/src/wire_client.ts": "39548576d6aa9f597bdb041331f414011cb3f17c"
  "web/tests/wire_client.spec.ts": "64cf3a72d834dfc9fdec0e39dd9d82278a2bae07"
  "web/src/play.ts": "97eb366210429caab036b84003e9d2b89f641899"
  "web/play.html": "e6a2c589db317bbbfae7f069d3030569b9dbd08a"
  "web/tests/sim_session.spec.ts": "24093fc1962c93fe55361d8e78d0c066ed6fc945"
  "web/tests/envelope.spec.ts": "7ebadb7223e1dff5aed916dc2ad76942b0cc3bde"
  "src/tankpit_bot/sim/net_web.py": "72c89c11d7afdb9f1e0c6b5b8f86d093ddc83481"
  "tests/sim/test_net_web.py": "3e694df44092a1318753ea5213e4ab27d9c8be81"
  "tests/sim/test_web_session.py": "df126d4952efe598b863035a496340badee99354"
provenance:
  - "Board task b008ab91 (the multiplayer track), Phase 5, 2026-10-06: tankpit-sim-serve --web-root web on 127.0.0.1:8795 with one practice room at 500 ms ticks, joined from Playwright's Chromium at device scale 2 as account 1001"
  - "2026-10-06 live hub check through Traefik: tankpit-bot:local built from clients/TankpitBot/Dockerfile and started with sim-server.compose.json (container tankpit-sim-sim-1, rooms 1 and 5, 2 s ticks); account 1006 issued with tankpit-sim-accounts in the tankpit_sim database; Playwright's Chromium at device scale 2 opened http://127.0.0.1/tankpit-sim"
fact_checked: "2026-10-06"
confidence: high
hubs: [architecture]
---

# Sim web client: the browser plays the sim server

*Phase 5 of the multiplayer track (board task `b008ab91`), 2026-10-06.*

[[sim-renderer]] draws the sim; this page is the client that feeds it.
A browser opens the play page, joins a room of `tankpit-sim-serve`
over its WebSocket, reads each tick's batch, and paints it. A click on
the field moves the tank. It is our own client on our own server, with
our own accounts. The web package carries nothing from tankpit.com: the
static key and the field reach the browser from our server at run time.

## The server hands out the client

`tankpit-sim-serve` answers plain HTTP `GET`s on its WebSocket port,
so one Traefik route carries the game and its client. A request that
asks to upgrade goes to the handshake; every other request is
answered by `net_web.py`:[^1]

| Path | Answer |
|---|---|
| `/` | `play.html` from `--web-root` |
| `/dist/<module>.js` | one built module |
| `/terrain/<room>` | 65,536 class bytes, row by row: 0 ground, 1 rock, 2 water |
| `/cipher-key` | the static key every connection's table is built from |

The terrain is read off the room's own `TerrainMap`, the terrain the sim
moves tanks over. The browser does not classify the field image
itself. The game's client samples its map's colours
([[terrain-system]]); a second classification in TypeScript could
disagree with the server about which tiles are water.

A server started without `--web-root` still serves the terrain and the
key. Asked for the page, it says so by name (`SIM_WEB_NO_ROOT`).

## The client speaks the page client's wire

- **Frames.** One WebSocket message is a run of frames, each with a
  two-byte little-endian length (`wire.ts`).
- **Lobby.** AUTH carries our account, its token and a random session
  magic. The client selects its room from the room list, enters on its
  troop after the join confirm, and sends enter-game after the enter
  response. A reply out of that order is refused (`LOBBY_ORDER`), as is
  play before entry (`WIRE_PHASE`).[^2]
- **Play.** Every frame of play is 0x2E and a XOR'd body. The table is
  the static key with the magic folded in, wrapping for long bodies.
  The subtype alone does not name a message: the production decoder
  also asks the body's shape. `envelope.ts` asks the same shapes, so a
  body with another message's shape goes to `unrendered` by the same
  rule.[^3]
- **Commands.** Enter-game and move are `!` and the ciphered type, code
  and tile (`commands.ts`).

What the client draws from: 0x21 and 0x3E (team), 0x28 and 0x29, 0x3D
(a tank on a tile), 0x47 (a walk; the tank faces along the last step),
0x58, 0x41 (a corpse), 0x5A (the window and its patch), and 0x43, 0x4F
and 0x4A (caches, mines and rocks). Each was checked against bodies the
sim's own `encode_envelope_body` wrote.[^4]

## The world view paints what changed

`WorldView` holds what the client knows:

- the window, which only 0x5A moves ([[viewport-shift-protocol]]);
- the rock, cache and mine layers over the static terrain. The wire's
  dynamic terrain values map to rocks: 1 rock A, 2 and 3 rock B, 5 and
  7 ferry, 0 none ([[movable-blocks]]); any other value is refused;
- every tank: its team, tile, facing, and whether it is a corpse.

A tile's terrain byte comes from its class and its diagonal
neighbours, the contract `terrain.ts` decodes. A neighbour off the
field counts as sharing. `paint()` sets every window tile when the
window moves, and otherwise only the tiles written since. It places,
moves or removes only the tanks that changed. A tank is drawn while it
is placed, its team is known, and it is inside the window.[^5]

## Proof against the server's own bytes

`tests/sim/test_web_session.py` records one whole session from a real
practice-room `NetHost`. That is AUTH, select, enter, enter-game, a
move and three ticks, ciphered with a key of the test's own. It
requires `web/tests/sim_session.ts` to be exactly that recording.
`sim_session.spec.ts` plays it through the client, which must send the
recorded bytes exactly and draw the tank where the server left it.
Neither side can change its bytes without failing the other's
suite.[^6]

## Running it

```
cd clients/TankpitBot/web && npm run build
cd .. && poetry run tankpit-sim-serve --accounts accounts.json --web-root web --room 1:field01:p
# open http://127.0.0.1:8765/, enter an account and token from tankpit-sim-accounts
```

In the image, the Dockerfile's `web` stage builds `dist/` and copies it
with `play.html` into `/app/web`. `sim-server.compose.json` passes
`--web-root /app/web`. Traefik redirects the bare `/tankpit-sim` to
`/tankpit-sim/` and then strips the prefix: the page loads its modules,
key, terrain and socket relative to its own URL, which must end in a
slash.

Checked live on 2026-10-06, locally on port 8795:
- The page, `dist/play.js` and `terrain/1` (65,536 bytes) answered 200.
- Chromium at device scale 2 joined account 1001 into room 1 and
  reached `playing, last frame 256 tiles and 1 tanks`. It had six
  canvases (768x512, menu 768x96) and no page error.
- A click on window tile (10, 5) moved the tank's drawn pixels from
  tile (8, 8) to (10, 5).

Checked live on the hub through Traefik on 2026-10-06, in the image
`sim-server.compose.json` starts:
- `/tankpit-sim` answered 301 to `/tankpit-sim/`. The page, `dist/play.js`,
  `terrain/1`, `terrain/5` and `cipher-key` each answered 200.
- Chromium opened the bare `/tankpit-sim`, joined as account 1006 (issued
  by `tankpit-sim-accounts` in `tankpit_sim`), and reached `playing,
  last frame 256 tiles and 1 tanks` with no page error. A click on
  window tile (10, 5) moved the tank there from (8, 8).
- The first hub run found a fault the local one could not. The hub lists
  rooms 1 and 5; the client selects room 1 as soon as its row arrives,
  so room 5's row reached a client already in `selected`. That was a
  `LOBBY_ORDER` error on every join. Rows after a select are now kept as
  the list, and a test holds that order.[^7]

## Not done here

- Shots (0x53), radar sweeps and the map overlay are not drawn yet:
  they fall to `unrendered`, which `WorldView` counts by subtype.
- The client sends enter-game and move only; other commands (shoot,
  radar, mine, scope) are the next pieces.

[^1]: `src/tankpit_bot/sim/net_web.py`, `WebPages.answer` and `terrain_classes`; `tests/sim/test_net_web.py`, `test_the_page_and_its_modules_come_from_the_web_root` and `test_a_server_without_a_web_root_serves_no_page_but_still_its_terrain`.
[^2]: `web/src/wire_client.ts`, `WireClient.open`, `WireClient.message` and `WireClient.room`; `web/src/lobby.ts`, `readLobbyFrame`.
[^3]: `web/src/envelope.ts`, `readEnvelope`; `web/tests/envelope.spec.ts`, `leaves to unrendered a subtype it does not draw from, and one whose body has another message's shape`.
[^4]: `web/tests/envelope.spec.ts`, `hands each tank message the server writes to its method`; `web/src/tile_messages.ts`, `decodeViewport` and `decodeRadarScan`.
[^5]: `web/src/world_view.ts`, `WorldView.paint` and `tileKey`; `web/src/field.ts`, `FieldTerrain.terrainByte`.
[^6]: `tests/sim/test_web_session.py`, `record_session` and `test_the_fixture_is_the_session_the_server_plays`; `web/tests/sim_session.spec.ts`, `is sent byte for byte, read whole, and drawn where the server left the tank`.
[^7]: `web/src/wire_client.ts`, `WireClient.room`; `web/tests/wire_client.spec.ts`, `keeps the rows listed after its own room without selecting again, and refuses one after the confirm`.
