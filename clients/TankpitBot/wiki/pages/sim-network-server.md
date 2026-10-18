---
title: Sim Network Server (the sim's rooms over WebSockets)
tags: [sim, architecture, multiplayer, protocol]
related:
  - "[[multiplayer-field]]"
  - "[[container-census]]"
  - "[[session-state-deglobalisation]]"
  - "[[recipient-policy]]"
  - "[[sim-world-parameterization]]"
source_paths:
  - "src/tankpit_bot/sim/net_server.py"
  - "src/tankpit_bot/sim/net_host.py"
  - "src/tankpit_bot/sim/net_room.py"
  - "src/tankpit_bot/sim/net_accounts.py"
  - "src/tankpit_bot/sim/net_store.py"
  - "src/tankpit_bot/sim/net_accounts_cli.py"
  - "tests/sim/test_net_store.py"
  - "src/tankpit_bot/sim/lobby.py"
  - "src/tankpit_bot/sim/transport.py"
  - "tests/sim/test_net_server.py"
  - "tests/sim/test_net_host.py"
  - "tests/sim/test_net_room.py"
  - "sim-server.compose.json"
  - "tests/sim/test_sim_compose.py"
  - "scripts/sim_host.py"
  - "tests/scripts/test_sim_host.py"
source_git_blobs:
  "sim-server.compose.json": "e69e3911b3b203e1a8f079eeb499a4aac440ea76"
  "tests/sim/test_sim_compose.py": "76445721a43ecb28ce4849de38f64e71d2356771"
  "scripts/sim_host.py": "cae7b1c5e0e39ec06cc66c0a145c55e00589a391"
  "tests/scripts/test_sim_host.py": "0dc7323b7d487ef532e506563ca9144031b9b263"
  "src/tankpit_bot/sim/net_server.py": "7e6edfe71cfe6e9c06c3bd4cc85c8c17e30383ee"
  "src/tankpit_bot/sim/net_host.py": "faaa8be06944b58216788115e4cbc7192a83ad83"
  "src/tankpit_bot/sim/net_room.py": "f242a4ba888b471670e9f58f92847b12fe18c27e"
  "src/tankpit_bot/sim/net_accounts.py": "1b08e40f5533385aad8e9c5cca88514fbdb9328d"
  "src/tankpit_bot/sim/net_store.py": "caf69b7492df5bb0d8c306b3e2ebd65978b361d7"
  "src/tankpit_bot/sim/net_accounts_cli.py": "d1e35b45f62ac8987de0bb1f492d2cd50e9116f6"
  "tests/sim/test_net_store.py": "37e21282a3bceb279091f9d161eaaf74f479bced"
  "src/tankpit_bot/sim/lobby.py": "a3c4eedb0230aca2f075c0434c8caeb54656674e"
  "src/tankpit_bot/sim/transport.py": "b84317e74e3aca75c007b1e26bff8d8b633e4b03"
  "tests/sim/test_net_server.py": "4ecff1180ba167b7f3193c102e6e3b7062a93c50"
  "tests/sim/test_net_host.py": "c16b92cc112c7a223e18231b00b9373440d8c7b0"
  "tests/sim/test_net_room.py": "dd1d028f8c670d461558eaaa28310e2047fe2d49"
provenance:
  - "Board task b008ab91 (the multiplayer track), Phase 5, 2026-10-05: the scripts/sim_control.py comparison and the tankpit-sim-serve run quoted below, both from the committed tree"
  - "2026-10-05 live persistence check: tankpit_sim created on the local platform-postgres container (host port 55432); tankpit-sim-accounts init/add/list, then a scratch WebSocket client joining field05 room 1 as account 1001 for 5 ticks; rows read back with psql from sim_sessions and sim_accounts"
  - "2026-10-06 live Traefik check on the hub: tankpit-bot:local built from clients/TankpitBot/Dockerfile, sim-server.compose.json up beside the root compose's traefik and platform-postgres, a scratch client joining as account 1003 through ws://127.0.0.1/tankpit-sim, then docker stop with the account seated; sim_sessions rows 2 and 3 read back with psql"
  - "MCPs board task ff035b0d, 2026-10-13: the hub's tankpit_sim dumped with pg_dump from platform-postgres, its two COPY blocks loaded into sedona's sim-db (COPY 3, COPY 6; accounts austin, traefik-probe, webclient), then make sim-up at API e98e5840f ran tankpit-sim:e98e5840f on sedona and its smoke read 200:65536 from /terrain/1, and make sim-down left both containers Exited (0) with restart policy no"
fact_checked: "2026-10-13"
confidence: high
hubs: [architecture]
---

# Sim network server: the sim's rooms over WebSockets

*Phase 5 of the multiplayer track (board task `b008ab91`), 2026-10-05.*

[[multiplayer-field]] put several production bots on one `SimServer` in
one process. This page puts the server behind a socket. A client in
another process, or on another machine, joins a room over a WebSocket,
plays on it and leaves. The game is TankPit-derived and the client,
name and accounts are this project's own. The server never mirrors
tankpit.com and never handles anyone's tankpit.com credentials.

## How to run it

```
poetry run tankpit-sim-serve --accounts net_accounts.json
poetry run tankpit-sim-serve --accounts net_accounts.json --room 1:field01:p --room 5:field05:n --port 8765
```

`--room ID:FIELD:MODE` repeats. Mode `p` is a practice room with the
bot_policy roster and mode `n` an open field, and any of the 44 shipped
fields plays ([[container-census]], "Playing another field"). The
default is one practice room on field01 at `127.0.0.1:8765`. `--ticks N`
stops after N ticks and `--tick-ms` shortens the wire's 2 s tick, for a
smoke run.[^1]

## The wire is the page client's

Each binary WebSocket message is one wire payload: length-prefixed
frames, plaintext in the lobby and `!`-led XOR'd commands in play. These
are the bytes the page client puts on the real socket. A connection
goes through four steps:

1. **AUTH first.** `%AUTH !be <account>|<token>|<stamp> <magic>`, read by
   `parse_auth_frame`, the inverse of the builder the in-process link
   already used. The account must be one this server issued, and the
   magic builds the connection's cipher table.
2. **Lobby.** The room list, select and enter are the archive's own
   exchange ([[session-state-deglobalisation]]). Entry now records the
   troop as well as the room (`LobbyEntry`), and the room seats a tank
   at an open tile on that team, under the account's name and rank.
3. **Play.** The client asks for its join burst with CMD_ENTER_GAME, as
   a real client does. From then on each tick sends it its batch,
   enveloped and ciphered as the real server sends it.
4. **Leave.** A quit frame is echoed and takes the tank off the field.
   A closed socket does the same. The others see the 0x29.

One function now splits a payload into commands and lobby frames
(`route_client_frames`), and both the in-process link and the network
host call it. It replaced `decode_client_payload`, a commands-only copy
that only tests still called.[^2]

## Three layers, one decision-maker

- **`NetRoom`** is one `SimServer` and its lobby row. Its world is built
  the way a field session builds one, and the practice layout's client
  spawn is taken off the field because players are seated as they
  enter. Player ids start at 2000, above the roster (500-535) and below
  churn visitors (3000), and are never reused. At most 32 players sit at
  once.
- **`NetHost`** holds the rooms and every connection. It is synchronous
  and knows nothing of sockets: a payload in, the replies out, and one
  payload per seated connection each tick. A seated connection's batch
  is never empty, because every tick carries a status sync for each
  living tank.
- **`NetServer`** maps WebSockets to host connection ids and moves
  bytes. A text message, a refused AUTH or an undecodable frame raises
  in that client's handler. The library closes that one socket with
  1011, every other connection plays on, and the handler's `finally`
  unseats the tank.[^3]

**Stopping keeps every seat.** SIGINT and SIGTERM (what `docker stop`
sends) stop the tick loop, not the process. Leaving the listener then
closes each socket and waits for its handler, so every seat still in
play is recorded before the server returns. Python's default SIGTERM
would have ended the process with those seats unrecorded.[^8]

## Accounts are this server's own, and they keep what they earn

An account record holds the SHA-256 of its token, never the token, and
both books compare digests in constant time. A wrong token and an
unknown account draw the same `SIM_LOBBY_DENIED`, so a refusal tells a
guesser nothing. One account plays on one connection at a time.[^4]

A seat that leaves, by quit frame or closed socket, is recorded. The
account rejoins at the rank and the nine decoration levels it left
with, where a session alone starts every join afresh. Awards only ever
go up, so the restored levels are never granted twice. The thresholds
for the kill and death awards still count the current session's kills
and deaths, not a career total.

Two books answer the same `AccountBookProtocol`:

- **A JSON file** (`--accounts PATH`), kept in memory for the life of
  the process. It is for tests and local play.
- **The `tankpit_sim` database** (`--database-env NAME`), which on
  sedona is the `sim-db` container beside the server (below). Until
  2026-10-13 it was a database on the hub's `platform-postgres`.
  `sim_accounts` holds one row per account and `sim_sessions`
  one row per seat that left (room, field, ticks, and the rank, kills,
  deaths and levels it left with). The account update and its session
  row commit together. The connection string is named by the variable
  that holds it, never written on a command line, and psycopg is reached
  only through the `_test_hooks.connect_database` seam.[^7]

With the sim up on sedona, an account is added inside the running
container, whose environment already names the database:

```
docker --host ssh://austi@100.95.76.122 exec tankpit-sim-sim-1 tankpit-sim-accounts add --database-env TANKPIT_SIM_DATABASE_URL --id 1004 --name <name>
```

`add` prints the new token once; the database keeps only its digest.
The server also creates any missing tables when it starts, and `init`
does the same for a database no server has opened.

## What was checked

- **The join burst is the archived shape.** Read back through the
  production capture decoder, a lone player's burst is
  `21 3E 5A 3D 2E 49 49 74 3F`, the shape [[recipient-policy]] measured
  340/340. A second player is announced to the first with a 0x28, and
  its quit with a 0x29.[^5]
- **Over a real socket.** The tests listen on an ephemeral localhost
  port and join with a real WebSocket client sending the page client's
  bytes.[^6]
- **N=1 is unchanged.** `scripts/sim_control.py` recorded the tree
  before the shared split and after it: IDENTICAL on all 21 artifacts.
- **The CLI serves.** A 20-tick run of a practice room on field01 and an
  open room on field05 served both rooms; the field05 room settled 6
  field01-placed seeds onto open ground.
- **Against the real database.** On 2026-10-05 `tankpit_sim` was created
  on `platform-postgres` (host port 55432), `init` made its tables, and
  `add` issued account 1001 at rank 2. A WebSocket client then joined an
  open field05 room with that token, played 5 ticks and quit. One
  `sim_sessions` row was written (`1001|1|field05_r.gif|5|2|0|0`), and
  the account kept rank 2.

## On the game host, started by hand

The game host owns the game servers, and since 2026-10-13 that includes
this one (MCPs board task `ff035b0d`). The game host was sedona until
2026-10-18 and is colossus, a Linux machine, since (board task
`c481c259`). It runs only when asked. The operator, 2026-10-13: "make
sure it doesnt auro stsrt or anything i csn maniallt stsrt it wheb
needed". From `clients/TankpitBot` in any checkout that can ssh to the
host:

```
make sim-up      # build HEAD's image on the host if missing, start, and see it serve
make sim-down    # stop the sim and its database; the accounts stay
```

`sim-server.compose.json` declares two services. `sim` runs
`tankpit-sim-serve` from the package image, and `sim-db` runs
`postgres:16-bookworm` with the accounts in the named volume `sim-db`,
which outlives every `down`. The sim waits for a healthy database,
publishes port 8765 on the host's loopback only, and both services carry
`restart: "no"`, so a reboot of the host or of its Docker leaves them off.
The file is JSON, which compose reads as it reads YAML, so the package's
own JSON decoders test it against the CLI: its command must parse as the
server's flags and name the rooms it opens.[^9]

`scripts/sim_host.py` is what the two targets run. `up` refuses a host
without its own `secrets.env`, the database password no repository
holds, in `/home/user/tankpit-sim`; stages the compose file as committed
at HEAD and a `sim.env` naming the image there; pulls the database image
on the host; builds `tankpit-sim:<HEAD's short sha>` on the host from
HEAD's tree, unpacked there by `scripts/fleet_release.py`'s `send_head`
and removed after the build, only when the host lacks that tag; starts
compose with both env files; and asks the host's loopback for room 1's
terrain until the whole 65,536-byte field arrives. An `up` that exits 0
has been seen serving.[^10]

That smoke failed twice on its first day on sedona, both times on
quoting rather than on the sim: ssh joined its remote arguments with
spaces for sedona's `cmd.exe`, and the hub's MSYS ssh brace-expanded an
unquoted argument. On colossus every remote command is one
`shlex.join`-quoted line read by the host's shell, so the format reaches
curl as one word.

The accounts moved with the host on 2026-10-18: sedona's `sim-db` volume
was exported with `tar` (sha256 `f9dbdadf…` on both ends, 48,640,000
bytes), kept in `/home/user/tankpit-archive/sedona-2026-10-18/sim-db.tar`
on colossus, and restored into a volume labelled for the `tankpit-sim`
compose project, whose 1,279 files `diff -r` read identical to the
archive's.

Checked live on 2026-10-13: the three accounts and six sessions from the
hub's database were loaded into `sim-db`, `make sim-up` at API
`e98e5840f` ran `tankpit-sim:e98e5840f` and read `200:65536`, and
`make sim-down` left both containers exited with restart policy `no`.

Before the move it ran on the hub behind the API root compose's Traefik
at `ws://127.0.0.1/tankpit-sim`, checked live on 2026-10-06: a client
joined account 1003 there, and `docker stop` with the account seated
closed its socket with 1001, printed `32 ticks served`, exited 0 and
wrote the seat as a row of its own.

## The same port serves the browser client

A plain HTTP `GET` on the server's port gets the play page, its
modules, a room's terrain and the static key; an upgrade gets the game.
[[sim-web-client]] describes both halves. `--web-root` names the built
page, and the image passes `/app/web`.

## Not done here

The production bot still reaches a server through a browser page. A
bot speaking this socket directly is the next client.

[^1]: `src/tankpit_bot/sim/net_server.py`, `parse_serve_args`, `build_host` and `main`; `tests/sim/test_net_server.py`, `test_the_command_line_serves_its_rooms_for_its_ticks` and `test_every_flag_is_read`.
[^2]: `src/tankpit_bot/sim/lobby.py`, `parse_auth_frame` and `SimLobby._enter`; `src/tankpit_bot/sim/transport.py`, `route_client_frames`.
[^3]: `src/tankpit_bot/sim/net_room.py`, `NetRoom.seat` and `open_field_room`; `src/tankpit_bot/sim/net_host.py`, `NetHost.receive` and `NetHost.tick`; `src/tankpit_bot/sim/net_server.py`, `NetServer.handle`; `tests/sim/test_net_server.py`, `test_a_text_message_ends_only_that_connection`.
[^4]: `src/tankpit_bot/sim/net_accounts.py`, `MemoryAccountBook.verify` and `decode_net_account`; `tests/sim/test_net_host.py`, `test_a_wrong_token_is_denied` and `test_an_account_joins_once_at_a_time`.
[^5]: `tests/sim/test_net_host.py`, `test_enter_game_is_answered_with_the_join_burst_on_the_next_tick` and `test_a_second_player_is_announced_to_the_first_and_its_quit_too`.
[^6]: `tests/sim/test_net_server.py`, `test_a_client_joins_and_plays_over_a_real_socket`.
[^7]: `src/tankpit_bot/sim/net_store.py`, `PostgresAccountBook`, `SCHEMA` and `connect_store`; `src/tankpit_bot/sim/net_room.py`, `NetRoom.leave`; `src/tankpit_bot/sim/net_host.py`, `NetHost._unseat`; `tests/sim/test_net_host.py`, `test_a_seat_that_leaves_is_recorded_and_the_account_rejoins_as_it_left`; `tests/sim/test_net_store.py`, `test_a_seat_updates_the_account_and_adds_a_session_in_one_commit`.
[^8]: `src/tankpit_bot/sim/net_server.py`, `serve_rooms` and `NetServer.tick_for`; `tests/sim/test_net_server.py`, `test_a_signal_stops_the_server_after_the_tick_in_play` and `test_closing_the_listener_records_every_seat_still_in_play`.
[^9]: `sim-server.compose.json`, the `sim` and `sim-db` services; `tests/sim/test_sim_compose.py`, `test_the_command_serves_from_the_database_beside_it`, `test_it_publishes_the_port_the_server_binds_on_loopback_only`, `test_neither_service_starts_on_its_own`, `test_the_accounts_live_in_a_named_volume` and `test_the_sim_waits_for_a_healthy_database`.
[^10]: `scripts/sim_host.py`, `stage`, `ensure_image`, `smoke`, `up` and `down`, and `scripts/fleet_release.py`'s `send_head`; `tests/scripts/test_sim_host.py`, `test_up_stages_pulls_builds_composes_and_smokes`, `test_ensure_image_builds_heads_tree_on_the_host_and_removes_it`, `test_smoke_hands_the_curl_format_to_the_host_as_one_word` and `test_down_stops_the_sim_and_its_database`.
