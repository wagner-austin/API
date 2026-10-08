---
title: Multiplayer Field (several production bots on one sim server)
tags: [sim, architecture, multiplayer, combat, testing]
related:
  - "[[physics-module-roadmap]]"
  - "[[recipient-policy]]"
  - "[[session-state-deglobalisation]]"
  - "[[sim-world-parameterization]]"
  - "[[enemy-bot-behavior]]"
source_paths:
  - "src/tankpit_bot/sim/field_run.py"
  - "src/tankpit_bot/sim/field_clients.py"
  - "src/tankpit_bot/sim/server_sessions.py"
  - "tests/sim/test_field_run.py"
  - "scripts/sim_control.py"
  - "src/tankpit_bot/sim/field_choice.py"
  - "src/tankpit_bot/sim/cli_args.py"
source_git_blobs:
  "src/tankpit_bot/sim/field_run.py": "1bdf096f5e4c32ea8ba64535178a6e5caea06340"
  "src/tankpit_bot/sim/field_choice.py": "cbfcc15d038057798795875d393e6149a0d23f2a"
  "src/tankpit_bot/sim/cli_args.py": "179ec5e69c3d9ea92b2623f5cac78b2f1ea9565b"
  "src/tankpit_bot/sim/field_clients.py": "73987f5fc24a998c7bd9abbfd42fea0b1358b3be"
  "src/tankpit_bot/sim/server_sessions.py": "b75b71414d6ff7422c934739e230f56522c37fe2"
  "tests/sim/test_field_run.py": "f44c7d8967963ee52160ea711f8f9d919916f282"
  "scripts/sim_control.py": "a0f2adf52b4b2229885abb4d5a34f4191f244dbc"
provenance:
  - "Board task b008ab91 (the multiplayer track), Phase 2, 2026-10-05: the live field runs on field01 quoted below, played from the committed tree into a scratch archive"
fact_checked: "2026-10-05"
verified: 2026-10-08 (code re-read against API HEAD for MCPs board task 3d71a8e1; field_run.py gained only the field parameter)
confidence: high
hubs: [architecture]
---

# Multiplayer field: several production bots on one sim server

*Phase 2 of the multiplayer track (board task `b008ab91`), 2026-10-05.*

A one-bot sim session fights the scripted opponent, the practice
roster or a recorded ghost, and none of those is a second real client:
nothing fires the bot's own combat machinery back at it, nothing owns a
connection whose receipts must NOT reach the first bot, and nothing
quits the room mid-fight. Solo practice-room runs were blind to exactly
those wire patterns ([[recipient-policy]]). Phase 1 gave `SimServer`
any number of connections ([[physics-module-roadmap]], "The registry,
the fan-out and mid-field join"); this page is what plays on them.

## How to run it

```
poetry run tankpit-sim-run --clients 2 --layout bot-20260706-223721 --population-seed 7
poetry run tankpit-sim-run --clients 3 --practice --rounds 300
```

`make sim-field ARGS=...` is the first form with two bots.
`--clients N` (2 to 8) seats N production bots; `--practice` puts them
among the bot_policy roster instead of alone in the arena. Each bot's
capture lands at `<out>/sim-<stamp>-tank-<id>.capture_session.json`,
its events under `<runs-root>/tank-<id>/probe/`, and the field's final
world once at `<out>/sim-<stamp>.world.json`. The ghost, ferry, larder
and atlas-forage scenarios, and `--human-opponent`, are written around
one client and are refused with `SIM_FIELD_SCENARIO`.[^1]

Since `b54cefd67` (2026-10-05, Phase 4 of the same track, repo API) a
field can also be played on any of the shipped field minimaps:
`--field NAME` reaches `run_field_session`'s `field` parameter, which
sets the world's terrain through `field_choice.resolve_field` and
otherwise keeps field01 (`SIM_FIELD`). An unknown name is refused with
`SIM_FIELD_UNKNOWN`.[^8]

## Who sits at the field

The primary client is tank 9, seeded exactly as a one-bot session seeds
it. Rival `k` takes id `9 + k` on the first open tile 8 to 14 tiles
(Chebyshev) from it, so both see each other from the join window, and
MIRRORS it — rank, fuel, stocks and enabled slots — so a duel measures
the bot against itself, not against a handicap. Sides alternate: the
first rival plays team 1 (the scripted opponent's team), the next the
primary's own team, so every bot has an enemy and, from three up, an
ally. Rivals keep the practice-bot name shape `red-<id>`; a human-shaped
name would sit behind the consent gate and never be engaged first. In
the arena the scripted opponent is removed — the rivals are the
opposition.[^2]

## What keeps the bots apart in one process

- **A contextvars context per bot.** The runtime logging's active run
  and the tick's runtime context are `ContextVar` slots
  ([[session-state-deglobalisation]]), so each bot is configured and
  ticked inside its own copied context and writes its own event stream.
- **A connection per bot.** Each link queues under its own tank and
  receives only its own batch from `advance_tick`.
- **A bot that exits leaves.** The production exit raises
  `SessionExitError`; the bot sends its graceful quit and
  `SimServer.disconnect` takes the tank off the field. The rest see the
  0x29 a departing churn visitor draws — one builder,
  `wire_statements.exit_statement`, for both — and the field forgets
  the tank's queue, its outbox batch, its unbilled firing costs, its
  corpse window and its place in every other connection's viewport.
  Its kills and deaths stay on the record.[^3]

Two server rules changed with it, both inert at one connection, which
the N=1 control confirms (below):

- **A joiner is announced only to connections already in the room.**
  `connect` used to broadcast the 0x28 to every connection; a connection
  not yet sent its join burst learns the room from that burst, so a
  second 0x28 there was a duplicate. Seating all bots before any join
  burst made each bot believe its rival stood at `(0, 0)`, because an
  entry carries no position — the first field run refused the rival as
  `no_standoff_landing` for exactly that reason.
- **A connection told of a joiner forgets it from its view**, so a
  joiner already in the window is re-placed by the end-of-tick
  membership pass with a 0x3D, after its positionless 0x28.[^4]

The practice roster learns hits from the 0x53s the connections saw. With
several connections one shot is narrated to each observer, so the
batches are merged by greatest count: a shot seen by two connections is
one hit, a dual's two identical 0x53s stay two. With one connection the
merge is that connection's batch.[^5]

## N=1 is unchanged

`make sim-control` (`scripts/sim_control.py`) recorded the tree with
all of the above and compared it with the Phase 1 tree: IDENTICAL on all
21 artifacts (six scenarios and the ghost self-replay of
`bot-20260802-205105`, 150 rounds each).[^6]

## What the first live fields showed (field01, layout `bot-20260706-223721`, seed 7)

- **Arena, two bots, 300-round budget.** Real fire crossed: bot 9 saw
  bot 10 in view and landed six dual shots in six ticks. Bot 10
  teleported adjacent, scanned, took fire, then its engagement model
  concluded *"engagement with red-9 unwinnable at any fuel (needs 1305,
  capacity 1100)"*, blocked red-9, foraged, and exited
  `no_viable_targets` at round 8; bot 9, now alone, exited the same way
  at round 9. **A finding about the bot, not the sim:** against a
  mirror of itself at rank 1 the production risk model refuses the
  fight, because the projected cost of the kill exceeds a full tank.
  Two bots of equal strength therefore never finish a duel; the first
  to run the numbers leaves.
- **Practice room, two bots among the roster, 300 rounds.** Both played
  every round and neither scored a kill; each spent its time collecting
  (about 300 actions and 175 pickups apiece).

The same field over a socket, for clients in other processes, is
[[sim-network-server]].

The deterministic version of the arena run is pinned in the test suite
on all-ground terrain, where the same sequence happens on schedule:
bot 9 fires at bot 10, tank 10 leaves at round 8, tank 9 reads its 0x29,
and the field ends at round 10.[^7]

[^1]: `src/tankpit_bot/sim/run.py` `_main_field`; `src/tankpit_bot/sim/cli_args.py` `_apply_valued_flag` (`--clients`).
[^2]: `src/tankpit_bot/sim/field_clients.py` `seed_field_rivals`, `field_account`, `RIVAL_TEAM`, `RIVAL_RING_MIN`, `RIVAL_RING_MAX`, `MAX_FIELD_CLIENTS`; `src/tankpit_bot/sim/field_run.py` `make_field_world`.
[^3]: `src/tankpit_bot/sim/field_run.py` `_open_seat`, `_leave`, `_play_round`; `src/tankpit_bot/sim/server_sessions.py` `disconnect`; `tests/sim/test_server_departures.py`.
[^4]: `src/tankpit_bot/sim/server_sessions.py` `connect`; `tests/sim/test_server_connections.py` `test_a_connection_still_joining_is_not_told_of_an_arrival` and `test_a_joiner_in_view_is_placed_again_after_its_positionless_entry`.
[^5]: `src/tankpit_bot/sim/practice_room.py` `note_field_batches`; `tests/sim/test_field_run.py` `test_one_shot_seen_by_two_connections_is_one_hit`.
[^6]: `scripts/sim_control.py`, run 2026-10-05: `IDENTICAL: all 21 artifacts` comparing the Phase 1 recording with this tree's.
[^7]: `tests/sim/test_field_run.py` `test_the_two_bots_fight_each_other` and `test_a_bot_that_exits_leaves_the_field_and_its_rival_is_told`.
[^8]: `src/tankpit_bot/sim/field_run.py:298-342`, `run_field_session(..., field: str | None = None)` and its `world["field"] = SIM_FIELD if field is None else resolve_field(field)` at `:342`; `src/tankpit_bot/sim/field_choice.py:41-59`, `resolve_field`, raising `FieldChoiceError` with `SIM_FIELD_UNKNOWN`; `src/tankpit_bot/sim/cli_args.py:145`, the `--field` token.
