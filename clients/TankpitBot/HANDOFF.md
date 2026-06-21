# Bot Reliability + Observability Handoff Brief

**Audience:** A second AI engineer who has not been part of the prior conversation.
**Date:** 2026-06-20
**Owner of remaining work (NOT yours):** the other Claude session.

## Read these first

1. `CLAUDE.md` (project root) — strict coding standards: no Any/cast/type:ignore/.pyi/noqa, `_test_hooks.py` DI exclusively, 100% statement + branch coverage, no fallbacks/back-compat/legacy code, no mocks in tests.
2. `wiki/index.md` — wiki is the single source of truth for game mechanics, wire protocol, architecture decisions.
3. `wiki/pages/coding-standards.md` — full standards list.

## Environment

- Windows 11, git-bash (Unix paths in Bash, Windows paths in PowerShell).
- Python 3.11 via poetry. Tests via `make check`.
- For `make check` capture use PowerShell: `make check 2>&1 | Select-Object -Last 100`.
- Static XOR key at `xor_static_key.txt` (read with `tankpit_bot.capture.xor.load_xor_static_key`).

## What was just delivered (locked in)

- `src/tankpit_bot/sniffer/world_state_dispatch.py` — `_dispatch_map_data` now calls `ws.mark_map_data_processed()` after applying tank observations. This fixes the open-close-map loop in the bot.
- `tests/sniffer/test_world_state_dispatch_tank.py::TestDispatchMapData::test_map_data_marks_action_complete` — locks the fix in.
- `make check` passes at 3866 tests, 100% coverage.

## DO NOT TOUCH

Pending work owned by the other Claude session. Hands off:

- `src/tankpit_bot/sniffer/world_state_dispatch.py` line where `_dispatch_map_data` sets `is_wire_sourced=False, storage_source="world_state"`. The wire-presence semantic is under review (stationary practice-room bots fail the kill gate; design question is whether MapData should count as wire-truth).
- `src/tankpit_bot/bot/ai/combat_strategy.py` — kill gate logic (`is_wire_present`, `is_position_fresh`, `_combat_shoot`).
- `src/tankpit_bot/bot/ai/threats.py` — wire-presence TTL constants and `analyze_threats`.
- `tests/replay/test_ghost_wire_presence_regression.py` and `tests/replay/fixtures/ghost_map_refresh_wire_silent.capture_session.json` — being re-evaluated; current assertions may invert.

If you find yourself touching any of those files, stop and flag it back to the user.

## Your work — three tiers

### Tier 1 — `make smoke` health gate + first integration tests

**Spec:** `make smoke` must run the bot for 30 seconds (`$env:TANKPIT_BOT_SESSION_SECONDS = '30'; poetry run tankpit-bot`), then assert against `runs/bot/latest.events.jsonl`:

1. Login completed: a STATE event recorded `INITIALIZING -> WAITING_FOR_POSITION -> IDLE` in order.
2. At least one `map_open` action cleared via `signal=map_data_processed` (look in `emit_wire_complete` events).
3. At least one tick where HUNT scored a non-zero target — `combat_target_x` or `combat_target_y` non-zero in some `emit_ai` event whose text starts with `HUNT score=`.
4. At least one action attempted (a `WIRE: shoot_at` or `WIRE: teleport_to` event, or `combat_shoot`/`combat_teleport` diagnostic).
5. Zero `signal=stall_timeout` events in the first 10 seconds.

Exit 0 only if all pass. On failure, print which assertion failed and dump the surrounding 10 events from JSONL.

Add the target in `Makefile`. The runner can live in `scripts/smoke.py` so it's reusable.

**Integration tests** (in `tests/integration/`, run as part of `make check`):

- `test_map_open_clears_on_map_data` — drives `dispatch_world_state_update` with a `MapDataDict`, asserts `check_and_clear_map_data_processed()` returns True. (Mirror of `test_map_data_marks_action_complete` but at the integration boundary using the replay engine. Locks in today's fix at one more layer.)
- `test_hunt_acquires_wire_confirmed_enemy` — replays a fixture where a `0x21 TankInfo` then `0x3D MovementResponse` arrives for a non-self tank; asserts `analyze_threats` returns the tank.
- `test_combat_fires_when_gates_pass` — both wire-presence and position-fresh gates pass; asserts `_combat_shoot` emits a fire command (do NOT modify `_combat_shoot` logic; only assert against current behaviour).
- `test_combat_blocks_on_wire_stale_target` — `last_wire_seen_ms` is older than `WIRE_PRESENCE_TTL_MS`; assert block-and-replan fires.
- `test_refuel_triggers_below_threshold` — self_state fuel below `fuel_low_threshold`; assert RECOVER_FUEL mode is selected (see `tankpit_bot/bot/ai/recover_fuel_mode.py`).

Each fixture must be sourced from real corpus data under `runs/bot/*.capture_session.json`, NOT synthesised. The corpus has 130+ sessions.

### Tier 2 — additional integration tests

Once Tier 1 is green, add coverage for:
- `test_stall_timeout_replans_to_idle`
- `test_teleport_landed_clears_action`
- `test_radar_scan_returns_to_idle`
- `test_mine_placement_updates_world`
- `test_combat_hit_advances_damage_state`

Each follows the same pattern: drive `dispatch_world_state_update` (or the relevant entrypoint) with a real-corpus-sourced message, assert observable state change.

### Tier 3 — observability builds (independent, build in this order)

1. **`make debug-run [DURATION=30]`** target. Runs the bot for `$DURATION` seconds, then auto-runs `make analyze` and prints a tick-by-tick timeline from `runs/bot/latest.events.jsonl` (one line per `STATE`, `WIRE`, `DIAGNOSTIC` event with timestamp + tick_n once Tier 3.2 is done).

2. **Auto-enrich events with self-context**. In `src/tankpit_bot/runtime_logging.py`, every `emit_*` call must auto-include `tick_n`, `bot_state`, `in_flight_action_kind` from a thread-local context. The context is set in the tick loop (`tankpit_bot/bot/tick_loop.py`). Don't break any of the existing 3866 tests — the additional fields go in `RuntimeEventRecordDict.fields` (not the reserved keys).

3. **`runs/bot/_index.tsv`** — one row per run, updated atomically after each session ends. Columns: `stamp`, `duration_s`, `exit_reason`, `ticks`, `stalls`, `shots_fired`, `kills`, `kills_per_min`. Plus a `bot-runs list / find / show STAMP` CLI in `src/tankpit_bot/diagnostics/` for querying.

4. **Signal-safe writes**. Register `signal.SIGINT` and `signal.SIGTERM` handlers in the bot CLI entrypoint that flush the events JSONL, write a partial `latest.summary.txt` with `exit_reason=interrupted`, and append to `_index.tsv`. Tests with monkeypatch-free DI (use `_test_hooks`) to verify the path.

5. **`bot-query` CLI** with named queries: `timeline`, `stalls`, `action-spans`, `target-decisions`. Wraps the JSONL reads we'd otherwise do by hand.

### Mechanical task — #71 Delete 6 dead container types

Empirically verified dead via two pieces of evidence:
- Corpus sweep of 150 sessions / 48,304 0x2E bodies: 0 fires.
- Live 5-min `make run` 2026-06-20: 0 fires.

Types to delete:

| Type | Subtype | Length | Evidence |
|---|---|---|---|
| `PositionUpdateDict` | 0x24 | 13 B | superseded by 0x3D MovementResponse (4197 corpus samples take the 13-byte slot) |
| `DeactivationDeathDict` | 0x43 | 7 B | 7-byte 0x2E bodies all route to 0x49/0x67/0x4A/0x4F via tunneled dispatch |
| `TankLeaveDict` | any | 6 B | 6-byte 0x2E bodies route to 0x74 EquipmentToggle or 0x4F CombinedTileUpdate |
| `TankRegistryDict` | any | 16–20 B | superseded by 0x21 TankInfo (5143 corpus samples eat all 16/17/19-byte bodies) |
| `PlayerListShortDict` | 0x79 | 4 B | bot never sends `/` query; 4-byte 0x2E bodies are 0x44 FuelGain or 0x52 SupervisorText |
| `PlayerListExtendedDict` | 0x79 | 7 B | same; never fires |

Files to touch (do these in order, run `make check` after each to catch breakage):

1. `src/tankpit_bot/container/types.py` — delete the 6 TypedDicts, their `ContainerMessageType` enum entries, their `MESSAGE_TYPE_LEVELS` entries, drop them from the `ContainerMessage` union, drop from `__all__`.
2. `src/tankpit_bot/container/decoders/misc.py` — delete PlayerListShort/Extended decoders + structure checks.
3. `src/tankpit_bot/container/decoders/tank.py` — delete TankLeave + TankRegistry decoders + structure checks + DIRECTION_/SUBTYPE_ constants if only used there (grep first).
4. `src/tankpit_bot/container/decoders/position.py` — delete the file entirely (only contained PositionUpdate).
5. `src/tankpit_bot/container/decoders/combat.py` — delete DeactivationDeath decoder + structure check.
6. `src/tankpit_bot/container/decoders/__init__.py` — remove imports, dispatcher arms, `__all__` entries.
7. `src/tankpit_bot/container/identification.py` — remove `_identify_player_list_type`, `_identify_deactivation_type`, the dispatch arms in `identify_container_type`.
8. `src/tankpit_bot/container/__init__.py` — remove public re-exports.
9. `src/tankpit_bot/sniffer/world_state_dispatch.py` — remove the `tank_leave`, `deactivation_death`, `tank_registry` (× 2) `match` arms.
10. `src/tankpit_bot/sniffer/world_state_dispatch_position.py` — remove `position_update` `match` arm.
11. `src/tankpit_bot/sniffer/formatters.py` — remove the `tank_registry` and `position_update` `match` arms (and any helpers only used by them — `handle_tank_registry`, `format_position_update`).
12. `src/tankpit_bot/sniffer/world_state_tanks.py` — delete `update_world_state_from_tank_registry`.
13. `src/tankpit_bot/sniffer/world_state_containers.py` — delete `update_world_state_from_tank_registry_container`.
14. `src/tankpit_bot/capture/viewport_analysis.py` — delete `_handle_position_update` and `position_update_count`.
15. `src/tankpit_bot/sniffer/constants.py` — remove the corresponding `MSG_TYPE_NAMES` entries.
16. Tests: `tests/container/test_data.py`, `test_misc.py`, `test_world_decoders.py`, `test_structure.py`, `test_dispatcher.py`, `test_tank_decoders.py`, `test_combat_decoders.py`, `test_status_decoders.py`; plus `tests/sniffer/test_world_state_dispatch_tank.py`, `test_world_state_dispatch_other.py`, `test_world_state_dispatch_container.py`, `test_world_state_dispatch_movement.py`, `test_formatters_details.py`; plus `tests/capture/test_signature.py`, `test_protocol_census.py`. Remove tests for the deleted decoders/dispatchers — do NOT just delete the assertions, delete the whole test method/class for any test that's exclusively about a deleted type.
17. `wiki/pages/decode-coverage.md` — already updated by the prior session to remove the rows. Double-check.

After full deletion run `make check`. Coverage must stay at 100%.

### Mechanical task — #73 Promote ShootEvent `unk1`/`unk2` → `aim_x`/`aim_y`

JS source (`tpclient.pretty.js` lines 4082–4097 for `Gg.h` and 2980–3088 for `yf`) confirms:
- `a[7]` and `a[8]` are passed to the projectile-animation constructor `yf` as `z` and `O`.
- Inside `yf`, `this.qa = 24 * z + 12` and `this.ta = 16 * O + 8` are PIXEL centres of the tile the tank's gun is aimed at after firing.
- `yf.start()` uses `atan2(this.h - this.qa, this.ta - this.i)` to set the tank's facing direction.

So `a[7]/a[8]` are the **aim tile (x, y)** — the tile the tank's barrel points at. Equals `(target_x, target_y)` for straight shots, may differ for guided weapons.

Files:
- `src/tankpit_bot/protocol/types.py` — rename `unk1`/`unk2` fields in `ShootEventDict` to `aim_x`/`aim_y`, update docstring.
- `src/tankpit_bot/protocol/decoders/combat.py` — rename in `decode_shoot_event`.
- All tests that touch `ShootEventDict` — update field names.
- `wiki/pages/decode-coverage.md` ShootEvent block — update field map.

Then corpus-verify: walk the 4267 ShootEvent samples in the corpus. Expect `aim_x == target_x AND aim_y == target_y` for ≥95% of straight-shot events. If <95%, flag back to the user before promoting.

## Verification

Run `make check 2>&1 | Select-Object -Last 100` after every change. Required: 3866+ tests passing, 100% coverage statements + branches. No mocks, no monkeypatching (the `_test_hooks` DI guard fails otherwise).

For Tier 1, after `make smoke` is built, run it. Exit code 0 means the fix from today is locked in end-to-end.

## What's already on disk

- `runs/bot/latest.events.jsonl` — last run's structured events
- `runs/bot/latest.capture_session.json` — wire bytes
- `runs/bot/latest.log` — human-readable log
- `runs/bot/latest.summary.txt` — scorecard
- `runs/bot/bot-*.capture_session.json` — 141 historical corpus sessions
- `tpclient.pretty.js` — deminified JS client source (7664 lines, single source of truth for wire format)
- `wiki/pages/decode-coverage.md` — current decode coverage map

## Communication

When you finish a tier, append your status to this file (new section: "Status update YYYY-MM-DD"). Don't update the other session's pending sections.
