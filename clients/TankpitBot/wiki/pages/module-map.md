---
title: Module Map
tags: [codebase, architecture, navigation]
related:
  - "[[services]]"
  - "[[inheritance-chain]]"
source_paths:
  - "src/tankpit_bot"
source_git_blobs:
  "src/tankpit_bot": "dcfc8ef145312a293178da152b4ca08321824b2d"
fact_checked: "2026-09-05"
confidence: high
hubs: [codebase]
---

# Module Map

All source lives under `src/tankpit_bot/`. Tests mirror the structure under `tests/`. Standalone scripts live in `scripts/` — layout conventions in [[coding-standards]].

## Core packages

| Package | Purpose | Key files |
|---------|---------|-----------|
| `bot/` | The game-playing bot — HFSM states, command dispatch, tick loop | `base.py` (Bot class), `ai/` (all decision logic), `tick_loop.py` (orchestrator), `executor.py` (dispatch + ledger recording), `config.py` (env-resolved launch settings) |
| `bot/ai/` | The two durable mode owners and every planner they delegate to | `mode_controller.py` (entry/exit rules); the HUNT family — `hunt_mode.py` (owner + phases), `hunt_acquire.py` (search/greet/acquire), `hunt_lock.py` (pursuit fire + break escape), `hunt_relay.py` (dot-relay travel); the COLLECT family — `collect_mode.py` (owner + sense/safety gates), `collect_pickups.py`, `collect_locks.py`, `collect_hops.py` (larder + marooned ladder), `collect_common.py` (score/blacklist); `ferry.py` (`compose_decision_terrain` — the single walkability owner, see [[terrain-composition]]), `intent.py` (see [[committed-intent]]). Split 2026-07-31 under the 400-600-line file rule ([[coding-standards]]) |
| `browser/` | Browser automation — Playwright, CDP, login, room join | `session_base.py` (shared composition), `lifecycle.py` (standalone functions), `login.py` + `room_join.py`, `game_log.py` (one owner for scraper setup + poll, was duplicated on Bot and BrowserSession), `page_client_snapshot.py` + `client_structure.py` (CDP page readers, moved out of `action_lab/` so production code stops importing the probe package) |
| `state/` | World state types and mutations — tanks, containers, viewport | `types/` (TypedDicts), `self_mutations.py` / `tank_mutations.py` / `terrain_mutations.py` / `container_mutations.py` (one module per subject), `projections/` (the `Fact[T]` read models), `line_of_sight.py` (a QUERY over terrain, not a physics rule), `viewport_geometry.py` |
| `protocol/` | Wire protocol — framing, encoding, decoding, command constants | `commands.py` (CMD_* constants), `codec.py` (XOR encode/decode), `decoders/` + `encoders/` (mirrored; byte-identity proven by `make roundtrip`), `types/` (one module per payload family, membership mirroring `decoders/`) |
| `wire/` | The byte layer both codecs sit on — a true leaf | `helpers.py`: x16/x24 and their pack inverses, the length validators, `DecodeError`/`EncodeError`. Lived in `protocol/helpers.py`, which forced `container` to import `protocol` while `protocol` imported `container` back |
| `container/` | The container message family, decoded from inside `protocol/decoders/tank.py` | `types.py`, `decoders/`, `encoders.py`. Imports `wire` and nothing else |
| `types/` | The shared vocabulary every layer names — a true leaf, imports nothing from this codebase | `constants.py` (terrain/team/damage codes, ASCII glyphs — was `state/types/constants.py`, where it made `physics` and `state` mutually dependent), `modes.py` (`BehaviorMode` — was `bot/ai/modes.py`, the one file forcing `service` to import `bot`), `literals.py`, `message.py`, `session.py`, `cdp.py`, `config.py`, `probe.py` ([[package-layering]]) |
| `analysis/` | The typed capture-scan owner — one load-XOR-split-decode pipeline instead of forty | `scan.py` (`scan_session`, direction-tagged frames), `types.py`, `_test_hooks.py`. Thirty of the forty `analysis_scripts/` each wrote this pipeline for themselves; every step now delegates to the module that already owned it ([[capture-differ]]) |
| `stream/` | Display-capture video — Xvfb + ffmpeg around a streamed session | `capture.py` (process lifecycle + argv builders), `hls.py` (HTTP answers for the produced files), `types.py` (`StreamConfigDict` + codecs). Owned by `Bot.run` for exactly one session; serving reads plain files, so video touches neither the tick loop nor the page (2026-09-05, replacing the canvas-scrape caster + frame bus) |
| `sniffer/` | Passive WebSocket sniffer — captures traffic without playing | `core.py` (entry point), `world_service.py` + `world_service_beliefs.py` / `world_service_movement.py` / `world_service_radar.py` (the service each session now owns — three sibling modules, not a `world_service_beliefs/` package as this row read until 2026-09-03), `world_state_*.py` (the state machine — `world_state_dispatch*.py` plus one module per subject: combat, containers, inventory, radar, tanks, tiles). The singleton module `world_state.py` that this row used to name **no longer exists**; see [[services]] [^4] and [[session-state-deglobalisation]] step 8 |
| `capture/` | Post-hoc capture analysis — shot correlation, viewport analysis | `stats.py`, `viewport_analysis.py`, `trackers/` |
| `action_lab/` | Live probes — isolated experiments against the real server | `probe_base.py` (ProbeBase), `probe_factory.py` (DI), teleport/fuel/equipment/movement probes |
| `diagnostics/` | Runtime + offline diagnostics — issue reports, alignment checks | `issue_report.py`, `entity_alignment.py`, `self_alignment.py`, `session_stats.py` |
| `replay/` | Replay engine — re-runs captures through bot decision logic | `engine.py`, used by `tests/replay/` regression tests |
| `physics/` | The game's measured laws, one symbol per machine-checked wiki claim | `costs.py`, `capacity.py`, `damage.py`, `combat.py`, `map.py` (see [[physics-module-roadmap]] Phase 1) |
| `sim/` | The server twin — laws, world, transport, practice room | `server.py` (routing/orchestration), `viewport_window.py` (stored 0x5A window + patch memory + visibility), `narrate/` (the emission side, split out of the former `combat_emissions.py` + `emissions.py`: `combat.py`, `movement.py`, `resources.py`, `world.py` — pure narration taking a resolved outcome plus an `observer_id`, so the law modules own every mutation and calling narration once per observer no longer applies each effect N times), `wire_statements.py` (pure builders), `world.py` + `world_seed.py` (static population + mined layouts), `bot_policy.py` + `practice_room.py` (certified bot minds), `opponent.py` (scripted harness) |
| `validate/` | Archive-priced law validators — `make audit` / `make shadow` / roundtrip | `audit.py`, `shadow*.py`, `roundtrip.py`, `wire_timeline.py` |
| `ledger/` | Live physics bookkeeping — fuel, ammo, and per-enemy damage books, divergence verdicts | `fuel_book.py` (windows + per-kind session totals), `ammo_book.py`, `damage_book.py` (dealt/taken per enemy by weapon), `outcome/` (per-command outcome resolvers) |
| `facts/` | The belief VOCABULARY — provenance and confidence, the Facts layer of [[self-observing-architecture]] | `fact.py`, `provenance.py`, `confidence.py`, `source.py`. A leaf: the `Fact[T]` projections that read world state back live in `state/projections/`, because keeping them here made `facts` and `state` mutually dependent |
| `service/` | The phone-driven bot service — aiohttp + SSE around the tick loop | `http_server.py`, `session_runner.py`, `watch_page.py`, `config.py`; the fleet is `fleet_manager.py` (registry) + `fleet_routes.py` (HTTP) + `fleet.py` (entry point only) |
| `bus/` | The cross-thread session buses and the status contract they carry | `mode_bridge.py`, `status_bus.py`, `session_status.py`. Below both `bot` and `service`: a standalone session gets inert buses with zero subscribers, the service injects shared ones ([[package-layering]]). `frame_bus.py` left with the canvas-scrape video pipeline (2026-09-05); video is now `stream/` — Xvfb + ffmpeg display capture into HLS files, no bus involved |
| `contracts/` | Enforcement decorators backing the `scripts/contract_rules.py` guard rule | `base.py`, `enforcement.py` |

## Support modules (top-level, not packages)

| Module | Purpose |
|--------|---------|
| `_test_hooks/` | Protocol interfaces for DI — 8 submodules by domain (bot, browser, cdp, env, fs, etc.) |
| ~~`_hooks_guard.py`~~ | **Gone** (deleted in `8c6453da`). It never enforced anything, and `MonkeyPatchBanRule` lives in the shared guards library at `libs/monorepo_guards/src/monorepo_guards/monkey_patch_rules.py` — see [[testing-patterns]], whose footnote was corrected for this same misattribution on 2026-08-07 while this row was missed |
| `_pillow.py` | Typed Pillow boundary — `PillowImageProtocol` / `PillowImageModuleProtocol` + `load_pillow_image_module()`, so the dynamic PIL import stays strict-typed without a mypy import exception (consumed by `terrain.py`) |
| `protocol/` | Wire constants, framing, encode/decode |
| `parser.py` + `parser_messages.py` | CDP message parsing |
| `decoder.py` + `state_decoder.py` | Wire blob decoders |
| `terrain.py` | Terrain map loader (from GIF files) |
| `combat.py` + `combat_tracker.py` | Combat event tracking |
| `inventory.py` | Inventory state management |
| `runtime_logging.py` | Structured logging setup |
| `runtime_artifacts.py` | Run directory management (runs/bot/, runs/sniffer/) |

## Dependency flow

```
bot/ai/ ──→ bot/ ──→ browser/ ──→ protocol/
  │           │         │
  └───────────┴─────────┴──→ state/
                              │
action_lab/ ─→ browser/ ──→ protocol/
  │
  └──→ state/

sniffer/ ──→ browser/ ──→ protocol/
  │
  └──→ state/

service/ ──→ bot/ (owns the tick loop on a worker thread)
bot/     ──→ bus/ ──→ types/   (bot imports service ZERO times)
service/ ──→ bus/

capture/     ──→ state/              (no browser)
replay/      ──→ bot/ai/ + state/    (no browser)
sim/         ──→ bot/ai/ + protocol/ (no browser, no server)
validate/    ──→ sim/ + physics/     (offline, reads the runs archive)
diagnostics/ ──→ ledger/ + state/    (offline, reads events.jsonl)
```

`physics/`, `facts/`, `ledger/`, and `contracts/` are leaf layers — anything above may import them; they import nothing from the bot[^3].

All three consumers (Bot, ProbeBase, BrowserSession) inherit from `SessionBase` in `browser/session_base.py`. See [[services]] for how the DI wires together.[^1]

## Scripts (`scripts/`)

Standalone CLI tools, each with a `main()` entry point registered in `pyproject.toml`. Probes (`teleport_probe.py`, `fuel_probe.py`, etc.) wrap `action_lab/` probe classes. Analysis scripts (`analyze_session_timing.py`, `analyze_shot_viewport.py`) process capture files offline.[^2]

[^1]: Architecture phases A-F landed on the `combat-rework` branch, tip `c8a1d40eec64c3a0ce198d21f57af35df7449606`, fully merged (`git log main..origin/combat-rework` is empty). The arc opens at `e1d8f060` — "Extract WorldService class to own all mutable game state", 2026-06-14 — which is phase A. Recounted 2026-08-05: **56** commits dated 2026-06-14 through 2026-06-16, of which **41** touch `clients/TankpitBot`. **Corrected 2026-08-05:** this footnote said "52 commits", which matches neither count on either scoping. Both figures are reproducible from any checkout against the pinned tip sha. [[inheritance-chain]] carries the same figure and is corrected identically.
[^2]: `pyproject.toml` `[tool.poetry.scripts]` — **34** `main()` entry points as of 2026-08-05, e.g. `tankpit-teleport-probe = "scripts.teleport_probe:main"` at `:33`. This count is volatile: it was 33 earlier the same day and gained `tankpit-run-digest` in commit `29131c95`. Re-derive rather than trust the number — `awk '/^\[tool.poetry.scripts\]/{f=1;next} /^\[/{f=0} f && /=/{n++} END{print n}' pyproject.toml`.
[^3]: Re-verified 2026-08-05 (previously 2026-07-31, before the 84-file source change this page's pin now spans): a grep for `from tankpit_bot.bot` / `import tankpit_bot.bot` under `src/tankpit_bot/physics/`, `facts/`, `ledger/`, and `contracts/` returns **zero** matches in all four packages, so the leaf-layer direction still holds by construction rather than by convention. Representative modules in each, all present at re-verification: `physics/costs.py`, `facts/fact.py`, `ledger/fuel_book.py`, `contracts/base.py`.
