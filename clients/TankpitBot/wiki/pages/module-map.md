---
title: Module Map
tags: [codebase, architecture, navigation]
related:
  - "[[services]]"
  - "[[inheritance-chain]]"
source_paths:
  - "src/tankpit_bot"
source_git_blobs:
  "src/tankpit_bot": "3ca61a0e99cf8eaeb095396deabdc39637de3b6b"
fact_checked: "2026-06-16"
confidence: high
hubs: [codebase]
---

# Module Map

All source lives under `src/tankpit_bot/`. Tests mirror the structure under `tests/`. Standalone scripts live in `scripts/` — layout conventions in [[coding-standards]].

## Core packages

| Package | Purpose | Key files |
|---------|---------|-----------|
| `bot/` | The game-playing bot — HFSM states, command dispatch, tick loop | `base.py` (Bot class), `ai/` (all decision logic), `tick_loop.py` (orchestrator) |
| `browser/` | Browser automation — Playwright, CDP, login, room join | `session_base.py` (shared composition), `lifecycle.py` (standalone functions), `login.py` + `room_join.py` |
| `state/` | World state types and mutations — tanks, containers, viewport | `types/` (TypedDicts), `mutations.py`, `viewport_geometry.py` |
| `protocol/` | Wire protocol — framing, encoding, decoding, command constants | `commands.py` (CMD_* constants), `codec.py` (XOR encode/decode), `decoders/` |
| `sniffer/` | Passive WebSocket sniffer — captures traffic without playing | `core.py` (entry point), `world_state.py` + `world_state_*.py` (state machine) |
| `capture/` | Post-hoc capture analysis — shot correlation, viewport analysis | `stats.py`, `viewport_analysis.py`, `trackers/` |
| `action_lab/` | Live probes — isolated experiments against the real server | `probe_base.py` (ProbeBase), `probe_factory.py` (DI), teleport/fuel/equipment/movement probes |
| `diagnostics/` | Runtime + offline diagnostics — issue reports, alignment checks | `issue_report.py`, `entity_alignment.py`, `self_alignment.py`, `session_stats.py` |
| `replay/` | Replay engine — re-runs captures through bot decision logic | `engine.py`, used by `tests/replay/` regression tests |

## Support modules (top-level, not packages)

| Module | Purpose |
|--------|---------|
| `_test_hooks/` | Protocol interfaces for DI — 8 submodules by domain (bot, browser, cdp, env, fs, etc.) |
| `_hooks_guard.py` | MonkeyPatchBanRule enforcement |
| `protocol/` | Wire constants, framing, encode/decode |
| `parser.py` + `parser_messages.py` | CDP message parsing |
| `decoder.py` + `state_decoder.py` | Wire blob decoders |
| `terrain.py` | Terrain map loader (from GIF files) |
| `game_state.py` | Top-level game state container |
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

capture/ ──→ state/ (no browser dependency)
replay/  ──→ bot/ai/ + state/ (no browser dependency)
```

All three consumers (Bot, ProbeBase, BrowserSession) inherit from `SessionBase` in `browser/session_base.py`. See [[services]] for how the DI wires together.[^1]

## Scripts (`scripts/`)

Standalone CLI tools, each with a `main()` entry point registered in `pyproject.toml`. Probes (`teleport_probe.py`, `fuel_probe.py`, etc.) wrap `action_lab/` probe classes. Analysis scripts (`analyze_session_timing.py`, `analyze_shot_viewport.py`) process capture files offline.[^2]

[^1]: architecture phases A-F, 52 commits on combat-rework; see [[inheritance-chain]]
[^2]: pyproject.toml [tool.poetry.scripts] section — all CLI entry points
