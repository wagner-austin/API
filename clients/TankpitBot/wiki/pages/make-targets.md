---
title: Make Targets
tags: [codebase, cli, tooling]
related: [[module-map]], [[adding-a-probe]]
sources: [Makefile inspection 2026-06-16]
fact_checked: 2026-06-16
confidence: high
---

# Make Targets

`SHELL := powershell.exe` — Make runs PowerShell internally. From the terminal, just run `make <target>`.[^1]

## Code health (safe, offline)

| Target | What it does |
|--------|-------------|
| `make check` | `lint` + `test` — **the gate**; run before every commit |
| `make lint` | guard + ruff + mypy |
| `make test` | pytest + coverage (100% required) |
| `make install` | poetry install + playwright install chromium |

## Live bot (needs browser + accounts.json, touches live server)

| Target | What it does |
|--------|-------------|
| `make bot` | Run the HFSM bot indefinitely (no timeout) |
| `make run` | 5-minute timed session + scorecard (`TANKPIT_BOT_SESSION_SECONDS=300`) |
| `make play` | Human session capture via sniffer (you play, it records) |
| `make sniff` | Passive WebSocket capture to disk |

## Live probes (need browser + accounts.json, touches live server)

| Target | What it does |
|--------|-------------|
| `make movement-probe` | Walk to 3 targets — cheapest smoke test |
| `make teleport-probe` | Both safe + aggressive teleport strategies |
| `make teleport-probe-safe` | 3 teleports with sync_before_teleport |
| `make teleport-probe-aggressive` | 3 teleports with immediate_after_map_open |
| `make fuel-probe` | 3 fuel pickups via 9 attempts |
| `make fuel-drill` | Fill tank to 1100 (long-running) |
| `make equipment-probe` | 3 equipment pickups via 9 attempts |
| `make queue-probe` | Test multi-command batching against server |

## Offline analysis (safe, reads capture files)

| Target | What it does |
|--------|-------------|
| `make analyze` | Issue report + cross-session stats on latest run |
| `make analyze-timing` | Command-response timing analysis |
| `make decode` | Replay a capture through real decoders |
| `make discover` | Extract command constants from JS client |
| `make analyze-viewport` | Analyze viewport bounds in captures |

## Output

Bot runs save to `runs/bot/`, sniffer to `runs/sniffer/`. `latest.events.jsonl` and `latest.capture_session.json` are symlinks to the most recent run.[^2]

[^1]: Makefile line 1 — `SHELL := powershell.exe`; Make handles PowerShell internally
[^2]: runtime_artifacts.py — creates run directories and latest symlinks
