---
title: Make Targets
tags: [codebase, cli, tooling]
related:
  - "[[module-map]]"
  - "[[adding-a-probe]]"
source_paths:
  - "Makefile"
source_git_blobs:
  "Makefile": "700340821708d152af90d026c6504b1608d6b87e"
fact_checked: "2026-07-25"
confidence: high
hubs: [codebase]
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
| `make run` | Timed session + scorecard. Honors a pre-set `TANKPIT_BOT_SESSION_SECONDS` (default 300) and `TANKPIT_BOT_SESSION_KILLS` (wind-down at the Nth kill). Sessions > 120 s end themselves cleanly (`session_complete`): finish the live fight, top off, quit — [[bot-behavior-contract]] §1.2 wind-down |
| `make sim-run` | Production bot vs the simulator on real field01 terrain — no server, no browser, no fuel spent. Artifacts: `runs/probe/latest.sim.*` + `runs/sim/sim-<stamp>.capture_session.json` (standard CaptureSession — `tankpit-audit --runs-dir` can price it). `tankpit-sim-run --rounds N --no-opponent` for variants. See [[physics-module-roadmap]]. |
| `make sim-run-practice` | Production bot vs a REAL practice room (2026-07-25 rework): a stamp-selected mined layout seeds the full 36-bot roster (ids 500-535, 9/team) at archive-observed positions plus the client's real join spawn, on a static container field (~620-dot exposure atlas at the live ~40% hold rate + measured hidden population (840 fuel, half drained + 180 equipment); no runtime spawning — the respawn law was falsified). Bots driven by the certified `sim/bot_policy`. The fidelity soak: 150/150 rounds sustainably, kills across the map, exposure law 18/18 on the sim's own capture. `tankpit-sim-run --practice`. |
| `make sniff` | WebSocket capture to disk — also the human-session recorder (you play, it records). `OUTPUT=<path>` overrides the capture file location. The former `make play` alias was removed 2026-07-01 (identical command). |

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
| `make respawn-watch` | Teleport adjacent to a bot, fire a single at its registry position every 2 s for up to 30 s (kills already-damaged bots; full-fuel ones teleport off at 7-8 hits either way), then map-poll every 2 s for 60 s so the 0x4C snapshots pin the same-id reactivation tick and tile. Up to 4 targets per session. Analysis is offline from the capture (0x41 kill vs 0x58 flee). Knobs: `TANKPIT_RESPAWN_WATCH_*`. |
| `make key-probe` | Press each safe physical key once (own capture window per press) and attribute sent frames to keys. Settled R=radar 2026-07-24. |
| `make radar-watch` | Stationary spawn-law watch on the account: slot-5 extras toggled off (verified via wire state; stock preserved), free built-in 5×5 scan per 15 s + free map open per 30 s + 1-tile walk shuffle per beat (immune to the ~12-min never-playing disconnect). |
| `make density-probe` | Budgeted extra-radar density sweep (2026-07-25): teleport a 4×4 map-spread site grid, map-open before every hop, verify each landing before spending an extra, one full-viewport scan per landed site. Funds itself (viewport pickups → dot hops → blind dot-walks), aborts + quits to lobby when marooned, restores the slot-5 enable state, archives per run under `runs/probe/density-<stamp>`. Knobs: `TANKPIT_DENSITY_*`. Run-5 measurement in [[game-economy]]. |
| `make bot-watch` | Teleport adjacent to a practice bot, then dwell 10 min at a 1.5 s walk-shuffle heartbeat (2026-07-24: query heartbeats were falsified — only real gameplay actions hold the push stream open, ~40 fuel/min; each beat drains the CDP buffer then walks 1 tile). See [[server-push-gating]] for the law and the seven-run proof. |
| `make viewport-probe` | Autoscroll/viewport law probe (2026-07-25): normalize autoscroll to OFF via wire-verified 'a' presses (plaintext `A0`/`A1` acks), then per phase (OFF, ON) anchor-teleport with landing verification, walk terrain-routed steps to the window's east edge, attempt one crossing step, and fire long boundary moves. Quits to lobby on success AND abort. Measured the edge-recentering + acceptance-boundary laws in [[viewport-shift-protocol]]. Knobs: `TANKPIT_VIEWPORT_*`. |

## Offline analysis (safe, reads capture files)

| Target | What it does |
|--------|-------------|
| `make analyze` | Issue report + forage economy + run audit (deterministic verdicts + capture replay diff) + cross-session stats on latest run. The forage-economy section (`tankpit-forage-economy`, 2026-07-26) answers "where did the time go": hunt/collect split, forage viewports per kill, pickups per viewport, weapons per equipment pickup, hop selected/declined breakdown. Pass two events paths to diff runs — built for the 803 s vs 1,187 s 10-kill pair whose deciding number (weapons/pickup 3.34 vs 2.14) was invisible to the issue report. |
| `make audit` | Re-derive every validated wiki physics claim from the full runs archive (`tankpit-audit --stamp` rewrites `fact_checked:` on green pages). See [[physics-module-roadmap]] Phase 2. |
| `make shadow` | Price the SIM's laws against the archive — every validator imports its predictor from sim source (sync cadence, grant invariants, kill mercy bundle, corpse window). A failure = sim and real server disagree. See [[physics-module-roadmap]]. |
| `make roundtrip` | Decode→encode→decode every archive message; byte-identity proof for the sim's encoders |
| `make analyze-timing` | Command-response timing analysis |
| `make decode` | Replay a capture through real decoders |
| `make discover` | Extract command constants from JS client |
| `make analyze-viewport` | Analyze viewport bounds in captures |

## Output

Bot runs save to `runs/bot/`, sniffer to `runs/sniff/`. `latest.events.jsonl` and `latest.capture_session.json` are symlinks to the most recent run.[^2]

[^1]: Makefile line 1 — `SHELL := powershell.exe`; Make handles PowerShell internally
[^2]: runtime_artifacts.py — creates run directories and latest symlinks
