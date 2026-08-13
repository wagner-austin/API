---
title: Adding a New Probe
tags: [codebase, action-lab, howto]
related:
  - "[[services]]"
  - "[[module-map]]"
  - "[[testing-patterns]]"
  - "[[make-targets]]"
source_paths:
  - "src/tankpit_bot/action_lab"
source_git_blobs:
  "src/tankpit_bot/action_lab": "dfa2492b0c379b9831dbbdd2d0a7c5d0ad856776"
fact_checked: "2026-08-07"
confidence: high
hubs: [codebase]
---

# Adding a New Probe

Action lab probes are isolated experiments against the live game server. Each probe tests one mechanic (teleport, fuel pickup, movement, etc.) without the full bot HFSM.[^1]

## Step 1: Create the probe class

In `src/tankpit_bot/action_lab/`, create `my_probe.py` (every existing probe follows this shape[^1]):

```python
class MyProbe(ProbeBase):
    """One-line description."""
    
    def execute(self) -> None:
        # Your experiment logic here
        # self has CDPService + CommandService via SessionBase
        pass
```

`ProbeBase(SessionBase)` gives you everything the bot has — CDP events, command dispatch, world state — without the HFSM decision loop.[^1]

## Step 2: Create types (if needed)

If the probe produces structured results, create `my_probe_types.py` with TypedDicts. Follow the `encode`/`decode`/`require_*` validation pattern used by other probe types.[^2]

## Step 3: Create the script entry point

In `scripts/my_probe.py` (mirrors the six existing probe scripts[^3]):

```python
from tankpit_bot.action_lab.probe_factory import create_probe
from tankpit_bot.action_lab.my_probe import MyProbe

def main() -> None:
    probe = create_probe(MyProbe, "https://tankpit.com")
    probe.run()
```

`create_probe()` handles all DI — it creates CDPService + CommandService and injects them.[^3]

## Step 4: Register the CLI entry point

In `pyproject.toml` under `[tool.poetry.scripts]` — the existing `tankpit-*-probe` entries are the template ([[make-targets]]):

```toml
tankpit-my-probe = "scripts.my_probe:main"
```

## Step 5: Add the Makefile target

```makefile
my-probe: install
    @Write-Host "==> my-probe" -ForegroundColor Cyan
    poetry run tankpit-my-probe
```

## Step 6: Write tests

Test the probe's logic using `_test_hooks` protocol implementations. No mocking. See [[testing-patterns]] for the pattern.[^4]

## Step 7: Verify

```bash
make check          # lint + test + coverage
make my-probe       # live run
```

## Existing probes (11 types)

| Probe | File | What it tests |
|-------|------|--------------|
| Teleport | `teleport.py` | Teleport landing accuracy, timing strategies |
| Enemy Teleport | `enemy_teleport.py` | Teleporting near enemies |
| Respawn Watch | `respawn_watch.py` | Engaging adjacent bots + map-polling their same-id reactivation (subclasses EnemyTeleportProbe via the `_post_landing_phase` hook) |
| Key | `key_probe.py` | Pressing physical keys and attributing sent frames per press (settled the R-key keymap empirically) |
| Radar Watch | `radar_watch.py` | Stationary spawn-law watch: extras toggled off, free built-in scans + map polls + a walk shuffle per beat |
| Fuel | `fuel_probe.py` | Fuel container pickup |
| Equipment | `equipment_probe.py` | Equipment container pickup |
| Larder | `larder_probe.py` | Own-tile equipment pickup vs adjacent control (subclasses DensityProbe for funded hops + extras etiquette; settled the larder-plan gate) |
| Mine landing | `mine_landing_probe.py` | Teleport aimed AT enemy mines: displacement vs detonation vs coexist (settled the ring-2 doctrine gate: displaces, 3/3) |
| Movement | `movement_probe.py` | Walking to targets |
| Queue | `queue_probe.py` | Multi-command server batching |

[^1]: action_lab/probe_base.py — ProbeBase(SessionBase), shared composition with Bot
[^2]: e.g. fuel_probe_types.py, movement_probe_types.py — TypedDict with encode/decode pattern
[^3]: action_lab/probe_factory.py — create_probe() creates services and injects
[^4]: _test_hooks/ protocols match real API signatures; see testing-patterns.md
