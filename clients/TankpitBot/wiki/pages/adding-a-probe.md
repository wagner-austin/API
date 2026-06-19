---
title: Adding a New Probe
tags: [codebase, action-lab, howto]
related: [[services]], [[module-map]], [[testing-patterns]], [[make-targets]]
sources: [codebase inspection 2026-06-16]
fact_checked: 2026-06-16
confidence: high
---

# Adding a New Probe

Action lab probes are isolated experiments against the live game server. Each probe tests one mechanic (teleport, fuel pickup, movement, etc.) without the full bot HFSM.

## Step 1: Create the probe class

In `src/tankpit_bot/action_lab/`, create `my_probe.py`:

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

In `scripts/my_probe.py`:

```python
from tankpit_bot.action_lab.probe_factory import create_probe
from tankpit_bot.action_lab.my_probe import MyProbe

def main() -> None:
    probe = create_probe(MyProbe, "https://tankpit.com")
    probe.run()
```

`create_probe()` handles all DI — it creates CDPService + CommandService and injects them.[^3]

## Step 4: Register the CLI entry point

In `pyproject.toml` under `[tool.poetry.scripts]`:

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

## Existing probes (6 types)

| Probe | File | What it tests |
|-------|------|--------------|
| Teleport | `teleport.py` | Teleport landing accuracy, timing strategies |
| Enemy Teleport | `enemy_teleport.py` | Teleporting near enemies |
| Fuel | `fuel_probe.py` | Fuel container pickup |
| Fuel Dot | `fuel_dot_probe.py` | MAP_DATA fuel dot verification |
| Equipment | `equipment_probe.py` | Equipment container pickup |
| Movement | `movement_probe.py` | Walking to targets |
| Queue | `queue_probe.py` | Multi-command server batching |

[^1]: action_lab/probe_base.py — ProbeBase(SessionBase), shared composition with Bot
[^2]: e.g. fuel_probe_types.py, movement_probe_types.py — TypedDict with encode/decode pattern
[^3]: action_lab/probe_factory.py — create_probe() creates services and injects
[^4]: _test_hooks/ protocols match real API signatures; see testing-patterns.md
