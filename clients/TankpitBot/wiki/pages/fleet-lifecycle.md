---
title: Fleet Manager Lifecycle
tags: [fleet, architecture, lifecycle, operations]
related:
  - "[[fleet-coordination]]"
  - "[[bot-service-architecture]]"
  - "[[fleet-live-reads]]"
source_paths:
  - "src/tankpit_bot/service/fleet.py"
  - "src/tankpit_bot/service/fleet_control.py"
  - "src/tankpit_bot/service/fleet_record.py"
  - "src/tankpit_bot/service/fleet_adoption.py"
  - "src/tankpit_bot/service/fleet_manager.py"
  - "src/tankpit_bot/service/serving.py"
source_git_blobs:
  "src/tankpit_bot/service/fleet.py": "3bdbe67257c01d402b4a4f5dce81fb337e1363ff"
  "src/tankpit_bot/service/fleet_control.py": "847007ba72621abd7a0e23942c4d109a1a6fafac"
  "src/tankpit_bot/service/fleet_record.py": "8cc5acfca6333d8c51ec46921d7b7dc5a6bfc898"
  "src/tankpit_bot/service/fleet_adoption.py": "8de564766a3bb4a389c4df6996d9a637e0a86f11"
  "src/tankpit_bot/service/fleet_manager.py": "f9f31cc03765d47e659993317d8a0ad02e4cf0cf"
  "src/tankpit_bot/service/serving.py": "5bc4eeb8e04acca18551ab9bb153b812f6b50dbf"
fact_checked: "2026-09-01"
confidence: high
hubs: [architecture]
---

# Fleet manager lifecycle: no orphans, no killed tanks

*Established 2026-09-01 (operator ruling: "i dont want orphaned
processes"), against the older doctrine that the manager may simply
walk away from its children.*

The fleet spawns bots as **child processes** so that losing the
manager can never kill a live tank.[^1] That choice bought safety and
cost supervision: the registry lived only in memory, so every manager
restart produced bots that were still fighting and no longer
reachable — not stoppable, not inspectable, not visible on the page.
The only way to end one was to find its pid by hand.

Two mechanisms close that gap from opposite ends.

## Draining: the manager exits last

An interrupt, or `POST /shutdown`, asks every live bot to stop and the
manager **keeps serving while they tear down**, exiting only once the
last one is gone.[^2] That is the only moment at which exiting orphans
nothing.

Draining never kills. Each bot ends through the same stop sentinel a
bounded session ends on, so it writes its scorecard and **quits to the
lobby** rather than being cut down mid-game — a tank killed outright
loses its rank ([[bot-behavior-contract]]).[^3] The wait therefore has
**no deadline**: hurrying a teardown to meet a timeout is precisely
how a tank gets left exposed.

`make down` is a CLIENT of that drain, not the thing performing it.[^4]
It posts the shutdown and watches the port; interrupting it changes
nothing, because the manager owns the drain and still exits only when
its bots have landed. An indefinite wait can therefore never itself
create the orphans it is avoiding.

## Adoption: a restarted manager finds its bots

If the manager dies anyway — a crash, a closed window — the bots
survive by design, and the **next** manager adopts them. Every spawn
writes `runs/bot/<instance>/process.json`; every boot reads those
records back, re-attaches to the processes still alive, and deletes
the records of the ones that finished unwatched.[^5]

A record names an **identity, not a pid**. Windows recycles pids, so a
manager restarted minutes later could otherwise adopt an unrelated
program that inherited the number, then refuse to restart that
instance forever because its imaginary bot never exits. The process
creation time is recorded beside the pid and compared **exactly** —
the same `(pid, create_time)` identity psutil itself uses.[^6]

Records are written atomically, so a record that fails to decode is
real corruption rather than a torn write. Adoption **raises** on one
instead of skipping it: starting a manager on top of a damaged record
would mean silently forgetting a tank that may still be playing.[^5]

**Liveness is asked directly, never inferred from the exit code.** The
registry's process surface splits `is_running()` from `exit_code()`,
and `alive` is the authoritative one. Deriving one from the other is
sound only while they always agree, and for an adopted process they do
not: the OS can be certain a process ended while being unable to say
what code it ended with. The first live drain proved the cost --
a bot that had already landed read as running, and the manager waited
on it forever ([[fleet-live-reads]] records the same lesson about
measurement).[^9]

## Boot identity: the page re-syncs itself

`GET /bots` carries the manager's `boot` id.[^7] A control page that
sees it change knows every instance name it holds belongs to a manager
that no longer exists, and reloads. Before this, a tab left open
across a restart kept polling names the new process had never heard
of, aiming a steady stream of 404s at it — the symptom that opened
this work.

## Operator surface

| Command | Effect |
|---|---|
| `make up` | THE fleet command (operator consolidation 2026-09-02, [[fleet-forage-allocation]] era): resolve the newest release, build its image if the tag (`tankpit-fleet:v<ver>-<sha>`) does not exist yet, and compose the fleet CONTAINER up with the release's own `runs/` and `accounts.json` mounted. Prints the fleet page URL. |
| `make down` | Drain: `docker stop`'s SIGTERM enters `drain_on_interrupt`, every bot quits to the lobby, the container exits after the last one (grace 10 m). |
| `make dev` | The only other manager: hot tree, foreground, development only — Ctrl+C drains. |

Retired the same day, one system not two: `make fleet` (foreground
release manager), `make fleet-dev`, and the short-lived
`image`/`up-docker`/`down-docker` trio. The detached HOST lifecycle
(`tankpit-fleet-up`/`-down` entry points, the adoption of host
processes below) remains in code but is no longer on the operator
surface; a container manager cannot adopt HOST processes, so any
pre-container host fleet must be drained (`poetry run
tankpit-fleet-down` from its release folder) before the first
containerized `make up`.

## Port 27300 is arbitrated, because Windows loopback is not

Observed live 2026-09-02, the first transition attempt: a leftover
HOST manager held `127.0.0.1:27300`, and the containerized `make up`
still reported `Started` — Windows loopback has no exclusive bind by
default, so Docker's host-side port proxy co-bound without error and
then never worked. Every request went to the host manager: the
operator watched a "fleet online" page through a full container
up/down cycle (the container stopped in 1.4 s because it held zero
bots — the drain was correct, the page was somebody else's). The
reverse direction also fails invisibly: while a fleet container
EXISTS (even stopped), Docker reserves `127.0.0.1:27300` with a
socket `netstat` does not list, and a host bind gets `WinError
10048`; the reservation releases seconds after `docker compose down`
removes the container.

Both `make` targets therefore check WHO owns the port. `up` refuses
to start behind a non-Docker listener (names the process and pid,
prints the drain command), then after composing verifies the page
actually answers `/bots` and force-recreates once if the proxy is
stale — the URL is printed only after a real response. `down` names
any non-Docker survivor on the port after the container stops, so
"still online" is never a mystery. The in-container manager needs no
guard: it binds `0.0.0.0` in its own namespace, and the observed
failure was never inside the container.

A detached manager has no terminal, so its console goes to
`runs/fleet/manager.log` — the same reasoning that sends each bot's
child console to a file.[^8]

[^1]: `src/tankpit_bot/service/fleet.py:3-6` states the doctrine; the
      children are `subprocess.Popen` handles created in
      `_real_spawn_bot_process`
      (`service/_test_hooks/processes.py`).
[^2]: `exit_when_drained` (`service/fleet.py`) polls
      `FleetManager.live_instances` and sets the serve loop's stop
      event only when it is empty.
[^3]: `FleetManager.request_drain` (`service/fleet_manager.py`) writes
      the same `runs/bot/<instance>/STOP` sentinel as `stop`, via the
      shared `_request_stop`.
[^4]: `down` in `src/tankpit_bot/service/fleet_control.py`; the
      module docstring records why the waiting belongs to the manager.
[^5]: `adopt_recorded_bots` (`service/fleet_adoption.py`) over the
      records defined in `service/fleet_record.py`.
[^6]: `_real_open_adopted_process`
      (`service/_test_hooks/processes.py`) compares
      `psutil.Process.create_time()` against the recorded value.
      Verified against real processes in
      `tests/service/test_fleet_process_hooks.py`, including the
      recycled-pid refusal.
[^7]: `FleetManager.boot_id`, served by `encode_fleet_snapshot`
      (`service/fleet_wire.py`) and compared in the page's poll loop
      (`service/fleet_page.py`).
[^8]: `FLEET_LOG_PATH` (`src/tankpit_bot/runtime_artifacts.py`).
[^9]: `SpawnedProcessProtocol`, `_AdoptedProcess` and `_PopenProcess`
      in `service/_test_hooks/processes.py`; the property is pinned by
      `test_liveness_never_depends_on_recovering_an_exit_code`
      (`tests/service/test_fleet_process_hooks.py`). Diagnosed live
      2026-09-01, see `wiki/log.md`.
