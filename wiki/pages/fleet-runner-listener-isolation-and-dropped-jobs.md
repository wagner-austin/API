---
title: WSL runner listeners sit outside their jobs' budget, and the hub ends a job GitHub dropped
tags: [fleet, github-actions, self-hosted-runners, systemd, cgroups, memory-pressure, infrastructure]
related:
  - "[[monorepo-discipline]]"
source_paths:
  - tools/fleet/src/fleet/contracts/runner_slice.py
  - tools/fleet/src/fleet/core/runner_job_scope_render.py
  - tools/fleet/src/fleet/core/runner_dropped_check.py
  - tools/fleet/src/fleet/core/runner_reap_dropped.py
  - tools/fleet/src/fleet/core/runner_reaper_render.py
  - tools/fleet/tests/test_runner_placement_host.py
source_git_blobs:
  "tools/fleet/src/fleet/contracts/runner_slice.py": b6b2115238d91a8485cb24fc2ed5edac0bbf9fea
  "tools/fleet/src/fleet/core/runner_job_scope_render.py": aadaf3447a0ae0923c980fd772bcbc86806013f8
  "tools/fleet/src/fleet/core/runner_dropped_check.py": e946b9ae156b05042934830476b9d9f0f24335d1
  "tools/fleet/src/fleet/core/runner_reap_dropped.py": 32f8ee0d947fbc9283c91230e531d3aa2f4e6ada
  "tools/fleet/src/fleet/core/runner_reaper_render.py": 931c3e1d4db9f1a804bd82f9e5c898b588dae0e5
  "tools/fleet/tests/test_runner_placement_host.py": f79e194e1a0ac7b2e2e5948eabfd8f0e46015f52
fact_checked: "2026-10-17"
confidence: high
hubs: [infrastructure]
---

# WSL runner listeners sit outside their jobs' budget, and the hub ends a job GitHub dropped

Lavender's WSL instance runs eight self-hosted GitHub Actions runners whose jobs share one bounded systemd slice, `runners.slice`.[^ci-slice] Until MCPs board task 715b9ec5 the runner units sat in that slice too, `Runner.Listener` included, and the listener is the process that holds each job's lease with GitHub. On 2026-10-02 and 2026-10-16 the slice sat at its `MemoryHigh`, the listeners' calls to GitHub timed out, GitHub dropped the jobs, and their Workers ran on unseen for 1 h 45 min.[^why]

## Why no memory protection could have saved the listener

The kernel throttles every task that charges memory beneath a cgroup over its `memory.high`, whatever that task's own `memory.low`. Measured on lavender-wsl: a process holding 4.5 MB under a 200 MB `MemoryLow` sat 227 s in `mem_cgroup_handle_over_high` beside a hog over its slice's high, while the same process in a slice of its own ran unhindered.[^low] The only fix is a different cgroup.

## The split

- **`listeners.slice`**, unbounded, holds every runner unit; no dash in either name, so neither is a child of the other.[^listener-slice]
- **Each job is moved out of its unit** before its first step: a drop-in sets `ACTIONS_RUNNER_HOOK_JOB_STARTED`, the hook runs a root helper through a one-command sudoers rule, and the helper has systemd adopt the `Runner.Worker` into `runners.slice/fleet-job-<unit>-<pid>.scope`, bound to the unit so a stop of the unit still ends its job. A refusal fails the job by name rather than run it in the listener's cgroup.[^scope]
- **`StartTransientUnit` answers when the start job is queued, not when the Worker has moved.** An immediate read failed every job on each placed runner from 17:43Z on 2026-10-17, so the helper now polls the cgroup every tenth of a second for up to `MOVE_WAIT_SECONDS = 5`.[^move]
- **A running unit cannot be moved in place.** After the drop-in, `systemctl show` reads `Slice=listeners.slice` while the `ControlGroup` stays under `runners.slice`, and lavender's runners go from one job straight into the next. The host reaper's 30 s pass restarts such a unit only when its Listener runs and no Worker is in its cgroup or scopes; no Listener means a self-update, which a restart would cut short.[^place] A single read per pass missed the seconds between jobs: MCPs lavender-wsl-3 held a Worker at every pass for 82 minutes. So after reaping, the pass re-reads each busy unit owed a move five times a second for up to `PLACE_WAIT_SECONDS = 20`.[^watch]

## Proof the machine writes

A note transcribing these readings was rejected as proof by two reviews of the task's closure. The proof is now a fleet project, `tools/fleet-execution-runners`, pinned to lavender-wsl, whose two cases run on the host as the fleet user.[^host-cases] One requires every unit `runners.json` declares for lavender to be active in `/listeners.slice`, that slice at `memory.high` max, `runners.slice` at the roster's bound, and every live Worker in its own job scope. The other fills a slice of the fleet user's systemd manager past its high with swap denied, beside an unbounded sibling. It requires the probe inside to stall at least a second and the sibling at most a hundredth of that. Run before lavender-wsl-3 moved, the first case failed on that unit, which is the fault it exists to catch. Its run at 3e4066d4a (fleet job c43c18f2) read 8 of 8 units in `/listeners.slice` and 4 Workers each in its scope, and the probe inside a slice over its high stalled 7,630,156 µs against the sibling's 0.[^a3]

## Ending a job GitHub dropped

The host reaper ends a Worker only past the host's six-hour job timeout, because the host alone cannot tell a long job from a dropped one. GitHub can.[^dropped] A Worker older than `DROPPED_GRACE_SECONDS = 300` holds a dropped job when it is its unit's newest and GitHub reads its runner `busy=false` (online or offline) or does not list it, or when a newer Worker shares its unit, since a runner runs one job at a time.[^rule]

Every hub tick runs a reap pass that reads the roster **at the rolled ref**, reads each host against GitHub, reads again `CONFIRM_SECONDS = 120` later where a Worker was named, and ends each Worker both readings name with the same pid and start through the reaper's `--dropped UNIT PID START`, which writes each pid's argv before any signal and leaves the listener running.[^reap] The audit row shares the reading and says how many runners it examined.[^shared]

**Offline is not dropped.** The first rolled pass, at 18:26:33Z on 2026-10-17, counted `offline busy=true` as dropped and ended two Workers GitHub still held, each failing on GitHub within two seconds. A listener out of touch reconnects and its job outlives it, so a runner GitHub reads busy is never reaped.[^offline]

[^ci-slice]: `tools/fleet/src/fleet/contracts/runner_slice.py:41` -- `CI_SLICE_NAME = "runners.slice"`.
[^why]: `tools/fleet/src/fleet/core/runner_job_scope_render.py:3-8` -- module docstring: "the slice sat at its ``MemoryHigh`` and every listener's calls to"
[^low]: `tools/fleet/src/fleet/contracts/runner_slice.py:45-51` -- comment on `LISTENER_SLICE_NAME`: "the budget its jobs fill timed out its calls to GitHub (2026-10-02 and"
[^listener-slice]: `tools/fleet/src/fleet/contracts/runner_slice.py:54-55` -- "This slice carries no bound. No dash, so it is no child of either."
[^scope]: `tools/fleet/src/fleet/core/runner_job_scope_render.py:17-33` -- numbered steps 1 to 3: "A refusal fails the job by name, before its first step, rather than running"
[^move]: `tools/fleet/src/fleet/core/runner_job_scope_render.py:76-83` -- `MOVE_WAIT_SECONDS` comment: "move to land. ``StartTransientUnit`` answers once the scope's start job is"
[^place]: `tools/fleet/src/fleet/core/runner_reaper_render.py:285-300` -- `place()`: "# No Listener is a self-update under way, which a restart would cut short."
[^watch]: `tools/fleet/src/fleet/core/runner_reaper_render.py:77-83` and `:120` -- module docstring: "A pass that only looked once missed the gap between two jobs, which lasts"
[^host-cases]: `tools/fleet/tests/test_runner_placement_host.py:182-211` and `:294-341` -- the two `host_runner_units` cases: "assert outside_stall * SIBLING_RATIO <= inside_stall"
[^a3]: [synthesis] -- the two `host reading:` lines of fleet job c43c18f2 (tools/fleet-execution-runners at 3e4066d4a, node lavender-wsl, 2 passed), in its log `/home/corvis/fleet/stage/logs/tools-fleet-execution-runners-lavender-wsl-1792267202.log` on lavender-wsl; the repo carries the cases that print them, not the figures.
[^dropped]: `tools/fleet/src/fleet/core/runner_reap_dropped.py:3-8` -- module docstring: "the host alone cannot tell a long job from one GitHub let go of; on"
[^rule]: `tools/fleet/src/fleet/core/runner_dropped_check.py:15-20` and `:45` -- "A runner GitHub reads busy holds its job however its listener reads."
[^reap]: `tools/fleet/src/fleet/core/runner_reap_dropped.py:12-22` and `:61` -- "writes each pid's argv before it kills the Worker's tree and leaves the"
[^shared]: `tools/fleet/src/fleet/core/runner_dropped_check.py:22-26` -- "(:func:`attempt_dropped_check`), one per host, which says how many runners it"
[^offline]: `tools/fleet/src/fleet/core/runner_dropped_check.py:193-214` -- `_github_said` docstring: "counted ``offline busy=true`` as dropped, ended two Workers GitHub still"
