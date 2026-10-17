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
source_git_blobs:
  "tools/fleet/src/fleet/contracts/runner_slice.py": b6b2115238d91a8485cb24fc2ed5edac0bbf9fea
  "tools/fleet/src/fleet/core/runner_job_scope_render.py": aadaf3447a0ae0923c980fd772bcbc86806013f8
  "tools/fleet/src/fleet/core/runner_dropped_check.py": e946b9ae156b05042934830476b9d9f0f24335d1
  "tools/fleet/src/fleet/core/runner_reap_dropped.py": 32f8ee0d947fbc9283c91230e531d3aa2f4e6ada
  "tools/fleet/src/fleet/core/runner_reaper_render.py": 15922304e5b5e77d9641d22c373cc638ac16a22e
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
- **A running unit cannot be moved in place.** After the drop-in, `systemctl show` reads `Slice=listeners.slice` while the `ControlGroup` stays under `runners.slice`, and lavender's runners go from one job straight into the next. The host reaper's 30 s pass restarts such a unit only when its Listener runs and no Worker is in its cgroup or scopes; no Listener means a self-update, which a restart would cut short.[^place]

Measured on lavender-wsl at 18:21:52Z on 2026-10-17, with `runners.slice` at 17,243,774,976 bytes over its 17,179,869,184-byte high: the slice's `memory.pressure` read `full avg10=95.27`, and `listeners.slice` and each placed listener read `0.00`.[^a3]

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
[^place]: `tools/fleet/src/fleet/core/runner_reaper_render.py:269-283` -- `place()`: "# No Listener is a self-update under way, which a restart would cut short."
[^a3]: [synthesis] -- read on lavender-wsl from `/sys/fs/cgroup/runners.slice` and `/sys/fs/cgroup/listeners.slice` (`memory.current`, `memory.high`, `memory.pressure`) by board task 715b9ec5's A3 probe; no file in the repo carries the reading.
[^dropped]: `tools/fleet/src/fleet/core/runner_reap_dropped.py:3-8` -- module docstring: "the host alone cannot tell a long job from one GitHub let go of; on"
[^rule]: `tools/fleet/src/fleet/core/runner_dropped_check.py:15-20` and `:45` -- "A runner GitHub reads busy holds its job however its listener reads."
[^reap]: `tools/fleet/src/fleet/core/runner_reap_dropped.py:12-22` and `:61` -- "writes each pid's argv before it kills the Worker's tree and leaves the"
[^shared]: `tools/fleet/src/fleet/core/runner_dropped_check.py:22-26` -- "(:func:`attempt_dropped_check`), one per host, which says how many runners it"
[^offline]: `tools/fleet/src/fleet/core/runner_dropped_check.py:193-214` -- `_github_said` docstring: "counted ``offline busy=true`` as dropped, ended two Workers GitHub still"
