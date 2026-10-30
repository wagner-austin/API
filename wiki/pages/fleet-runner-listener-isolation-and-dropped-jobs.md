---
title: Runner listeners sit outside their jobs' budget, and the hub ends a job GitHub dropped
tags: [fleet, github-actions, self-hosted-runners, systemd, cgroups, memory-pressure, infrastructure]
related:
  - "[[monorepo-discipline]]"
  - "[[fleet-runner-disk-hygiene]]"
source_paths:
  - tools/fleet/src/fleet/contracts/runner_slice.py
  - tools/fleet/src/fleet/core/runner_job_scope_render.py
  - tools/fleet/src/fleet/core/runner_dropped_check.py
  - tools/fleet/src/fleet/core/runner_reap_dropped.py
  - tools/fleet/src/fleet/core/runner_reaper_render.py
  - tools/fleet/tests/test_runner_placement_host.py
  - tools/fleet/registry/machines/lavender.json
  - tools/fleet/registry/projects/tools/fleet-execution-runners.json
source_git_blobs:
  "tools/fleet/src/fleet/contracts/runner_slice.py": 7bbc8575b2d86783b82c83623f683a0ad9b3bf8d
  "tools/fleet/src/fleet/core/runner_job_scope_render.py": caef149218d487e076d6f84110e6ec5f3ef0e419
  "tools/fleet/src/fleet/core/runner_dropped_check.py": ea59f5c2549941f7837dd8a99bb414cfd29df883
  "tools/fleet/src/fleet/core/runner_reap_dropped.py": 99720a7f2a335f205c722df15154ecba8f792da0
  "tools/fleet/src/fleet/core/runner_reaper_render.py": 75ddd7944f569bf1a95d5d46d572eddd434489e8
  "tools/fleet/tests/test_runner_placement_host.py": e59308c8c36a312274e53cbd0f298a9dd1ffbe40
  "tools/fleet/registry/machines/lavender.json": f883e91c14cb7a216f58fa93521faef6d63acc1f
  "tools/fleet/registry/projects/tools/fleet-execution-runners.json": baeefbb12b207d6e5b71acd6fc7f60f8c6f99858
fact_checked: "2026-10-30"
confidence: high
hubs: [infrastructure]
---

# Runner listeners sit outside their jobs' budget, and the hub ends a job GitHub dropped

lavender runs eight self-hosted GitHub Actions runners whose jobs share one bounded systemd slice, `runners.slice`.[^ci-slice] They ran in its WSL guest, lavender-wsl, until lavender was reimaged to Ubuntu on 2026-10-19 and the guest went with the Windows install.[^reimage] Until MCPs board task 715b9ec5 the runner units sat in that slice too, `Runner.Listener` included, and the listener is the process that holds each job's lease with GitHub. On 2026-10-02 and 2026-10-16 the slice sat at its `MemoryHigh`, the listeners' calls to GitHub timed out, GitHub dropped the jobs, and their Workers ran on unseen for 1 h 45 min.[^why]

## Why no memory protection could have saved the listener

The kernel throttles every task that charges memory beneath a cgroup over its `memory.high`, whatever that task's own `memory.low`. Measured on lavender-wsl: a process holding 4.5 MB under a 200 MB `MemoryLow` sat 227 s in `mem_cgroup_handle_over_high` beside a hog over its slice's high, while the same process in a slice of its own ran unhindered.[^low] The only fix is a different cgroup.

## The split

- **`listeners.slice`**, unbounded, holds every runner unit; no dash in either name, so neither is a child of the other.[^listener-slice]
- **Each job is moved out of its unit** before its first step: a drop-in sets `ACTIONS_RUNNER_HOOK_JOB_STARTED`, the hook runs a root helper through a one-command sudoers rule, and the helper has systemd adopt the `Runner.Worker` into `runners.slice/fleet-job-<unit>-<pid>.scope`, bound to the unit so a stop of the unit still ends its job. A refusal fails the job by name rather than run it in the listener's cgroup.[^scope]
- **`StartTransientUnit` answers when the start job is queued, not when the Worker has moved.** An immediate read failed every job on each placed runner from 17:43Z on 2026-10-17, so the helper now polls the cgroup every tenth of a second for up to `MOVE_WAIT_SECONDS = 5`.[^move]
- **A running unit cannot be moved in place.** After the drop-in, `systemctl show` reads `Slice=listeners.slice` while the `ControlGroup` stays under `runners.slice`, and lavender's runners go from one job straight into the next. The host reaper's 30 s pass restarts such a unit only when its Listener runs and no Worker is in its cgroup or scopes; no Listener means a self-update, which a restart would cut short.[^place] A single read per pass missed the seconds between jobs: MCPs lavender-wsl-3 held a Worker at every pass for 82 minutes. So after reaping, the pass re-reads each busy unit owed a move five times a second for up to `PLACE_WAIT_SECONDS = 20`.[^watch] Since 2026-10-30 the same watch also restarts a unit whose main process lacks a job hook its drop-in sets, because a runner reads its hooks only when it starts ([[fleet-runner-disk-hygiene]]).[^hooks]

## Proof the machine writes

A note transcribing these readings was rejected as proof by two reviews of the task's closure. The proof is a fleet project, `tools/fleet-execution-runners`, pinned to lavender, whose placement cases run on the host as the fleet user.[^host-cases] Its disk, hygiene and audit cases are on [[fleet-runner-disk-hygiene]]. It was pinned to lavender-wsl until the reimage, and is pinned to lavender again since its eight runners were laid there.[^pin] One case requires every unit `runners.json` declares for lavender to be active in `/listeners.slice`, that slice at `memory.high` max, `runners.slice` at the roster's bound, and every live Worker in its own job scope. The other fills a slice of the fleet user's systemd manager past its high with swap denied, beside an unbounded sibling. It requires the probe inside to stall at least a second and the sibling at most a hundredth of that. Run before lavender-wsl-3 moved, the first case failed on that unit, which is the fault it exists to catch. Its run at 3e4066d4a on lavender-wsl (fleet job c43c18f2) read 8 of 8 units in `/listeners.slice` and 4 Workers each in its scope, and the probe inside a slice over its high stalled 7,630,156 µs against the sibling's 0.[^a3]

## Ending a job GitHub dropped

The host reaper ends a Worker only past the host's six-hour job timeout, because the host alone cannot tell a long job from a dropped one. GitHub can.[^dropped] A Worker older than `DROPPED_GRACE_SECONDS = 300` holds a dropped job when it is its unit's newest and GitHub reads its runner `busy=false` (online or offline) or does not list it, or when a newer Worker shares its unit, since a runner runs one job at a time.[^rule]

Every hub tick runs a reap pass that reads the roster **at the rolled ref**, reads each host against GitHub, reads again `CONFIRM_SECONDS = 120` later where a Worker was named, and ends each Worker both readings name with the same pid and start through the reaper's `--dropped UNIT PID START`, which writes each pid's argv before any signal and leaves the listener running.[^reap] The audit row shares the reading and says how many runners it examined.[^shared]

**Offline is not dropped.** The first rolled pass, at 18:26:33Z on 2026-10-17, counted `offline busy=true` as dropped and ended two Workers GitHub still held, each failing on GitHub within two seconds. A listener out of touch reconnects and its job outlives it, so a runner GitHub reads busy is never reaped.[^offline]

[^ci-slice]: `tools/fleet/src/fleet/contracts/runner_slice.py:44` -- `CI_SLICE_NAME = "runners.slice"`.
[^reimage]: `tools/fleet/registry/machines/lavender.json:24` -- the machine's notes: "replacing the Windows machine and its WSL guest lavender-wsl, which no longer exist".
[^why]: `tools/fleet/src/fleet/core/runner_job_scope_render.py:3-8` -- module docstring: "the slice sat at its ``MemoryHigh`` and every listener's calls to"
[^low]: `tools/fleet/src/fleet/contracts/runner_slice.py:46-56` -- comment on `LISTENER_SLICE_NAME`: "the budget its jobs fill timed out its calls to GitHub (2026-10-02 and"
[^listener-slice]: `tools/fleet/src/fleet/contracts/runner_slice.py:57-58` -- "This slice carries no bound. No dash, so it is no child of either."
[^scope]: `tools/fleet/src/fleet/core/runner_job_scope_render.py:18-37` -- numbered steps 1 to 3: "A refusal fails the job by name, before its first step, rather than running"
[^move]: `tools/fleet/src/fleet/core/runner_job_scope_render.py:105-112` -- `MOVE_WAIT_SECONDS` comment: "move to land. ``StartTransientUnit`` answers once the scope's start job is"
[^place]: `tools/fleet/src/fleet/core/runner_reaper_render.py:314-336` -- the reaper's place function: "# No Listener is a self-update under way, which a restart would cut short."
[^watch]: `tools/fleet/src/fleet/core/runner_reaper_render.py:77-83` and `:133` -- module docstring: "A pass that only looked once missed the gap between two jobs, which lasts"
[^hooks]: `tools/fleet/src/fleet/core/runner_reaper_render.py:85-91` and `:326-327` -- "THE JOB HOOKS FINISH HERE THE SAME WAY (MCPs board task 621902c3). A runner"
[^host-cases]: `tools/fleet/tests/test_runner_placement_host.py:174-212` and `:288-343` -- two of its `host_runner_units` cases: "assert outside_stall * SIBLING_RATIO <= inside_stall"
[^pin]: `tools/fleet/registry/projects/tools/fleet-execution-runners.json:8` -- `"node": "lavender"`.
[^a3]: [synthesis] -- the two `host reading:` lines of fleet job c43c18f2 (tools/fleet-execution-runners at 3e4066d4a, node lavender-wsl, 2 passed), in its log `/home/corvis/fleet/stage/logs/tools-fleet-execution-runners-lavender-wsl-1792267202.log` on lavender-wsl, a guest that no longer exists; the repo carries the cases that print them, not the figures.
[^dropped]: `tools/fleet/src/fleet/core/runner_reap_dropped.py:3-8` -- module docstring: "the host alone cannot tell a long job from one GitHub let go of; on"
[^rule]: `tools/fleet/src/fleet/core/runner_dropped_check.py:15-20` and `:45` -- "A runner GitHub reads busy holds its job however its listener reads."
[^reap]: `tools/fleet/src/fleet/core/runner_reap_dropped.py:12-22` and `:61` -- "writes each pid's argv before it kills the Worker's tree and leaves the"
[^shared]: `tools/fleet/src/fleet/core/runner_dropped_check.py:22-26` -- "(:func:`attempt_dropped_check`), one per host, which says how many runners it"
[^offline]: `tools/fleet/src/fleet/core/runner_dropped_check.py:192-213` -- `_github_said` docstring: "counted ``offline busy=true`` as dropped, ended two Workers GitHub still"
