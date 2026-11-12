---
title: A runner job's trees are emptied as it ends, and the host's runner audit runs on the host
tags: [fleet, github-actions, self-hosted-runners, disk, ci-clean, infrastructure]
related:
  - "[[fleet-runner-listener-isolation-and-dropped-jobs]]"
source_paths:
  - tools/fleet/src/fleet/core/runner_work_clear.py
  - tools/fleet/src/fleet/core/ci_clean.py
  - tools/fleet/src/fleet/core/runner_job_scope_render.py
  - tools/fleet/src/fleet/core/runner_reaper_render.py
  - tools/fleet/tests/test_runner_hygiene_host.py
  - tools/fleet/tests/test_runner_audit_host.py
  - tools/fleet/tests/test_runner_disk_host.py
  - tools/fleet/runners.json
source_git_blobs:
  "tools/fleet/src/fleet/core/runner_work_clear.py": d9a63362c811e949198880ac023cad309785e55a
  "tools/fleet/src/fleet/core/ci_clean.py": b8a9d60c39f4b3166b57f009f855ef894e4ca2b4
  "tools/fleet/src/fleet/core/runner_job_scope_render.py": e6a2c892a049383e588ea8d47d828645c088f17e
  "tools/fleet/src/fleet/core/runner_reaper_render.py": 227d7a140ce000cd24092278131aa38039c4ff89
  "tools/fleet/tests/test_runner_hygiene_host.py": 16c5ae8a97684784b9c63810196c10089d418c3b
  "tools/fleet/tests/test_runner_audit_host.py": 9cb400df0deed94ea83aeb9dfbab27bb3b2691af
  "tools/fleet/tests/test_runner_disk_host.py": c4d503617359533a0fdfeefb70dc4b5157461aa1
  "tools/fleet/runners.json": 43fe7f532c4f974ef67d16d9491aee7043fa84bf
provenance:
  - "2026-11-05 runner pin review after the Ubuntu runner recipe was deleted (API 04c8dcf69): the provision's restart of an idle unit lacking a hook (render_hooks_live_lines) is gone, so that sentence is withdrawn and the reaper's between-jobs restart stands alone; the chown -R sentence is now the Ubuntu-era history ci_clean.py records; runners.json's assets lost their provision_command; locators moved with the files; board task 1b7a7624-e767-42b4-a1d9-888b87d6fd32."
  - "2026-11-12 runner pin review: runners.json's base gained pipx among its nix packages, so lavender's disk ceiling moved from line 100 to 101; two lavender MCPs runners gained the mcps-fleet-ci label and a sedona host was appended, neither of which this page's claims rest on; the four assets still sit at 62-91; board task 96ef0809-a111-4120-9d24-5b35bdc408bc."
fact_checked: "2026-11-12"
confidence: high
hubs: [infrastructure]
---

# A runner job's trees are emptied as it ends, and the host's runner audit runs on the host

lavender's root has a 150 GB ceiling in `runners.json`, and on 2026-10-30 the audit's disk row failed it at 190 GB while every bounded area sat inside its own bound.[^ceiling] The runner lane now reads that row on the host itself and fails a root over the ceiling.[^disk-case] What filled the gap was each runner install's `_work` tree, which nothing bounded.

## What the work trees held

The seven installs' `_work` trees held 29437 MB. Two torch venvs left by the last job of each API install made up 12095 MB of it. Each MCPs install kept the sibling clones its jobs make, and every install kept the runner's own self-update under `_work/_update`.[^measured] None of it is read again: every workflow these runners serve checks out with actions/checkout's default `clean: true`, so the leftovers last only until the next job on the same install.[^unread] ci-clean runs beside the jobs, so it cannot remove a job tree without racing a job that may start in it.

## The job-completed hook

- **Where it runs.** The same drop-in that sets the job-started hook also sets `ACTIONS_RUNNER_HOOK_JOB_COMPLETED`.[^dropin] The runner runs it after the job's last step, inside the job, so no other job of that runner can start meanwhile. It empties the job's workspace and removes everything beside it under `_work/<repository>`.[^hook]
- **What it refuses.** A `GITHUB_WORKSPACE` that is not `_work/<repository>/<name>` of the job's own install, so it never empties a workspace a workflow moved elsewhere.[^refuse] The workspace directory itself stays, because the runner made it and a later job runs its steps there.[^stays]
- **Its record.** It writes the instant and what it removed to the install's marker directory, because the runner account's home is closed to the fleet's user and ci-clean reads the record from there.[^record]

## Reaching a runner that is already running

A runner reads its hooks from its environment only when it starts, so a changed drop-in reaches a running unit only through a restart.[^env] The host reaper takes that restart between jobs: a unit whose main process lacks a hook its drop-in sets is restarted on the first read its job has ended, by the same watch that moves units into `listeners.slice` ([[fleet-runner-listener-isolation-and-dropped-jobs]]). At 2026-10-30T20:46Z, when lavender still ran Ubuntu's provision, both MCPs units on lavender were in that state.[^reaper]

## What ci-clean still does

ci-clean removes only the runners' unpacked self-updates (682 MB in each install), and only once no file in one has changed within the grace period. It reports what every install's job trees hold.[^update] Under lavender's Ubuntu provision, its `chown -R` of each install reset every file's change time, so on 2026-10-30 that day's own pass kept all seven updates, and the first daily pass an hour or more after a provision removed them.[^chown] Its venv sweep removes hashed poetry venvs whose source project no longer exists, printing `venv-sweep: removed <n> of <total> (<mb> MB)`. That frees the venvs whose sources the hook has deleted.[^sweep]

## The runner audit, run on the host

`fleet-runners --spec runners.json --host lavender` runs its audit from a session, so the board had nothing it could run again. The runner lane now renders the same audit script, runs it with `sudo -n` on lavender and scores it with the same scorer, requiring every host row to read OK.[^audit] A second case declares again the manual asset `/opt/corvis/rw-game/game-lib.jar`, which `runners.json` dropped because nothing read it after RustedWarfareBot's CI left lavender. That case requires exactly the asset's two rows to drift, so each run shows the audit still catches a missing file.[^control] `runners.json` now declares four assets, none of them manual.[^assets]

## Proof the machine writes

The hygiene case requires every unit's drop-in to name the hook and the installed hook to be the render. It also requires no file in the job trees of any idle install whose hook ran after its last job started.[^hygiene] Fleet job 3e2ab692 (`tools/fleet-execution-runners` at API 796767077 on lavender) passed all 12 cases. lavender's root used 128 GB of its 150 GB, both MCPs installs read 0 files after their hook, the audit's 52 host rows read OK, and the reinstated jar read exactly 2 DRIFT rows.[^job]

[^ceiling]: `tools/fleet/tests/test_runner_disk_host.py:6` -- "2026-10-30 that row failed lavender at 190 GB against 150 while every"; `tools/fleet/runners.json:101` -- `"ceiling_gb": 150,`.
[^disk-case]: `tools/fleet/tests/test_runner_disk_host.py:36` -- `assert used <= disk["ceiling_gb"]`.
[^measured]: `tools/fleet/src/fleet/core/runner_work_clear.py:3-12` -- module docstring: "seven installs' ``_work`` trees held 29437 MB. 12095 MB of it was two venvs"
[^unread]: `tools/fleet/src/fleet/core/runner_work_clear.py:14-22` -- "None of it is read again. Every workflow of every repository a runner here"
[^dropin]: `tools/fleet/src/fleet/core/runner_job_scope_render.py:18-24` -- numbered step 1: "``ACTIONS_RUNNER_HOOK_JOB_COMPLETED``, which empties the job's trees as it"
[^hook]: `tools/fleet/src/fleet/core/runner_work_clear.py:24-29` -- "runs :data:`JOB_COMPLETED_SCRIPT` after the job's last step, in the job,"
[^refuse]: `tools/fleet/src/fleet/core/runner_work_clear.py:29-31` and `:59-72` (`CLEAR_LINES`) -- "``GITHUB_WORKSPACE`` that is not ``_work/<repository>/<name>`` of the job's"
[^stays]: `tools/fleet/src/fleet/core/runner_work_clear.py:36-37` -- "The workspace directory itself stays: the runner made it when the job was"
[^record]: `tools/fleet/src/fleet/core/runner_work_clear.py:32-34` -- "then records the instant and what it removed in the install's marker"
[^env]: `tools/fleet/src/fleet/core/runner_reaper_render.py:85-87` -- module docstring: "reads its hooks from its environment when it starts, so a unit Ubuntu's"
[^reaper]: `tools/fleet/src/fleet/core/runner_reaper_render.py:85-91`, `:134-139` (`HOOK_CASES`, the reaper's `stale_hooks` lines) and `:325-326` (the `place` function's hooks reason) -- "2026-10-30T20:46Z both MCPs units on lavender were. So a unit whose main"
[^update]: `tools/fleet/src/fleet/core/ci_clean.py:59-63` -- "removes only the runners' unpacked self-updates and reports what every"; `:258-265` -- "``_work/_update`` and never removes it: on lavender each install held 682"
[^chown]: `tools/fleet/src/fleet/core/ci_clean.py:266-269` -- "each install changes every file's too: on lavender on 2026-10-30 its own"
[^sweep]: `tools/fleet/src/fleet/core/ci_clean.py:65-66` -- "WHAT IT DELIBERATELY KEEPS: hashed poetry venvs whose source project still"; `:111` -- "``venv-sweep: removed <n> of <total> (<mb> MB)``."
[^audit]: `tools/fleet/tests/test_runner_audit_host.py:41-65` -- `_audit_here` and the first case: `assert _drifted(findings) == []`
[^control]: `tools/fleet/tests/test_runner_audit_host.py:32-38` and `:76-82` -- `DROPPED_ASSET` and `assert drifted == [f"asset:{DROPPED_ASSET['path']}", f"sha256:{DROPPED_ASSET['path']}"]`
[^assets]: `tools/fleet/runners.json:62-91` -- `"assets": [`, four entries, each `"manual": false`.
[^hygiene]: `tools/fleet/tests/test_runner_hygiene_host.py:363-404` -- `assert [f"{name}: {trees[name].files} files" for name in owed if trees[name].files] == []`
[^job]: [synthesis] -- the `host reading:` lines of fleet job 3e2ab692 (tools/fleet-execution-runners at API 796767077, node lavender, 12 passed), in its log `/home/corvis/fleet/stage/logs/tools-fleet-execution-runners-lavender-1793398407.log` on lavender; the repo carries the cases that print them, not the figures.
