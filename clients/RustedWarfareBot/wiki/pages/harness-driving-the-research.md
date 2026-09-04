---
title: Driving the Research — The Operating Page
tags: [harness, operations, hpc3]
related: ["[[harness-doctrine-search]]", "[[harness-population-search]]", "[[campaign-ledger]]"]
source_paths:
  - "scripts/search.py"
  - "scripts/search_specs.py:175"
  - "scripts/panel.py:70"
  - "scripts/evolve.py:97"
  - "scripts/batch.py:71"
  - "scripts/pairs.py"
  - "scripts/margin.py"
  - "../../tools/hpc3/runs/hpc3-rusted.json"
source_git_blobs:
  "scripts/search.py": "344576525bb482cbf8b8effd2cf755538665ea1b"
  "scripts/search_specs.py": "81dc71a49259df7dc6761848c509cb4739c17b5c"
  "scripts/panel.py": "fd7c3f0620f205bae6168a5136e6943e066e341d"
  "scripts/evolve.py": "7fdc4885b2a87df729902739fb91867e18cb0fa4"
  "scripts/batch.py": "e3183dff9f12edff09f374e7bf6bd2bb49e09dd9"
  "scripts/pairs.py": "18311147e3897a81608c1d551c83a488000176ed"
  "scripts/margin.py": "1b40454709d859c785bded3e3337b0ed06438736"
  "../../tools/hpc3/runs/hpc3-rusted.json": "6f97e362db8befa63107a201d3f9d25a2014ca1a"
fact_checked: 2026-09-03
confidence: high
hubs: [headless-harness]
---

# Driving the Research — The Operating Page

A fresh session drives everything below from
`clients/RustedWarfareBot/`, via `poetry run`, against the hpc3
workspace `../../tools/hpc3/runs/hpc3-rusted.json`. There are exactly
FOUR canonical commands and no other submission path; each is
kill-resumable (relaunch with identical arguments replays
deterministically and fast-forwards off the cluster's completed work),
because the session harness sweeps long-lived local processes and the
transport to the cluster drops. Free partition only; the operator's
zero-billing ruling stands.

## The four commands

1. **Knob search** (screening; the VH knob space is a measured local
   optimum, so new runs of the `vh` spec need a reason):
   `python -m scripts.search hpc3:../../tools/hpc3/runs/hpc3-rusted.json <spec> <name> <rng>`
   -- specs live in `scripts/search_specs.py` (`vh`, `imp`); add a
   regime by adding a validated spec there, never by editing constants.
2. **Win-bar panel** (the ONLY instrument that adopts):
   `python -m scripts.panel hpc3:<workspace> <batch> <control.doctrine> <arm-label> <arm.doctrine> <pairs> <difficulty>`
   -- lays out fresh disjoint seeds itself, plays, converges its own
   casualties, judges (pairs + margin), one command.
3. **Population search** (the composition simplex):
   `python -m scripts.evolve hpc3:<workspace> <name> <rng>`
4. **Bespoke batches** (factorials, transfer panels): write the sweep
   file with `used_seeds`/`seed_block`/`fresh_seeds` (never inline seed
   picking), commit it, then
   `python -m scripts.batch hpc3:<workspace> <batch> sweeps/<batch>.txt <difficulty>`
   -- the whole freeze/stage/converge-until-full/pull chain in one
   command (the batch name must equal the file's stem, enforced). Judge
   with `scripts.pairs` / `scripts.margin` per the experiment's own
   questions.

## The operating posture (learned the hard way, all measured)

- **Boundary relaunches, not babysitting.** Drivers die (harness
  sweeps, transport drops); the cluster holds all state. Watch drains
  with a persistent Monitor over ssh counts; when a boundary hits
  (drain complete or drained-short), relaunch the same command. It
  fast-forwards in seconds.
- **Preemption is rent, not failure.** `PreemptMode=CANCEL` kills free
  jobs instantly; `--requeue` is inert; `hpc3-campaign` re-run IS the
  requeue and never resubmits finished or running members.
- **Seed disjointness is law.** Panels allocate below 200k, search
  rounds at 200k+, evolution generations at 500k+ -- all mechanical.
  Never hand-pick seeds.
- **Selection evidence is not confirmation.** Cross-round or
  cross-generation consistency inside one search proved worthless twice
  in one night (strike5000, flame2). Only untouched-seed panels dispose,
  against the +4 win bar, then fresh replication (laws six and nine).
- **Every verdict lands in three places** before the work is done: the
  wiki log (dated entry), the campaign ledger (if a rung or question
  moved), and a board post under this session's agent identity.

## Where things live

- Champion per rung: [[campaign-ledger]]. Doctrines: `doctrines/*.doctrine`
  (committed record; `doctrines/search|evolve/` are gitignored run
  artifacts). Scorecards: `runs/sweeps/<batch>/` locally, mirrored from
  `/pub/wagnera3/rusted/runs/sweeps/<batch>/`. Judged by
  `scripts.pairs` and `scripts.margin`.
- Payloads freeze the working tree; every experiment's campaign doc in
  `provenance/` names its payload, and digests are verified end to end.
