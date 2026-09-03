---
title: Feature Corpus Provenance
tags: [architecture, decisions, ml, provenance, corpus, diagnostics]
related:
  - "[[self-observing-architecture]]"
  - "[[coding-standards]]"
  - "[[module-map]]"
source_paths:
  - "src/tankpit_bot/diagnostics/feature_provenance.py"
  - "src/tankpit_bot/diagnostics/feature_rows.py"
  - "src/tankpit_bot/diagnostics/feature_row_types.py"
  - "src/tankpit_bot/_test_hooks/runtime.py"
source_git_blobs:
  "src/tankpit_bot/diagnostics/feature_provenance.py": "e80cee8cd49b694c59f7b7d41199e5416b72d494"
  "src/tankpit_bot/diagnostics/feature_rows.py": "450c70275e522cd78c9ae0dd2d86330594b77d3d"
  "src/tankpit_bot/diagnostics/feature_row_types.py": "a24390e153d46ddfda06b53ce76a9f9f042a9bd4"
  "src/tankpit_bot/_test_hooks/runtime.py": "beac39f5f046adf0f604d39f5ce9a5f3b1250e92"
provenance:
  - "docs/RESEARCH.md — the monorepo research index, and the four-step registration procedure this page is measured against"
  - "clients/RustedWarfareBot/src/rw_bot/provenance.py — the `rusted` precedent, adopted 2026-08-29"
  - "runs/bot/artax/bot-20260806-210413.events.jsonl — the archived artifact the first record was produced from"
fact_checked: "2026-09-02"
confidence: high
hubs: [architecture]
---

# Feature corpus provenance: the record the tick table never carried

*Established 2026-09-02. `tankpit-feature-rows` shipped the tick-level
table on 2026-09-01 and wrote it with nothing attached saying what
produced it.*

## What was wrong

The derivation reshapes an events artifact into one row per tick — 539
runs, 132,266 rows, a design matrix somebody will train on. It wrote
`<stem>.features.jsonl` and stopped: no digest, no fingerprint, no run
record.[^1] Two exports taken from different working trees were
byte-comparable and otherwise indistinguishable, and `hpc3-trace` could
never answer which run produced a given row.

This is not a hypothetical cost. `covenant-radar-api`'s optimisation
history carries 3,068 rows written before its fingerprint existed; they
state `"fingerprint": null` and are permanently unreproducible.[^2] The
corpus here was 539 runs when this was fixed, which is why it was fixed
then.

## The two configurations, and why only one is recorded

The whole design turns on a distinction that is easy to blur:

| | knowable? | how it is recorded |
|---|---|---|
| What produced the EVENTS (a live bot run — build, doctrine, account, rank) | **No** | only as a digest of the artifact |
| What produced the FEATURE ROWS (the derivation) | Yes | fully, as a `RunFingerprint` |

An events record carries `timestamp`, `level`, `logger`, `mode`,
`channel` and `message` — and no build stamp, commit or version
anywhere.[^3] For the runs already in `runs/bot/` that cannot be
recovered, so no fingerprint written today can honestly claim it. The
record therefore describes the derivation and identifies the events
only by `tankpit-events`, a digest of what was read.

Inventing a bot version for the first column is exactly the failure
`docs/RESEARCH.md` exists to prevent, and the same gap it states
plainly for `turkic-lstm`: results already on disk have no sidecar and
cannot be given an honest one retroactively.

**Stamping the build at emission time is the fix for FUTURE runs.** It
belongs in the runtime logging path, not in the derivation, and is not
done.

## The shape

`RunRecord` beside the result on the `run_record_sidecar` convention —
the shared monorepo vocabulary, not a second one. `rusted` took the same
step on 2026-08-29 and `rw_bot.provenance` is the model followed.[^4]

- **experiment** `tankpit-tick-features`, one name across every run, so
  the longitudinal question ("did radar dispatch collapse after that
  change") stays askable.
- **label** `<instance>/<stamp>`, e.g. `artax/bot-20260806-210413`. The
  instance is load-bearing: a fleet runs several bots concurrently and a
  stamp alone would make two runs look like two readings of one.
- **packages** the input digest and the reshaping code's version. Both
  decide the rows — a change to `COUNTED_KINDS` or to the
  last-outcome-wins rule rewrites every row without touching the input.
- **observations** `ticks`, `tick_span`, `tick_density`, `action_ticks`
  and a total per counted kind. Density is recorded because 400 rows
  across 400 ticks and 400 across 9,000 are the same size and are not
  the same run.
- **determinism** stack `tankpit-feature-rows`, stating no
  floating-point arithmetic and ascending row order — not
  `UNPINNED_STACK`, which would say nothing pinned the arithmetic when
  the truth is there is none to pin.

## Two decisions a later reader will want to undo

**The digests are over decoded text, not file bytes.** `write_text` and
`read_text` translate line endings in text mode, so the same logical
table is `\n` on Linux and `\r\n` on this workstation. Digesting the
file as it sits on disk would fingerprint a Windows export and its
identical Linux copy differently — and every cluster-versus-workstation
comparison, which is the comparison the corpus exists for, would report
a changed input that did not change.

**The record is written by the one write path, not a second command.**
An opt-in provenance step is provenance nobody runs. The corpus would
refill with unattributable rows the first time somebody exported in a
hurry.

## Status against the research index

TankpitBot is **not** a registered hpc3 project, and this closes only
step 3 of four.[^5] Steps 1 and 2 (a `tools/hpc3/runs/` entry and a
`RESEARCH.md` section) remain open, deliberately: the bot drives
Playwright against live tankpit.com and is not cluster work. The
cluster-shaped payloads are the headless `sim/` tree and offline
modelling over these rows — and the second probably belongs to the
already-registered `cleargbm` rather than to a sixth project.

[^1]: `src/tankpit_bot/diagnostics/feature_rows.py`, `write_feature_rows` before 2026-09-02 — the body was joined and written with no sidecar. Shipped in commit `3c961bef`.
[^2]: `docs/RESEARCH.md`, the `cleargbm` entry: "The 3,068 rows written before that state `"fingerprint": null` explicitly."
[^3]: `runs/bot/artax/bot-20260806-210413.events.jsonl` line 1, read 2026-09-02: keys are `timestamp`, `level`, `logger`, `mode`, `channel`, `message`, `diagnostic_kind`, `room_id`, `field_image`.
[^4]: `clients/RustedWarfareBot/src/rw_bot/provenance.py`, `arm_run_record` / `sweep_fingerprint`.
[^5]: `docs/RESEARCH.md`, "Adding a research project" — register in `tools/hpc3/runs/`, add a section, emit `RunRecord`s, submit through the CLI.
