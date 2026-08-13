---
title: Project History
tags: [architecture, codebase, history]
related:
  - "[[coding-standards]]"
  - "[[session-state-deglobalisation]]"
  - "[[package-layering]]"
fact_checked: "2026-08-10"
confidence: high
hubs: [codebase, architecture]
---

# Project History

Why this page exists: `log.md` records **what happened on a date** and is 4,250 lines long. It never says **how the bot got its shape**. On 2026-08-10 that gap cost real time — two analyses of `capture/trackers/` were built and thrown away before anyone checked that the trackers *predate* the live decode path, which was the fact that actually answered the question.[^1]

This page is a map, not a second copy. Detail lives in `log.md` and the subsystem pages; a duplicate would rot.

## Scale, as of 2026-08-10

| | |
|---|---|
| First commit | 2025-12-30, `070bccf7` — "Add TankpitBot project configuration with strict type checking" |
| Commits touching the project | **1,320** |
| Source | ~109,582 lines |
| Tests | ~143,314 lines, 5,980 tests, 100% line + branch coverage |
| Wiki | 72 content pages, 6 hubs |
| Guard rule modules | 8, all wired into `make check` with no allowlist |

Tests outweigh source by ~1.3×. That is a consequence of the 100% coverage gate with no exemptions ([[coding-standards]]).

## The eras

### Strict typing before features (2025-12)

72 commits. The very first commit is project configuration — mypy strict with every `disallow_any_*`, ruff bans on `Any`/`cast`/`TypeAlias`. The type discipline predates the code it governs, which is why there has never been a migration to it.[^2]

### Everything at once (2026-01)

134 commits. On **2026-01-12** five packages land the same day: `capture/`, `sniffer/`, `protocol/`, `state/`, `browser/`.[^3] This is the reverse-engineering phase — the question was "what is this game even sending?"

`capture/trackers/` belongs to this era and explains its own shape: each tracker is `process_message(payload) -> str | None`, base64 in, a **human-readable line** out. That is what a person reading a capture by eye wants. It is not what a bot wants.

`bot/` follows on 2026-01-17.

### The decision layer (2026-03 → 2026-04)

113 then **317** commits — the largest month in the project. `bot/ai/` arrives 2026-03-24; `action_lab/` and `replay/` on 2026-04-10.

The pivotal structural event is **`sniffer/world_state_dispatch.py` on 2026-04-04**: structured `WorldStateDict` updates, which is what an AI layer can consume. It superseded every tracker for practical purposes. Eleven of the twelve were then orphaned but not deleted — the package `__init__.py` re-export made unreferenced classes look referenced, which is why the 2026-08-07 shim sweep (`0ee86133`, "Four re-export shims and a pass-through decoder deleted") walked straight past them. They were finally removed 2026-08-10, −4,270 lines.[^1]

`action_lab/` is worth its own note: isolated probes, built because the integrated bot was too broken to trust. Diagnostic infrastructure, not the production path.

### The recovery commit (2026-06)

101 commits, and one of them is **`62643d2c`, 2026-06-14, "Recover codebase from transcript after accidental wipe."** The codebase was reconstructed from a conversation transcript. `diagnostics/` dates from this month.

`wiki/log.md`'s own record starts 2026-06-16, two days later — so the wiki has no entries covering anything before the recovery.[^4]

### The eight weeks with no commits (cause unestablished)

This is the one hole in the record. What follows separates what is measured from what is not, because an earlier draft of this page asserted a narrative it could not support.

**Measured.** The last TankpitBot commit before the gap is **2026-04-19**; the next is **2026-06-14**, `62643d2c`, whose message reads *"Recover codebase from transcript after accidental wipe"*. Between those dates, zero commits here. That commit changed 283 files, of which **277 are under `clients/TankpitBot`** (+43,279 / -7,215 lines).[^8]

**Not established: what the "accidental wipe" was.** That commit message is the only evidence in this repository, and Austin did not recall the event when asked on 2026-08-10. One line of the diff argues against reading it as rubble recovery: `+0 -876 _test_hooks.py` alongside `+151 _test_hooks/__init__.py` is a file-to-package **refactor**, which is ordinary work. So the commit may be a large batch of uncommitted work that was reconstructed, and the word "wipe" may be carrying more narrative than the facts support. **Do not build on it.**

What can be said either way: substantial TankpitBot work in that window has no incremental history. Whatever the cause, it arrived as one commit -- no messages, no diffs, no ordering, no record of what was tried and abandoned.

**The work did not stop; it moved.** Other projects live in separate repositories that a count here cannot see. `~/PROJECTS/MCPs` logged **910 commits inside this same Apr 19 - Jun 14 window**, including 77 in May (`data-platform` migrations 086-090, watcher DRY cleanups), and April was MCPs' own peak at 866 -- the same month TankpitBot peaked at 317. `~/PROJECTS/Dashboards` accounts for February the same way, at 469 commits. There are 60 project directories under `PROJECTS/`, so "the repo was idle" is never a safe inference from one repository's log.[^9]

Git's `--lost-found` holds 12 unreachable commits, all 2026-07 and 2026-08, orphaned by amends and rebases during recent work. None predate the gap, so nothing in this repository recovers it.

### Instrumentation and doctrine (2026-07)

**379 commits, 251 log entries** — the densest month by both measures. Seven packages arrive: `service/` (07-13), `contracts/` and `ledger/` (07-18), `physics/` and `validate/` (07-21), `sim/` (07-22).

The shift is from *making the bot work* to *being able to prove what it did*: a validation layer, a physics layer with explicit laws, a simulator, an append-only ledger. Behaviour contracts get written down and become binding ([[bot-behavior-contract]]).

### Enforcement (2026-08)

204 commits. `analysis/` arrives 08-06. The theme is turning held rules into machine-checked ones, after the discovery that **an unenforced rule rots**: the 400–600 line ceiling went from a 40-file backlog to 77 in the six days it was documented-but-unchecked.[^5]

Landmarks: the test suites split to mirror their modules (08-06, a dozen commits); the last nine over-bar modules split (`4a92e194`, 08-07); seventeen module-scoped session globals reach zero ([[session-state-deglobalisation]]); the coverage omit list deleted, because an exempted gate measures nothing.

Two guard rules were added in this era from defects found the hard way: `file_size_rules.py`, and `hook_restore_rules.py` after a `_test_hooks.remove_file` swap leaked across an xdist worker without 6,171 tests at 100% coverage noticing.[^6]

## The through-line

Three lessons recur across all eight months, each learned more than once:

1. **A covered line is not a pinned line.** 100% coverage proves a line executed and says nothing about whether any assertion depended on it. Only mutation testing distinguishes them, and a 37-guard sample found six guards no test could tell from absent.[^6]
2. **An unenforced rule rots.** Every standard that stayed prose drifted; every one wired into `make check` with no allowlist held at zero.
3. **A detector that has not failed on a known-bad input is not evidence.** The recurring failure mode is a confident zero from a detector pointed at the wrong thing — a filter reading a field that does not exist, a probe matching the wrong process name, a sweep handed a path it could not resolve.[^7]

[^1]: `git log --diff-filter=A -- src/tankpit_bot/capture/trackers/` gives 2026-01-12; `-- src/tankpit_bot/sniffer/world_state_dispatch.py` gives 2026-04-04. The tracker shape is visible in the surviving `capture/trackers/mine.py` (`process_message(self, payload: str) -> str | None`). The deletion and both discarded analyses are recorded in the `[2026-08-10] audit` entry in `log.md`. `MineTracker` survives because `sniffer/core.py:244` pipes its output to `log.info`; the comment at `sniffer/core.py:116` states "the bot never reads it".
[^2]: `git log --format=%ad --date=short -- .` tail gives 2025-12-30, `070bccf7`. The enforced set is `pyproject.toml` `[tool.mypy]` and `[tool.ruff.lint.flake8-tidy-imports.banned-api]`; see [[coding-standards]] footnote 1 for the field-by-field list.
[^3]: `git log --diff-filter=A -- clients/TankpitBot/src/tankpit_bot/<pkg>` per package directory, first run 2026-08-10 and re-run 2026-08-12 with the adding commit recorded for each. Same-day arrivals 2026-01-12: `capture` `389231df`, `sniffer` `a1c32e9a`, `protocol` `b42e09de`, `state` `9b163fe3`, `browser` `c47c1e20`. Then `bot` `0e3cedd0` 2026-01-17; `bot/ai` `06e88b9f` 2026-03-24; `action_lab` `c4e5a144` and `replay` `da8d5ccd` 2026-04-10; `diagnostics` `62643d2c` 2026-06-14; `service` `89ab2715` 2026-07-13; `contracts` `b2da7f5b` and `ledger` `2f1bb1ee` 2026-07-18; `physics` `bf339b8f` and `validate` `9ee3aeb0` 2026-07-21; `sim` `4e88af99` 2026-07-22; `analysis` `cb49da1f` 2026-08-06. **One of these dates is not an arrival.** `diagnostics` resolves to `62643d2c`, the "Recover codebase from accidental wipe" commit this page flags as the hole in the record — so 2026-06-14 is the date the package re-entered git history, and its true creation date is not recoverable from this repo.
[^4]: `wiki/log.md:7` is the first entry, dated 2026-06-16. Commit counts per month from `git log --format=%ad --date=format:%Y-%m -- .` piped through `uniq -c`: 2025-12 → 72, 2026-01 → 134, 2026-03 → 113, 2026-04 → 317, 2026-06 → 101, 2026-07 → 379, 2026-08 → 204. The two apparent gaps are different in kind, verified 2026-08-10 by comparing project-scoped against repo-wide counts: **2026-02** had 0 TankpitBot commits but **114 monorepo commits** and **469 in `~/PROJECTS/Dashboards`** — that month's attention was Dashboards. **2026-05** had 0 commits in this monorepo, but **71 in `~/PROJECTS/MCPs`**, a separate repository this repo's history cannot see -- corrected 2026-08-10 after an earlier draft called the month idle. It also falls inside the hole described in § The eight weeks git does not have. Project scale in context: the monorepo's own first commit is 2025-12-05 (`5c68b713`, monorepo-guards), 25 days before TankpitBot's, and TankpitBot is 1,320 of the repo's 2,978 commits (~44%).
[^5]: [[coding-standards]] § Code style, file-size bullet, and its footnote 4 citing `log.md:2426-2428` for the ruling and `log.md:2437` for the 40-file enumeration.
[^6]: [[coding-standards]] § Testing, and the `[2026-08-08] audit` plus `[2026-08-09] update` entries in `log.md`. The mutation sweep was at 28 of 474 guards when this page was written, with 0 survivors in that tranche.
[^7]: Enumerated with specifics in the `[2026-08-09] update` and `[2026-08-10] audit` entries of `log.md`.
[^8]: `git log -1 62643d2c^ -- .` gives 2026-04-19 as the last pre-wipe commit; `62643d2c` itself is dated 2026-06-14 with the message "Recover codebase from transcript after accidental wipe" and `git show --stat` reports 283 files changed, 46,957 insertions, 7,215 deletions. `git fsck --lost-found` returns 12 dangling commits, all dated 2026-07-25 through 2026-08-07 — checked individually, none predate the wipe.
[^9]: Per-repository counts, first run 2026-08-10 and **re-measured 2026-08-12 against pinned commits, which corrected two figures.** `~/PROJECTS/MCPs` (its own git root) at commit `3f34c38c` — the commit where its total is 2,242, matching the original measurement — and `~/PROJECTS/Dashboards` at `d0167cd5`. All figures are by **author date** (`git log --format=%ad`); this is load-bearing, because commit date differs after a rebase and gives 231/554 where author date gives Dashboards' 316/469.

    Reproduced exactly: MCPs total 2,242; MCPs April 866 (its peak month); Dashboards 2026-01 316 and 2026-02 469. **Not reproduced: the original "880 commits Apr 19 - Jun 14" and "71 in May."** Re-measurement at `3f34c38c` gives **910** and **77**, by author date, commit date, and with `--no-merges` alike — the three agree, so it is not a date-basis or merge artifact. The nearest window that lands near 880 ends 2026-06-13 (882); June 14 alone carries 28 commits, that being the wipe-recovery day. The original query is not recoverable, so the body now carries the re-measured figures. The direction of the correction does not affect the argument — 910 is more support for "the work moved", not less. Directory count under `PROJECTS/` from `ls -d`, 60 on 2026-08-12 (the page previously said "roughly 58").
