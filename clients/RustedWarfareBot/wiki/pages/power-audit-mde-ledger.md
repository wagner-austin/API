---
title: "Power audit — every flat verdict beside its minimum detectable effect"
tags: [methodology, experiments, statistics, ledger]
related:
  - "[[campaign-ledger]]"
  - "[[policy-determinism]]"
  - "[[policy-verdict]]"
source_paths:
  - "wiki/sources/power-audit-2026-09-08/paired-stats.txt:8"
  - "wiki/sources/power-audit-2026-09-08/paired-stats.txt:75"
  - "wiki/sources/power-audit-2026-09-08/binary-and-zero-failure.txt:7"
source_git_blobs:
  "wiki/sources/power-audit-2026-09-08/paired-stats.txt": "bf9bfeb578dafdaa4ef5c1b18b6e817dc1aecde3"
  "wiki/sources/power-audit-2026-09-08/binary-and-zero-failure.txt": "ad77b1b7e06c434289d5e4f56b322dea77e92dae"
provenance:
  - "wiki/log.md — the dated verdict entries this page audits ([2026-09-02] through [2026-09-08]). Deliberately NOT a pinned source_path, per campaign-ledger's own precedent: log.md is append-only, so a pin goes stale on every append with no information in the drift."
  - "runs/sweeps/<batch>/ scorecard mirrors (untracked by design; masters /pub/wagnera3/rusted/runs/sweeps) — the raw inputs; their distillation is the pinned paired-stats.txt above."
game_version: "1.15 (code 176, build #28)"
fact_checked: 2026-09-08
confidence: high
hubs: [bot-architecture]
---

# Power audit — every flat verdict beside its minimum detectable effect

Board task 05020324 (child of 9d34f1bb): a null without its minimum detectable
effect is a sentence about the instrument that reads like a sentence about the
world. Every paired statistic below was recomputed from the repo-local
scorecard mirrors, not transcribed from prose; the published figures verified
to the digit wherever both exist.[^1]

**Stated threshold (SEI, smallest effect of interest): 340 samples of paired
survival** — the smallest effect this campaign ever acted on (the e3standrep
adoption leg, +340.0 at t=2.56, recomputed +340.0 at t=2.53 on the sample
divisor).[^2] Margin-unit SEI: 0.27 worth, the smallest margin ever advanced
to confirmation (strike5000).[^3] TESTED means MDE < SEI.

## The verdict: the panels are clean, the screens carry the exposure

**TESTED — every 48-pair verdict.** imp48c6, imp48c0f4, imps5k48, impden48
(both arms), imprb48 (both), impstrike48 (both), impbrace48b, impe1v48,
impincome96, wwspc96, tacpanel96: own-spread MDEs run 180–270 samples, all
under the 340 SEI,[^1] most under the campaign's own era-scoped floor pricing
([[policy-determinism]]). Nothing at adoption scale could have hidden in any
of them. The two survival adoptions (+510.6 t=4.15, +340.0 t=2.53 on the
sample divisor) clear their own panels' MDEs outright.[^1]

**NOT TESTED at the SEI — the screen-tier closures.** imphunt60b (MDEs
264–494 across four arms, two above SEI), wwtac60 (598–818, all above),
wwait24 (708), impbank36 (657/613), impopen96 (253–583).[^1] Each decision stands
— every one refused an arm rather than asserting a zero, and refusal is safe
at any power — but "flat"/"closes" is not established at these n; only "not
adoptable on this evidence" is. Three specific corrections, logged 2026-09-08:

- **wwait24's published MDE figure (~382) used the identical-pair floor where
  its own cross-doctrine spread (sd 1,113.9) gives MDE 708.** Verdict
  unchanged (t=0.88 clears neither), but the operative number is 708.
- **The Hard pace question was "CLOSED as noise" on a test with 0.44 power
  against its own observed 25:13 split** (p=0.073, n=120 pairs, recomputed
  from scorecards).[^4] Noise was never certifiable there; the honest state
  is lean-against-v8, unresolved, v8 retained on detection value. Ledger
  row amended.
- **impopen96's tankfirst read (+263.1, t=+2.29, at its own MDE boundary of
  253) is the one nominally-significant screen positive never taken to a
  panel** — e1's weaker margin lean got one and died there. Under the
  four-mirage record (screens 0-for-4 at panels) skipping it was defensible;
  it is recorded here as an open cheap question, not an error.

Also for the record: g4m2imp48's margin lean (+0.210, sd 0.607, n=48, from
the log's own figures) is nominally t=2.40 — the ledger already reads it as
a lean, and its "no transfer" claim is the exact 0/48-wins count, which needs
no t; the survival lean recomputes to +102.8 at t=+0.98.[^1] And some log
entries published population-divisor sds (e3stand96 842.6 vs 851.5 sample;
impbank36 990.4 vs 1034.4) — a ~2% convention drift that changes no
verdict.[^5]

## Outcome types — the umbrella's 22:06Z correction, applied

The t-formula is the paired-CONTINUOUS instrument only; the umbrella's
correction names two more, and rusted uses all three.[^8] Everything in the
tables above is genuinely continuous (paired survival, paired margin), so no
classification changes. The Hard pace analysis already used the binary
instrument (exact binomial on discordants). Two additions land with the
correction: **fac2x2's win-based "every pairing flat / no interaction" is NOT
TESTED at the +4 adoption bar** — its discordant counts (d = 21/21/28) give
smallest rejecting splits of 16:5 / 16:5 / 20:8, i.e. minimum detectable
win-deltas of 11–12, so bar-scale effects were invisible; the knob chapter's
closure rests on the f2vh48 bar decision, a decision rule rather than a null
certification.[^8] The omission was structural, not arithmetic: fac2x2 sat
enumerated in the evidence table while this page's TESTED/NOT TESTED lists
are prose, so a row can fall out between inventory and verdict with nothing
announcing the hole. The durable fix is a classification column in the
enumeration itself — an unclassified row should be a visible blank, not an
absence — and the shared helper's table format should carry that column.[^8]
And the zero-failure claims get their exact bounds: a
0/48-wins panel excludes a true win rate above 6.05%, the 0/~120 era above
2.5% (1−α^(1/n)); byte-identity claims rest on construction plus the pinned
campaign e2e, never on a t over sd = 0, which would read as infinite
power.[^8]

## Cross-boundary table

Every boundary this project's comparisons cross has a measured term or a
declared scope — the audit found none unaccounted: frozen payload trees
(measured, raid8 +217→−198 across trees; law nine makes fresh-tree replication
the adoption bar), nodes and processes (measured into the floor — detpair24's
same-node forks are inside it), the agent/weave regime change (declared a new
comparability regime 2026-09-06; floors era-scoped, detpair24 sd 1,205 before,
detpair24b sd 661.8 after, and historical panels keep their eras' pricing),
Windows-vs-Linux figures (declared non-comparable on the ledger), the game
build (three sha256 fingerprint axes in every RunRecord via `rw_bot.provenance`),
and the instrument itself (its own draws measured, retracted, and fixed
2026-09-07). No comparison reports a p-value across an unmeasured boundary.[^6]

## Application note

This project independently derived the audit's rule before the audit existed:
detpair24 (2026-09-06) turned the measured noise floor into "the minimum
detectable effect for every future screen," and every verdict since sizes
against it. The exposure that remains is uniform: screen-tier reads at n=12
carry MDEs of 300–800 samples on their own spreads, so a screen can refuse an
arm but can never close a question at the SEI — the campaign's own five-for-five
record of law nine killing screen positives is the same fact from the other
side.[^7] The prose rule going forward is one clause long: a screen verdict
states its own-spread MDE beside its t, or it is an advertisement.

[^1]: `wiki/sources/power-audit-2026-09-08/paired-stats.txt` L8–57, the survival-unit table: batch, arm, n, mean, sd (n−1 divisor), se, t, MDE = t_crit(df, two-sided 95%)·sd/√n; per-seed differences for the screen-tier reads at L59–73.
[^2]: e3standrep row, `paired-stats.txt` L53; the adoption itself: `wiki/log.md` § "[2026-09-06] ADOPTION | evolve3-g3m10 takes the Impossible standing base".
[^3]: `wiki/log.md` § "[2026-09-03] search | impsearch1" (+0.317 → +0.271, the only margin ever advanced) and § "[2026-09-03] verdict | strike5000 REFUTED at confirmation".
[^4]: `wiki/sources/power-audit-2026-09-08/paired-stats.txt` L75–82, the ab24/ab48/ab48b win-pair block: per-wave discordants 7:2 / 9:4 / 9:7, combined 25:13, exact sign-test p=0.073, power 0.44 at the observed 0.66 split.
[^5]: sd comparisons at `paired-stats.txt` L5–6 (the divisor note) against `wiki/log.md` § "[2026-09-06] measurement | the graduate stands at evidence tier" (842.6) and § "[2026-09-06] verdict | the bank fires" (990.4).
[^6]: trees: `wiki/log.md` § "[2026-08-09]"-era raid8 retraction restated in § "[2026-09-07] correction"; nodes and floors: § "[2026-09-06] verdict | detpair24 reads the floor"; regime declaration: § "[2026-09-06] measurement | the routing weave ships"; era re-pricing: § "[2026-09-07] verdict | the weaves nearly HALVED the floor"; OS scope: [[campaign-ledger]]'s Hard Linux row; build fingerprints: `src/rw_bot/provenance.py` per RESEARCH.md's rusted entry.
[^7]: `wiki/log.md` § "[2026-09-08] verdict | the spacing panel flattens the screen's +588 to +25" — law nine's ledger at five-for-five against screen positives.
[^8]: `wiki/sources/power-audit-2026-09-08/binary-and-zero-failure.txt` L7–39: the fac2x2 discordant table with smallest rejecting splits, the falsifiability check (all d ≥ 21, above code-style's d=5 floor), the zero-failure bounds, and the sd=0 caution; umbrella correction: board task 9d34f1bb, note of 2026-09-08T22:06:17Z.
