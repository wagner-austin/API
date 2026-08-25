---
title: ClearGBM min_data_in_bin — the coarseness dial recovers the retired lead
tags: [ml, cleargbm, binning, regularization, config-honesty, roadmap-p6]
related:
  - "[[cleargbm-count-aware-binning]]"
  - "[[cleargbm-program-charter]]"
  - "[[cleargbm-decorative-knob-class]]"
source_paths:
  - libs/cleargbm_rs/src/binning/edges.rs
  - libs/cleargbm_rs/src/training/config_rules.rs
  - services/covenant-radar-api/scripts/cv_external.py
  - libs/cleargbm/docs/BENCHMARK_RESULTS_2026-08-25_min_data_in_bin.md
fact_checked: "2026-08-25"
confidence: high
hubs: [libs]
---

# ClearGBM min_data_in_bin — the coarseness dial recovers the retired lead

Config field 25, born from a measurement: [[cleargbm-count-aware-binning]]
retired rw_matches' 2-point lead as accidental coarse-binning
regularization and named the missing dial. This landing builds it under
the config-honesty rules and measures it on the corpus that named it
(board task `1ad15eb6`).

## The knob

`min_data_in_bin` (required-with-null; shipped `GreedyFindBin`
semantics): when set (>= 2), rare adjacent distinct values merge until
every bin holds at least the floor, and the greedy budget caps at
`n / floor`. **Some(1) is rejected in both the Rust rules and the
Python decoder** — it aliases null, and two spellings of one behavior
would make configs lie about themselves. The artifact records the
floor; pre-field artifacts decode as null in Python (absence keeps its
meaning) while the strict Rust serde boundary requires the field, as
with every prior schema field. Knob-sensitivity tests hold at the Rust,
Python and covenant-backend layers — the Python one exists because a
mechanical edit briefly hardcoded None at the Rust boundary (the exact
[[cleargbm-decorative-knob-class]] shape) and the detector caught it
before any gate. On the covenant wire the field is `NotRequired`;
radar's `cv_external` takes it as an optional fifth argument, refused
below 2 and for the LightGBM backend.

## The equivalence anchor and the curve

At null (and floors 3/8/32/1024) the flagship protocol reproduces the
count-aware landing's 0.7295 ± 0.0728 fold-for-fold — conventional
floors are INERT at 450k-rows-per-fold scale, where every counter value
holds thousands of rows and `n / floor` sits far above the 64-bin
budget. The dial bites near `n / max_bins` (~7k) and above:

| floor | mean held-out AUC |
|---|---|
| null … 1024 | 0.7295 |
| 4096 | 0.7297 |
| 16384 | 0.7355 |
| **32768** | **0.7460** |
| 65536 | 0.7424 |

Floor 32768 recovers most of the retired accidental 0.7492 (Δ0.003,
inside the ±0.079 fold spread) and sits clearly above LightGBM's
0.7299 on the identical protocol, with a real optimum (65536 declines).
Framed precisely: the MATCHED-protocol standing stays the tie — the
floor is a tuned dial, and LightGBM's own floor was not swept — but
what the old binning did by accident is now a stated, tunable,
artifact-recorded config value the optimizer can sample per corpus.

## Gates

cleargbm_rs full gate (100.00% segment coverage), cleargbm 257 tests
(the Rust-boundary key census is 26; test_types_model.py split by role
at the 600-line ceiling), covenant_ml 2,542, radar full gate — all
green at landing. Full numbers:
`BENCHMARK_RESULTS_2026-08-25_min_data_in_bin.md`.
