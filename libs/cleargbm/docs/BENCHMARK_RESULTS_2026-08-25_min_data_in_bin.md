# min_data_in_bin: the binning-coarseness dial (2026-08-25)

The count-aware binning landing retired rw_matches' 2-point lead as a
coarse-binning regularization artifact and named the missing dial. This
landing builds it: `min_data_in_bin`, ClearGBM config field 25, and
measures whether deliberate coarseness recovers what accidental
coarseness provided. It does — almost all of it.

## The knob

Following the shipped `GreedyFindBin` semantics (the same pinned
source as the binning landing): when a feature's distinct values fit
the bin budget, rare adjacent values merge until every bin holds at
least the floor; otherwise the greedy budget caps at `n / floor` so no
bin can be forced below it. Config honesty, by construction:

- Required-with-null in the engine: null = no floor (the count-aware
  default). **Some(1) is rejected** — a floor of 1 is exactly the unset
  behavior, and two spellings of one behavior would make configs lie
  about themselves. Enforced in the Rust rules AND the Python decoder.
- The artifact records the floor (`"min_data_in_bin":8` in the model
  JSON); artifacts saved before the field existed decode as null in
  Python (absence means what it meant), while the strict Rust serde
  boundary — the continued-training path — requires the field, exactly
  as it did for every prior schema field.
- Knob-sensitivity held at every layer: a Rust train test and a Python
  train test both assert a floored model differs from the unset one
  (the Python one guards the exact bug this landing's mechanical edit
  briefly planted — a hardcoded None at the Rust boundary — before the
  quality gate ever saw it), and an end-to-end probe through the
  covenant backend proves the wire value reaches the artifact.
- On the covenant wire the field is `NotRequired` (configs written
  before it keep their meaning); both cleargbm backends map it;
  radar's `cv_external` takes it as an optional fifth argument — a
  protocol variant, refused for the LightGBM backend (whose own floor
  is not exposed) and refused below 2.

## Equivalence gate: null is bit-identical

`cv_external rw_matches cleargbm` under the new field reproduces the
count-aware landing's number EXACTLY — 0.7295 ± 0.0728, every fold
value identical (0.8281 / 0.6036 / 0.7351 / 0.7226 / 0.7581). The
same held at floors 3, 8, 32 and 1024, which is itself a finding (see
below): not one bin edge moves.

## The measurement: the coarseness curve on rw_matches

Grouped 5-fold CV, production config, weighted — the flagship protocol,
one variable:

| min_data_in_bin | mean held-out AUC |
|---|---|
| null / 3 / 8 / 32 / 1024 | 0.7295 ± 0.0728 |
| 4096 | 0.7297 ± 0.0727 |
| 16384 | 0.7355 ± 0.0747 |
| **32768** | **0.7460 ± 0.0792** |
| 65536 | 0.7424 ± 0.0681 |

Two honest readings:

- **Conventional floors are inert on this corpus.** LightGBM-scale
  values (3-32) change nothing: at ~450k training rows per fold, every
  distinct value of the low-cardinality counters holds thousands of
  rows, and the greedy cap `n / floor` sits far above the 64-bin
  budget. The dial only bites at corpus scale — around `n / max_bins`
  (~7k here) and above, where equal-count bins would fall below it.
- **At scale, the dial recovers the retired regularization.** Floor
  32768 scores 0.7460 — recovering most of the retired accidental
  0.7492 (Δ0.003, well inside the ±0.079 fold spread) and clearly
  above LightGBM's 0.7299 on the identical protocol — and declines by
  65536, so it is a real optimum, not a monotone artifact. What the old
  quantile-of-multiset rule did by accident on duplicate-heavy features
  is now a stated, tunable, artifact-recorded config value.

Scoreboard framing, stated precisely: the MATCHED-protocol standing on
rw_matches remains the statistical tie (0.7295 vs 0.7299) — the floor
is a tuned knob, not part of the matched protocol, and LightGBM's own
floor was not swept here. What this measurement establishes is that
the retired lead was recoverable regularization, now available
honestly, with the optimizer free to sample the dial per corpus.

## The dial reaches the optimizer

The Optuna search space samples the floor as `min_data_in_bin_denom` —
a DIVISOR of the trial's training rows, categorical over {1, 256, 64,
16, 4} with 1 meaning no floor — because the floor's active range
scales with the corpus, so an absolute-valued space would be inert on
large corpora and absurd on small ones. The objective resolves the
divisor against its own training rows (`max(2, n_train // denom)`; 1 →
null), so the trained config always records the honest resolved value,
and the optimal-config record carries `best_min_data_in_bin_denom`.

Landing this surfaced — and fixed — one more member of the dataflow
class: the codebase has TWO ClearGBM sampling layers (the per-backend
`optuna_backend/cleargbm.py` and the strategy registry's
`_tpe_params.py`, which is what radar's production optimize actually
runs), and the dial initially reached only the first. The first
end-to-end smoke tuned without it — caught because the smoke checked
the record for the key, and now pinned by a strategy-layer test plus
the existing sample-then-extract round-trip. The end-to-end smoke
(radar `scripts.optimize`, 5 trials) confirms the record carries the
sampled divisor.

## Rung 4: the tuner uses the dial on the farm

The rung-1/rung-3 grid (4 datasets x 4 presets, 20 Optuna TPE trials
each) re-ran on HPC3 with the dial in the search space — 320 trials,
0 pruned, 0 failed. Best validation AUC per dataset, against rung 3
(same binning, no dial):

| dataset | rung 3 best | rung 4 best | delta |
|---|---|---|---|
| taiwan | 0.9557 | **0.9603** (ratios_only) | +0.0046 |
| us | 0.8438 | **0.8495** (full) | +0.0057 |
| polish | 0.9622 | **0.9646** (none / log_only) | +0.0024 |
| kaggle_give_me_credit | 0.8701 | 0.8695 (log_only) | -0.0006 |

Three of four datasets improved with the dial in the space at the same
20-trial budget; us's 0.8495 is the best us number across every rung,
including old-binning rung 1's 0.8457. kaggle is statistically flat.

**Every winner chose a floor.** The rung-4b replay (the four winning
members re-run under the fixed per-member filenames; per-config
determinism reproduced each best AUC to the last digit) records the
sampled divisor: taiwan, us and polish all tuned to
`min_data_in_bin_denom = 64` (floors of roughly n_train/64: ~74, ~860
and ~76 rows respectively) and kaggle to 256. The improvements are the
dial, not sampler-trajectory luck — on every corpus tested, deliberate
binning coarseness is a quality axis the tuner exploits.

Landing the farm rung surfaced one more infrastructure defect, fixed
at the root: the cluster env's INSTALLED covenant-radar-api package
was still the bootstrap build, predating the farm-filename fix — so
`scripts/` (resolved from the fresh clone's working directory) wrote
per-job-suffixed history files while `save_optimization_results`
(imported from the stale installed package) overwrote one unsuffixed
optimal-config per dataset, last-writer-wins. Per-job logs are
authoritative by design, so no rung result was lost, but the winners'
sampled dial values were — recovered by replaying the four winning
members under the fix (per-config determinism reproduces their exact
trials). The radar package on the cluster is now installed EDITABLE,
so a `git pull` is sufficient forever after.

## Gates at landing

- cleargbm_rs: full gate green, 100.00% segment coverage (1,680
  all-features tests; floor merging, greedy budget cap, zero-floor
  refusal, Some(1)/Some(0) config refusals, field-25 serde in the
  exhaustive per-field missing/wrong-type harnesses).
- cleargbm: 257 tests green (decode carries/refuses/reads-absent, the
  Rust-boundary key census grew to 26, knob-sensitivity;
  test_types_model.py crossed the 600-line ceiling and split by role —
  the config axes moved to test_types_config_axes.py).
- covenant_ml: 2,542 tests green, 100.00%.
- covenant-radar-api: full gate green at landing (cv_external's new
  argument surface fully tested: threads to every fold, announced in
  the header, refused below 2 and for LightGBM).
