# Benchmark 2026-08-21 — histogram interleave lands (−16%), prefetch measured and rejected

Two pure-perf levers from agent-board task `4faf01ac`, both run under the
bit-identity gate (architecture.md § "Two gate types") with the
branch → measure → collapse-to-winner procedure. One won by a lot; one lost
and was deleted the same hour. Manifest for the landed state:
[`BENCHMARK_MANIFEST_2026-08-21_interleave.json`](BENCHMARK_MANIFEST_2026-08-21_interleave.json)
(schema 2, seeds 42/43/44/45 — four seeds so the four-arm slot rotation
covers every slot per arm).

All runs: American bankruptcy 78,682×18, 200 trees, `max_depth` 6, 64 bins,
`num_leaves` 31, `n_jobs` 1, 2 warm-ups + 5 timed fits per arm per seed,
EcoQoS power-throttle opt-out active. Same machine, same session, LightGBM
anchor stable across all three runs (0.4812 / 0.4808 / 0.4766s) — the
paired comparisons below are clean.

## Lever: histogram interleave — LANDED

`HistogramBuffer` stored three parallel arrays (`gradient_sums`,
`hessian_sums`, `counts`); the hot loop's per-sample read-modify-write
touched three cache lines and paid three bounds checks. The accumulators are
now interleaved into one `Vec<BinAccumulator>` (24-byte records), so each
update touches one contiguous record and pays a single bounds check. This is
LightGBM's `hist_t` grad/hess interleaving, extended to carry the count.

| arm | before (baseline run) | after (this manifest) | change |
|---|---|---|---|
| `cleargbm` (depth-wise) | 0.8524s ± 0.0041 | 0.7115s ± 0.0068 | **−16.5%** |
| `cleargbm@leaf_wise` | 0.7999s ± 0.0101 | 0.6672s ± 0.0092 | **−16.6%** |
| `lightgbm` | 0.4812s ± 0.0047 | 0.4766s ± 0.0049 | −1.0% (anchor) |
| `xgboost` | 0.4801s ± 0.0106 | 0.4793s ± 0.0040 | −0.2% (anchor) |

Ratios against LightGBM: raw 1.771x → **1.493x**; per-leaf 1.167x →
**0.984x**. ClearGBM now builds a leaf *cheaper* than LightGBM builds one on
this workload — the remaining wall-clock gap is entirely the leaf-count gap
(47.0 vs 31.0 depth-wise), which is a policy choice, not a builder cost.
The leaf-wise arm at 0.6672s is 1.40x LightGBM raw.

Bit-identity evidence: every quality metric and leaf count in this manifest
reproduces the baseline run exactly (AUC-ROC 0.6967 / 0.7011 / 0.7001 /
0.6969, leaves 46.99 / 30.98 / 30.96 / 22.39) — the layout change moved no
arithmetic. The full cleargbm_rs suite (1,251 tests, including the exact
per-bin sum assertions in `histogram/tests/`) passes unchanged, and the
coverage gate holds at 100.00%.

A structural consequence, handled rather than suppressed: with the three
arrays fused into one record, a per-field length disagreement between them
can no longer be constructed, so the split scan's per-field error arms
became dead code. The scan now reads each bin's record directly under the
`bins.len() == n_bins` construction invariant, and the serde boundary
rejects mismatched wire arrays (`invalid_length`) — corruption is stopped
where it enters instead of re-checked at every read.

## Lever: gather prefetch — MEASURED, REJECTED, DELETED

The reorder gather (`reorder_grad_hess_into`) does two random reads per
sample over the full gradient/hessian arrays. The hypothesis: sparse node
walks defeat the hardware prefetcher, so a software touch-ahead helps. This
crate forbids unsafe code, so the lever was spelled as `black_box` forced
reads 16 iterations ahead (independent per-value calls; no dependency
chain), the safe equivalent of `_mm_prefetch`.

Measured on top of the interleave, same protocol: cleargbm 0.7148s →
0.7328s (+2.5%), per-leaf 0.980x → 1.007x. The mechanism in hindsight: the
sample indices are sorted ascending, so the walk is a monotone sweep the
hardware prefetcher already covers; the warming loads are pure overhead.
The lever is deleted per the collapse-to-winner procedure; the negative
result and the reasoning live in a comment at the loop so the idea is not
re-tried blind.

## Cross-run honesty

Fit times moved between the 2026-08-21 morning four-arm run (cleargbm
0.8888s) and this session's baseline (0.8524s) with no code change —
machine state. Every comparison in this document is within-session with a
stable LightGBM anchor; treat the ratios, not the absolute times, as the
durable numbers.
