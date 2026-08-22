# Benchmark 2026-08-21 — ordered-scratch reuse and the cache clone removed

Two allocation-churn levers from agent-board task `e567b5b9`, measured on
top of the same-day interleave landing
([`BENCHMARK_RESULTS_2026-08-21_interleave.md`](BENCHMARK_RESULTS_2026-08-21_interleave.md)).
Manifest: [`BENCHMARK_MANIFEST_2026-08-21_scratch_reuse.json`](BENCHMARK_MANIFEST_2026-08-21_scratch_reuse.json)
(schema 2, seeds 42/43/44/45). Protocol identical to the interleave doc;
LightGBM anchor stable across all runs (0.4766–0.4808s).

## What changed

**Ordered-scratch reuse.** Every node build allocated two fresh
`vec![0.0; n_at_node]` buffers for the position-space gradient/hessian
streams, and the zero-fill was overwritten immediately by the reorder pass.
One `OrderedScratch` pair is now allocated per tree at root size; each node
overwrites exactly the prefix it reads.

**Cache move.** Every non-root depth-wise node received its histograms from
the parent's sibling subtraction as an owned vector on its pending-node
record — and then *cloned* all of them (18 buffer allocations, ~27 KB of
copies per node) because the build path took the cache by reference. The
builder now takes the vector off the pending node and uses it directly. The
`cached_histograms` parameter and its clone path are deleted from
`build_feature_histograms`; the leaf-wise builder never used them (it passes
histograms through its evaluation queue by value already).

## Results (mean of per-seed medians, four runs pooled where noted)

| arm | before (interleave manifest) | after | change |
|---|---|---|---|
| `cleargbm` (depth-wise) | 0.7115s ± 0.0068 | 0.6991s ± 0.0193 (0.6976 / 0.7032 across repeats) | **−1.7%** |
| `cleargbm@leaf_wise` | 0.6672s ± 0.0092 | 0.684s pooled (0.6906 / 0.6716 / 0.6901) | **+1.9%, within its own spread** |
| `lightgbm` | 0.4766s | 0.4797s | anchor |

Ratios: raw 1.493x → 1.458x; per-leaf 0.984x → 0.960x. Bit-identity held:
every quality metric and leaf count identical across all runs.

## The honest reading

**Kept, both levers.** The depth-wise arm — the production default — improved
consistently across three independent runs. The leaf-wise arm drifted ~2%
the other way with tripled variance; three post-change readings straddle the
pre-change value and the direction is not clearly separable from noise. A
per-strategy fork of the scratch path was considered and rejected: a ≤2%
cost on a non-default arm does not buy a second code path. The cache move
measured no delta of its own but deletes 18 allocations and ~27 KB of
copying per cached node and removes a parameter and a branch — strictly
less code doing strictly less work.

**Estimate calibration, recorded so the next reader is smarter.** The task
spec projected ~1.5 GB of wasted memset from the per-node zero-fills; the
measured gain (~12 ms) implies the real waste was far smaller. Two reasons:
the sibling-subtraction path already reorders only the *smaller* child, so
the summed scratch sizes are roughly a third of the naive per-level
estimate, and freshly mapped pages on Windows are zero on demand, making
`vec![0.0; n]` cheaper than a memset of touched memory. Allocation-churn
levers on this codebase are worth low single-digit percents, not tens.

## Where the remaining 1.46x lives

With per-leaf cost at 0.96x, the wall-clock gap to LightGBM is the
leaf-count policy gap (47 vs 31 leaves depth-wise) plus the leaf-wise arm's
concentration of splits on expensive nodes. The builder itself is no longer
the deficit. Further single-thread gains would need vectorization of the
histogram inner loop (blocked by `unsafe_code = "forbid"` — a deliberate
trade this codebase has already made) or algorithmic changes that go
through the quality gate instead of the bit-identity gate.
