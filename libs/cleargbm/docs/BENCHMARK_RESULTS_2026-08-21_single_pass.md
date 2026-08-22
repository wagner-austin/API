# Benchmark 2026-08-21 — single-pass node histograms over row-major bins (−14%)

The structural lever from agent-board task `fb7206f1`, third and largest of
the day. Manifest:
[`BENCHMARK_MANIFEST_2026-08-21_single_pass.json`](BENCHMARK_MANIFEST_2026-08-21_single_pass.json)
(schema 2, seeds 42/43/44/45). Protocol as the earlier same-day docs;
anchors this run are the tightest of the day (lightgbm ± 0.0035s,
xgboost ± 0.0014s).

## What changed

ClearGBM walked each node's samples once **per feature**: 18 walks per node,
each re-reading the sample index, one bin byte, and both ordered
gradient/hessian streams — roughly 378 bytes of reads per sample per node.
LightGBM's `ConstructHistograms` walks once, updating every feature's
histogram from one contiguous bin row. ClearGBM now does the same:

- `FeatureBins` stores the bin matrix **row-major** (`sample_bins[sample_idx
  * n_features + feat_idx]`), filled rows-outer — the same orientation as
  the input matrix itself. The column-major layout it replaces was itself a
  measured 2026-07-21 improvement over a jagged `Vec<Vec>`; that comparison
  never included flat row-major (see the wiki page
  `cleargbm-perf-column-major-sample-bins` for the history).
- `build_node_histograms_single_pass` builds all features' histograms in one
  walk (~38 bytes of reads per sample), accumulating into one flat
  `n_features x n_bins` block for locality and carving per-feature buffers
  at the end. The per-feature builder, its request type, and the rayon
  per-feature fan-out are deleted; the DI hook moved to the node level.
- The split partition reads the row-major matrix strided; the tree-input
  surface carries one layout, not two.

Bit-identity holds by construction — for every (feature, bin) pair the adds
happen in `sample_indices` order in both shapes — and held in measurement:
every quality metric and leaf count is identical across all seven of the
day's benchmark runs.

## Results

| arm | fit (mean of per-seed medians) | vs pre-lever (0.6991 / 0.684) |
|---|---|---|
| `cleargbm` (depth-wise) | 0.5976s ± 0.0053 | **−14.5%** |
| `cleargbm@leaf_wise` | 0.5798s ± 0.0092 | **−15%** |
| `lightgbm` | 0.4757s ± 0.0035 | anchor |
| `xgboost` | 0.4749s ± 0.0014 | anchor |

Ratios against LightGBM: raw **1.256x** depth-wise / **1.219x** leaf-wise;
per-leaf **0.828x**. The prototype measured 0.6079s with a transpose bolted
onto column-major storage; the full land (native row-major binning, strided
partition, no dual layout) is another ~1.7% under that.

## The day's cumulative ledger

| state | depth-wise fit | raw ratio | per-leaf |
|---|---|---|---|
| morning baseline | 0.8524s | 1.771x | 1.167x |
| + histogram interleave | 0.7115s | 1.493x | 0.984x |
| + scratch reuse / cache move | 0.6991s | 1.458x | 0.960x |
| + single-pass row-major | **0.5976s** | **1.256x** | **0.828x** |

Fit time down 30% in one day, all under the bit-identity gate, plus one
measured rejection (gather prefetch, +2.5%, deleted). The leaf-wise arm now
runs at 1.22x LightGBM's wall clock.

## One measurement thrown out

The first recording of this configuration ran minutes after a maturin
rebuild and produced a LightGBM anchor of 0.5758s ± 0.1644 — thirty times
its usual variance. The run was discarded and repeated on a quiet machine
rather than quoted; its headline ratio (1.047x) would have been flattering
and wrong. Anchors are the tripwire: when the untouched arm moves, the run
is invalid.

## Where the residual 1.26x lives now

Per-leaf ClearGBM is well under LightGBM (0.83x). The remaining wall clock
is the leaf-count policy gap (47 vs 31 depth-wise — the leaf-wise-default
operator decision) plus leaf-wise's concentration of splits on expensive
nodes. On builder throughput this codebase now beats the reference
implementation per unit of work, in safe Rust, at 100% coverage.
