# P6 Landings A + B1 + B2 — the farm, rw_value, and weather_tmax (2026-08-24)

## Landing A — HPC3 as the experiment farm, demonstrated

The charter's P6 half A is live: ClearGBM experiments run on UCI HPC3's
`free` CPU partition through `tools/hpc3`, with the workspace at
`tools/hpc3/runs/hpc3.json` (project `cleargbm`: 4 cpus, 16 GB, 60 min,
`gpu: null`, deterministic, pins declared and enforced by preflight).

The substrate, built this landing: rustup stable (cargo 1.98) at
`/pub/wagnera3/rust`, a clone of the API repo at `/pub/wagnera3/api`
(origin main @ 2e8b8e1), and a conda-forge Python 3.11 env at
`/pub/wagnera3/envs/cleargbm` carrying the exact local benchmark stack
(numpy 2.3.5, lightgbm 4.6.0, xgboost 3.1.2, scikit-learn 1.7.2, optuna
4.6.0, polars 1.36.1, torch-cpu) plus the `cleargbm_rs` wheel built on
the cluster and every monorepo lib installed `--no-deps`. Import and a
smoke training were verified on the login node before any job queued.

### The first rung: 16 members, 320 trials, zero failures

`hpc3-sweep` submitted `runs/sweep-cleargbm-p6-rung1.json`: one member
per (dataset x feature preset) over taiwan/us/polish/kaggle_give_me_credit
and none/log_only/ratios_only/full, each running 20 Optuna TPE trials of
the ClearGBM backend through radar's `scripts.optimize` on a compute
node. All 16 preflighted OK, all submitted, `hpc3-triage` reconciled the
fleet with **0 findings**, and every member's own `.out` log carries its
full report (per-job logs are the authoritative record; see the
concurrency fixes below). Best validation AUC per member:

| dataset | none | log_only | ratios_only | full |
|---|---|---|---|---|
| taiwan | 0.9553 | 0.9553 | **0.9575** | 0.9394 |
| us | 0.8231 | 0.8231 | 0.8416 | **0.8457** |
| polish | 0.9571 | 0.9571 | **0.9593** | 0.9420 |
| kaggle_give_me_credit | 0.8689 | **0.8701** | 0.8688 | 0.8693 |

Trials: 20 complete / 0 pruned / 0 failed on every member. Two honest
readings the rung surfaced:

- **The winning preset tracks dataset width.** `ratios_only` wins on
  taiwan and polish, and the `full` preset (up to ~800 engineered
  features) UNDERPERFORMS there — 95/64 raw features explode into more
  search space than 20 trials can pay for. On us, whose raw width is
  only 18, `full` is the winner (0.8457): feature engineering pays
  exactly where there is room for it.
- **`log_only` ties `none` exactly (four decimals) on three datasets.**
  Verified honored, not decorative: the log arms trained on doubled
  matrices (95 vs 190 features on taiwan). The tie is structural:
  histogram split finding is invariant to per-feature monotone
  transforms, so a log-duplicate of a feature can never change a tree.
  The preset's value is for the non-tree backends.

### Two concurrency hazards found and fixed at the root

Fanning a single-machine CLI across 16 nodes exposed two shared-writer
hazards in `scripts.optimize`'s outputs, both real bugs of the fan-out
rather than of any single run:

1. **Shared history JSONL.** Every member appended one line to a single
   `optimization_history.jsonl`; BeeGFS does not guarantee cross-node
   append atomicity. This run's file validated clean (15/15 lines parse)
   — luck, not a mechanism. Fixed: under the farm the history filename
   embeds `HPC3_JOB_NAME` (which the hpc3 batch script always exports),
   so concurrent members never share a writer; local runs keep the
   accumulating file.
2. **Shared per-dataset result files.** Members for one dataset differ
   only by preset yet all wrote `{dataset}_{backend}_optimal_config.json`
   — last-writer-wins, a survivor whose identity is whichever member
   finished last. Fixed the same way: the filenames embed the job name
   under the farm.

Both fixes read the env through `platform_core.config._test_hooks`
(the codebase's DI law) and are held by tests.

### One test race fixed on the way

The radar streaming suite intermittently failed once the (unrelated,
another session's) `platform-kafka` container answered localhost:9092.
Root cause: three tests asserted `producer.poll(0.0) == 0` — an exact
count of ASYNCHRONOUS rdkafka connection events, green historically only
because nothing answered the port fast enough. Fixed by pinning the
deterministic contract instead: `flush(0.0) == 0` counts queued MESSAGES
(exactly zero when nothing was produced) regardless of connection
events. Three consecutive full radar gates with the live broker up:
2590 passed, three times.

## Landing B1 — rw_value joins the standing regression benchmark

The second regression corpus, derived from rw_matches by
`covenant_ml/scripts/derive_rw_value.py`: **569,561 rows across 99
matches**, target `frames_remaining` (this sample's distance from its
match's last recorded frame — a time-to-verdict value model). The
outcome restatements (`won`, `verdict`) and run identity (`arm`, `seed`,
`difficulty`) never reach the file; `match` is the group column. The
registry, `RegressionDatasetConfig`, `RegressionLoadedDataset` and the
regression CSV loader all gained group support, and the new
registry-driven harness (`covenant_ml.benchmarking.regression_quality` +
`scripts.benchmark_cleargbm_regression`) splits 0.6/0.2/0.2 **by whole
match** when the config declares grouping — 1,500 correlated snapshots
of one match never straddle train and test.

Four arms, matched P1 protocol (300 rounds, lr 0.05, depth 6 / 31
leaves, ES 30), seeds 42-46, mean test RMSE:

| arm | mean test RMSE |
|---|---|
| **cleargbm** | **175,525** |
| cleargbm@leaf_wise | 175,819 |
| lightgbm | 176,091 |
| xgboost | 177,091 |

ClearGBM leads the mean; per-seed wins split 2/2/1 (cleargbm/lightgbm/
xgboost). It is a genuinely hard corpus — R² ~0.2, and seed 44's fold is
negative-R² for every arm, which is what grouped splits over 99 matches
look like when the test matches differ from the training ones. Manifest:
`BENCHMARK_MANIFEST_2026-08-24_p6_rw_value_quality.json`.

## Landing B2 — weather_tmax: the corpus exists, and ClearGBM does not lead it

The charter's weather corpus, built through the DEPLOYED feature path so
training and serving cannot disagree: `fetch_ghcnd_weather.py` vendors 24
GHCN-Daily station files under a MECHANICAL selection rule (inventory
TMAX span <=1950 to >=2024, `USW` prefix, sorted by id, first 24; sha256s
pinned in `raw/MANIFEST.json`), and radar's
`scripts/build_weather_corpus.py` derives the corpus deterministically:
per-station Fourier/threshold state fitted with covenant_ml's McKinnon
machinery on **1950-1989 only** (no evaluation-period data touches the
state), rows from 1990-2024 summers (JJA, the season the thresholds are
defined on), features via the radar `WeatherFeatureExtractor` itself
(anomaly, hot/cold excess, extreme flags) plus day-of-year and three
lagged anomalies, target = the NEXT day's anomaly. Quality-flagged days
are dropped; gaps and season boundaries drop rows rather than impute.

The inventory's span said nothing about continuity — one vendored
station's TMAX record is 1939-1944 with nothing in the fit window at all
— so the builder applies a data-driven completeness gate (>=12,000
fit-window days, >=1,500 row-season days) and REFUSES failures by name:
9 of 24 stations skipped, each printed with its reason. The corpus:
**43,132 rows across 15 stations**, target mean +0.091 C — the 1990-2024
warming signal against the 1950-1989 climatology, visible exactly where
physics puts it.

Four arms, matched P1 protocol, grouped by station, seeds 42-46, mean
test RMSE (degrees C):

| arm | mean test RMSE |
|---|---|
| **lightgbm** | **2.0695** |
| xgboost | 2.0704 |
| cleargbm@leaf_wise | 2.0735 |
| cleargbm | 2.0751 |

**ClearGBM trails on this corpus** — LightGBM is best on four of five
seeds, XGBoost on the fifth, and ClearGBM wins none. The gaps are small
(~0.3% RMSE; R² ~0.44-0.50 on a real day-ahead forecasting signal) and
they are recorded exactly as measured: the standing scoreboard's first
corpus where ClearGBM loses is what makes its wins elsewhere credible,
and closing this gap is now a named target rather than an unknown.
Manifest: `BENCHMARK_MANIFEST_2026-08-24_p6_weather_tmax_quality.json`;
the derived `data.csv` is committed, so the benchmark reproduces without
a network.

## Remaining P6 scope (recorded, not hidden)

- **Metabolomics/BVOC**: real data located (Emily project: 23,134 x 58
  drought/watered/ambient abundance matrix; 10 VOC field sites, ~6.2k
  observations, in corvis `research_*`). Both need an honest supervised
  target designed WITH the operator's science — the leading candidate
  for Emily is blank-vs-real peak classification, per the
  metabolomics-dashboard's own analyses. Deferred to a written design
  rather than invented unilaterally.

## Gates at landing

- covenant_ml: 2510 tests, 100.00% (group support + harness + deriver +
  the weather_tmax registry entry).
- covenant-radar-api: 2608 tests, 100.00% at close (the race fix was
  additionally held to three consecutive green full runs with the live
  broker up, at 2590 tests before the farm-filename and corpus-builder
  tests landed).
- cleargbm / cleargbm_rs: untouched this landing.
- weather_tmax determinism: a full rebuild from the pinned raw files
  reproduces `data.csv` byte-for-byte (one sha256 for both).
