# P6 Landings A + B1-B4 — the farm, rw_value, weather_tmax, metab_confidence, and voc_match_quality (2026-08-24)

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

## Landing B3 — metab_confidence: the metabolomics corpus, from the artcal campaign

The metabolomics corpus comes from the operator's own artcal campaign
(corvis dataset 15's provenance chain: Waters MS-E raws -> ProteoWizard
-> mzMine 4.9.14 -> DIA pseudo-MS2 -> SIRIUS 6.3.12), NOT the separate
Emily/ProGenesis table — the provenance record says "Do not join them"
and this corpus honors that by living entirely on the mzMine/SIRIUS
side. The SIRIUS merged summaries ARE the target source; ZODIAC was
deliberately not run in that campaign (the `ZodiacScore` column is
empty), so there is no ZODIAC data to include.

The question the corpus asks is deployable: given only what the
instrument measured, how confident will CSI:FingerID be in its best
structure call? `covenant_ml/scripts/build_metab_corpus.py` joins three
sha256-pinned sources — the 372 MB DIA export MGF (22,906 features,
MS1+MS2 block pairs), the MetaboAnalyst quant table (114,814 aligned
features x 24 samples), and SIRIUS `structure_identifications.tsv`
(18,406 rank-1 structure calls) — into one row per rank-1 feature.
Target: COSMIC `ConfidenceScoreExact`, verbatim. Predictors:
pre-annotation measurables ONLY (precursor m/z, RT, log MS1 height,
correlated-MS1 peak count, MS2 peak count and log total/max intensity,
top-3 intensity fraction, and detection count / log mean / log max
across the 23 BIOLOGICAL samples — the pooled `combine.mzML` injection
is excluded). Annotation outputs (adduct, formula, ionMass, any SIRIUS
score) never become features: they are downstream of the answer.

Drops are counted, never imputed: 795 features whose confidence is
SIRIUS's `-Infinity` ("no exact confidence computable") are dropped by
rule; zero features were undetected in all biological samples. The
corpus: **17,611 rows across 138 retention-time bins** (0.1-minute
co-elution windows — adducts and in-source fragments of one molecule
co-elute, so `rt_bin` is the GROUP column and whole windows land in one
split). Target mean 0.1198. The rebuild is byte-deterministic (one
sha256 twice).

Four arms, matched P1 protocol, grouped by rt window, seeds 42-46,
mean test RMSE:

| arm | mean test RMSE |
|---|---|
| **xgboost** | **0.129348** |
| lightgbm | 0.129391 |
| cleargbm@leaf_wise | 0.129428 |
| cleargbm | 0.129497 |

Per-seed wins: xgboost 3, lightgbm 1, cleargbm@leaf_wise 1. Two honest
readings:

- **The arms are effectively tied.** The full spread is 0.12% of RMSE,
  far smaller than the seed-to-seed spread (0.1261-0.1342); no arm
  separates from the pack on this corpus.
- **The corpus's real finding is scientific: structure confidence is
  barely predictable from pre-annotation measurables.** R² is
  0.004-0.025 for every arm — spectrum quality, intensity and detection
  breadth explain ~2% of COSMIC confidence variance under grouped
  splits. That is worth knowing (it says confidence is dominated by
  what the compound IS, not how well it was measured), and it makes
  this a weak-signal benchmark entry: differences between learners here
  are noise until an arm moves R² materially.

Manifest: `BENCHMARK_MANIFEST_2026-08-24_p6_metab_confidence_quality.json`;
the derived `data.csv` (1.5 MB) is committed, so the benchmark
reproduces without the 372 MB sources.

## Landing B4 — voc_match_quality: the BVOC corpus, from the field-site peak table

The VOC corpus comes from the Faiola Lab's aggregated GC-MS peak table
(`Aggregated_Summarized_Output.xlsx`, tree-bot's sha256-pinned
2026-07-25 lab snapshot — the builder's manifest pin reproduces the
snapshot's own pin exactly): ten California reserve sites, one sheet
each, one row per chromatogram peak with its top-3 NIST library matches.
The corpus asks the GC-MS twin of metab_confidence's question: given
only what the instrument measured, how well will the NIST library
identify this peak?

`covenant_ml/scripts/build_voc_corpus.py` reads the workbook through a
new stdlib XLSX reader (`covenant_ml.datasets.xlsx_reader` — zipfile +
XML, verbatim cell strings, no third-party dependency, every cell shape
and refusal tested) and emits one row per peak. Target:
`Match1.Quality` (1-99), verbatim. Predictors: pre-annotation
measurables ONLY — the plant species (known at sampling; 41 codes
including `blk` blank cartridges), the retention time, and
chromatogram-context statistics computed from peak positions alone
(run peak count, RT rank fraction, gap to the previous peak, co-elution
counts within ±0.1/±1 min, run RT span; a peak with no library hit
still crowds its neighbours). Library outputs (match names, Match2/3
qualities), curation outputs (Compound, Class, Comments), `MatchScore`
(verified equal to `Match1.Quality` on all 6,238 rows) and run identity
never become features. `site` is the GROUP column: peaks from one
reserve share plants, weather and instrument sessions.

Drops counted, never imputed: 1 peak with no retention time, 350 rows
of the provenance-flagged misfiled cartridges (no species), 35 with no
match quality, and 2 whose recorded quality (944, 994) is outside
NIST's 1-99 scale — data defects dropped, not guessed at. The corpus:
**5,850 rows across 10 sites (228 chromatograms)**, target mean 75.72,
byte-deterministic rebuild (one sha256 twice).

Four arms, matched P1 protocol, grouped by site, seeds 42-46, mean
test RMSE (quality points):

| arm | mean test RMSE |
|---|---|
| **lightgbm** | **18.3170** |
| cleargbm | 18.4423 |
| cleargbm@leaf_wise | 18.4806 |
| xgboost | 18.5646 |

Per-seed wins: lightgbm 3, cleargbm 1, xgboost 1. Unlike
metab_confidence this corpus has REAL signal — R² 0.37-0.48 for every
arm: where a peak elutes, what plant it came from, and how crowded its
neighbourhood is explain nearly half the variance in library match
quality. The entry is honestly adverse: LightGBM leads it, ClearGBM is
second at a 0.7% gap — the second corpus on the named-target list
beside weather_tmax. Manifest:
`BENCHMARK_MANIFEST_2026-08-24_p6_voc_match_quality_quality.json`; the
derived `data.csv` (416 KB) is committed.

## Landing B5 — metab_blank: blank-vs-real peaks from the Emily table

The last named corpus (2026-08-25): the Emily/ProGenesis table
(23,134 x 58; never joined to metab_confidence, per provenance rule)
as a blank-vs-real peak classifier — the metabolomics-dashboard's own
leading design. The lab's standard blank-filter rule IS the label: a
peak is real when its biological-sample average is at least 3x its
individual-blank average, or appears in samples only (the pooled
combine blank excluded; the dashboard's open leaf-vs-root blank
assignment dispute is sidestepped by its documented all-blanks option).
`covenant_ml/scripts/build_metab_blank_corpus.py` computes the label
from the intensity columns and emits NOTHING intensity-derived as a
feature — the predictors are physicochemical measurables only: m/z,
charge, retention time, chromatographic peak width, and the m/z and
Kendrick (CH2) mass defects that contaminant homolog series like
plasticizers and PEG are known to carry. **19,064 rows across 127
retention windows** (positive ratio 0.641; 4,070 nowhere-detected rows
dropped by counted rule; byte-deterministic; the sheet's own average
columns cross-check the computed rule exactly).

The corpus also forced one honest instrument extension: co-elution
windows hold real and blank peaks TOGETHER, so the stratified group
splitter's any-positive group label is undefined (it refused, with
zero negative groups — correctly). covenant_ml gained
`group_kfold_split` (plain grouped k-fold), and `cv_external` selects
the instrument by a stated, data-driven property: label-uniform groups
run the stratified protocol exactly as before (the rw_matches anchor
re-reproduces 0.7295 ± 0.0728 fold-for-fold), mixed-label groups run
plain grouped k-fold and SAY SO in the output.

Grouped 5-fold CV, the flagship protocol, mean held-out AUC:

| arm | mean AUC |
|---|---|
| lightgbm | 0.8710 ± 0.0136 |
| cleargbm | 0.8691 ± 0.0120 |
| cleargbm-leafwise | 0.8690 ± 0.0120 |

A statistical three-way tie (the 0.002 spread sits inside the ±0.012
fold spread) — and the finding is scientific: **contamination is
physicochemically distinguishable at AUC ~0.87 with no intensity
information at all.** Mass defects, elution position and peak shape
carry most of what the blank injections measure, which means a
blank-filter prior exists even for runs where blanks are missing,
disputed, or contaminated — exactly the situation the dashboard's
blank-assignment findings describe.

## Remaining P6 scope (recorded, not hidden)

- Larger-budget farm rungs (100+ trials with the coarseness dial in
  the space) whenever the standing numbers warrant another push.

## Gates at landing

- covenant_ml: 2554 tests, 100.00% (group support + harness + deriver +
  the weather_tmax registry entry; Landing B3 adds the metab_confidence
  builder and registry entry; Landing B4 adds the voc builder, the
  stdlib XLSX reader, and the registry split — registry.py crossed the
  600-line ceiling and its verified config tuples moved to
  registry_configs.py).
- covenant-radar-api: 2608 tests, 100.00% at close (the race fix was
  additionally held to three consecutive green full runs with the live
  broker up, at 2590 tests before the farm-filename and corpus-builder
  tests landed).
- cleargbm / cleargbm_rs: untouched this landing.
- weather_tmax determinism: a full rebuild from the pinned raw files
  reproduces `data.csv` byte-for-byte (one sha256 for both).
- metab_confidence determinism: two builds from the pinned sources
  produce identical `data.csv` and `MANIFEST.json`; the MGF's sha256
  matches the corvis provenance pin exactly.
- voc_match_quality determinism: two builds from the pinned workbook
  produce identical `data.csv` and `MANIFEST.json`; the workbook's
  sha256 matches tree-bot's lab-snapshot pin exactly.
