---
title: Covenant Radar's Optuna search — two entry points, one recorded, and a pin the record denies
tags: [services, covenant-radar, optuna, hyperparameter-search, provenance, run-record, determinism]
related:
  - "[[covenant-radar-service-architecture]]"
  - "[[covenant-radar-backend-registry]]"
  - "[[model-trainer-run-record-provenance]]"
  - "[[determinism-env-read-once-at-library-load]]"
  - "[[cleargbm-hpc3-farm-and-rw-value]]"
source_paths:
  - services/covenant-radar-api/src/covenant_radar_api/worker/optimize_job.py
  - services/covenant-radar-api/src/covenant_radar_api/worker/_optimize_common.py
  - services/covenant-radar-api/src/covenant_radar_api/api/routes/ml_analysis.py
  - services/covenant-radar-api/scripts/optimize/__main__.py
  - services/covenant-radar-api/scripts/optimize/__init__.py
  - services/covenant-radar-api/scripts/optimize/_runners.py
  - services/covenant-radar-api/scripts/optimize/run_records.py
  - services/covenant-radar-api/scripts/optimize/history.py
  - libs/platform_core/src/platform_core/determinism_record.py
  - libs/covenant_ml/src/covenant_ml/optimizer/search_spaces/config.py
  - docs/RESEARCH.md
source_git_blobs:
  "services/covenant-radar-api/src/covenant_radar_api/worker/optimize_job.py": f301bd2b9e23ec95f39fde30497a7d281ad48cdf
  "services/covenant-radar-api/src/covenant_radar_api/worker/_optimize_common.py": 68c3a7eaa7a8a72b0249cfa0764e82f86d96b933
  "services/covenant-radar-api/src/covenant_radar_api/api/routes/ml_analysis.py": 8279fa6db5c84444f8d73639e0cd0fdde83f74d5
  "services/covenant-radar-api/scripts/optimize/__main__.py": b5246e4c973122144b2399ff39f42505d3eff3ff
  "services/covenant-radar-api/scripts/optimize/__init__.py": 70a2307d539af98af8ef4f4b668aeb1d138a3dc1
  "services/covenant-radar-api/scripts/optimize/_runners.py": 600ae002defa64f7fe500dbf3295b5c8c1f6b407
  "services/covenant-radar-api/scripts/optimize/run_records.py": b20420c11b25dbe339193541a362141b2b0bc282
  "services/covenant-radar-api/scripts/optimize/history.py": f7eefd5f3fa9bc64b0c8e673e37a203ab06bf0f6
  "libs/platform_core/src/platform_core/determinism_record.py": 1b098cb9098ce0db9079a87f3e3e9b39dd00f250
  "libs/covenant_ml/src/covenant_ml/optimizer/search_spaces/config.py": 7aef2d032b17daaa8b98c00cf5eb4487b2c7373d
  "docs/RESEARCH.md": 501940c688722b74b5b1a49b1825b2a22e2322e4
provenance:
  - "docs/RESEARCH.md repinned 2026-10-07 from fa186d5d to 501940c6 on a mechanical argument rather than a re-reading: the diff is +17/-0, one paragraph appended to the turkic-lstm section by c0c80420b (board task bc12f18b); the cleargbm section and its Provenance bullet that this page cites are byte-identical. Check with: git diff fa186d5df657233450dfac204982ad9eff07173d 501940c688722b74b5b1a49b1825b2a22e2322e4"
  - "measured 2026-10-05 on austinpc: ~/PROJECTS/API/services/covenant-radar-api/models/optimization_history.jsonl is 3,068 lines, all 3,068 carry \"fingerprint\":null, last modified 2026-08-28 02:25; no *.runrecords.jsonl file exists beside it. The file is machine-local and untracked, so this reading is not reproducible from the repository."
  - "dates: git log -S'UNPINNED because this entry point pins nothing' -- scripts/optimize/_runners.py -> a92a31ace, 2026-08-28; git log -1 -- scripts/optimize/__main__.py -> 460c20382, 2026-08-29 (the pin)"
fact_checked: "2026-10-05"
confidence: high
hubs: [services]
---

# Covenant Radar's Optuna search

Hyperparameter search reaches one optimiser by two doors, and only one of them
leaves a record anyone can compare.[^route][^runners]

## The shared core

Both doors end in `covenant_ml`'s strategy registry,
`optimizer_registry_factory().get(OptimizerStrategyName.OPTUNA_TPE)`, called
with the backend's own `get_default_search_space()` and an objective built by
`objective_factory` — no per-backend branching, which is how one job replaced
five.[^job] The config is `make_default_optimization_config`: a seeded split
of 70/15/15 train/validation/test, `random_state` default 42.[^split] Which
backends can be searched is [[covenant-radar-backend-registry]].

## Door 1: `POST /ml/optimize` (RQ)

The route enqueues `worker.optimize_job.process_optimize_job`;[^route] the job
writes `{dataset}_{backend}_optuna_result.json` and `_optimal_config.json`,
suffixed with `HPC3_JOB_NAME` when one is exported so concurrent sweep members
do not overwrite each other last-writer-wins.[^save] **It writes no
history row, no fingerprint and no `RunRecord`** — nothing under `src/`
references either.[^job] A number from this door says what the search found
and nothing about what it ran on.

## Door 2: `python -m scripts.optimize` (CLI)

This is the door `docs/RESEARCH.md` registers for the `cleargbm` project.[^research]
Each completed run appends one row to `optimization_history.jsonl` and one
`RunRecord` to the sibling `*.runrecords.jsonl`:[^runners]

- the history row holds the search's shape (backend, dataset, preset, trials,
  sample/feature counts, `best_val_auc`, duration) and a three-state
  `fingerprint` — present, explicit `null` for rows predating capture, and a
  missing key refused on decode;[^history]
- the record, experiment `covenant-radar-hyperparameter-search`, carries
  `best_val_auc`, `duration_seconds` and `trials_completed` as observations,
  with the backend in the label so two backends' records pair rather than
  being held apart;[^records]
- a `null`-fingerprint row is never turned into a record: `optimization_run_record`
  raises rather than synthesise one.[^records]

On disk, **all 3,068 history rows are still `null` and no records file
exists** — the capture landed 2026-08-28 and the CLI has not completed a run
on this machine since.[^disk]

## The pin the record denies

`__main__.run` calls `apply_cpu_determinism(os.putenv, SINGLE_THREAD, …)`
before the first numeric import, refusing outright if numpy is already loaded,
and `__init__.py` holds no re-exports precisely so that ordering is reachable.
That landed in `460c20382` on 2026-08-29.[^main]

**But the `DeterminismRecord` the pin returns is discarded** — `run` calls
`pin_cpu()` and ignores the result — and `_runners.py` still builds every
fingerprint from `determinism_record(UNPINNED_STACK, {})`, i.e. stack `"none"`,
under a comment written the day before the pin ("this entry point pins
nothing").[^runners][^unpinned] So the first fingerprinted run will record a
single-threaded process as unpinned, and `compare_configurations` will treat
it as differing on the `determinism` axis from every honestly-pinned
`covenant_ml` benchmark. `docs/RESEARCH.md`'s `cleargbm` entry says "the
fingerprint a run writes now names the pinned stack"; the code says
otherwise.[^research] `run_records.py`'s module docstring carries the same
stale "still pins nothing".[^records]

The repair is to thread the record from `run` into `_run_backend_with_progress`
instead of constructing one; nothing has to be invented.[^main][^runners]

[^job]: `services/covenant-radar-api/src/covenant_radar_api/worker/optimize_job.py` / `run_optimization` (module docstring: "Replaces 5 per-backend optimize job files").
[^split]: `libs/covenant_ml/src/covenant_ml/optimizer/search_spaces/config.py` / `make_default_optimization_config` — `random_state: int = 42`, `"val_ratio": 0.15`, `"test_ratio": 0.15`; called by `services/covenant-radar-api/src/covenant_radar_api/worker/_optimize_common.py` / `build_optimization_config`.
[^route]: `services/covenant-radar-api/src/covenant_radar_api/api/routes/ml_analysis.py` / `_register_optimize`.
[^save]: `services/covenant-radar-api/src/covenant_radar_api/worker/_optimize_common.py` / `save_optimization_results`.
[^research]: `docs/RESEARCH.md` § "`cleargbm` — ClearGBM benchmarks and covenant-radar optimisation", the `Provenance:` bullet beginning "`scripts/optimize` HAS pinned since 2026-08-29".
[^runners]: `services/covenant-radar-api/scripts/optimize/_runners.py:156` — `benchmark_fingerprint(determinism_record(UNPINNED_STACK, {}), config_env.get_env)`, then `history.append(entry)` and `append_optimization_record(history.path, entry)`.
[^history]: `services/covenant-radar-api/scripts/optimize/history.py` / `UnifiedHistoryEntry`, `_require_fingerprint_or_null`.
[^records]: `services/covenant-radar-api/scripts/optimize/run_records.py` / `OPTIMIZATION_EXPERIMENT`, `optimization_observations`, `optimization_run_record`, and the module docstring's "WHAT THIS DOES NOT FIX" paragraph.
[^main]: `services/covenant-radar-api/scripts/optimize/__main__.py` / `pin`, `run` (`pin_cpu()` on its own line, return value unused); `services/covenant-radar-api/scripts/optimize/__init__.py` (module docstring, "DELIBERATELY EMPTY OF RE-EXPORTS").
[^disk]: `services/covenant-radar-api/scripts/optimize/history.py` / `OptimizationHistory.for_output_dir` names the file; `services/covenant-radar-api/scripts/optimize/run_records.py` / `optimization_record_path` names its sibling [synthesis] — the line and null counts are the 2026-10-05 measurement in this page's provenance.
[^unpinned]: `libs/platform_core/src/platform_core/determinism_record.py` / `UNPINNED_STACK` (`"none"`).
