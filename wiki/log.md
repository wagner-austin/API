# Wiki Operation Log

Append-only. Log structural operations (new hubs, decomposition, audits, cleanups). Routine page edits don't need a log entry — git history covers those.

## [2026-07-06] init | api monorepo wiki scaffolded
Hubs created: services, clients, libs, infrastructure
Notes: initial scaffold via /wiki-init. Empty pages/ — content added as subsystems get documented.

## [2026-07-07] first-batch | 3 starter pages
Pages written: monorepo-discipline, platform-workers-rq-pattern, service-port-map
Hubs updated: services (+1), libs (+1), infrastructure (+1)
Notes: all pages audited claim-by-claim against the code before landing — Redis client factory names, RQ harness helpers, monorepo-guards.toml location, service port assignments all verified in the source. Skipped writing more api pages this batch because deeper subsystem context (Kafka streaming in covenant-radar-api, Kohya-ss backend in Art-Trainer, MMS-LID in platform_langid) would require reading service internals; those pages should be written by someone who's touched the subsystem code, not paraphrased from READMEs.

## [2026-07-20] audit | all pages verified against current code
Pages audited: monorepo-discipline, platform-workers-rq-pattern, service-port-map (plus all 4 hubs + index)
Pages updated: index.md, hubs/services.md, hubs/libs.md, hubs/clients.md, pages/service-port-map.md, pages/platform-workers-rq-pattern.md, pages/monorepo-discipline.md (fact_checked bump only)

Findings and fixes (all applied):
1. **`doc-extract-api` service was missing from the wiki entirely.** Real service under `services/doc-extract-api/` with its own Dockerfile, poetry env, README, and layered `docker-compose.yml`. Uses port 8012 (host) → 8000 (container) per `services/doc-extract-api/docker-compose.yml:24`; runs `hypercorn` via Dockerfile CMD; depends on `psycopg` (postgres) and `platform_workers`. Added to: `index.md` services enumeration; `hubs/services.md` services list; `pages/service-port-map.md` port table with an inline note that the assignment is authoritative in the service's own compose file, not the root README. Root cause: root `README.md` Services table still doesn't list this service — the wiki inherited the omission because its citation chain terminates at the root README.
2. **"8012 is free" claim in service-port-map was wrong.** Corrected to "8013 is free". Also expanded step 3 of "Adding a new service" to describe the layered-compose pattern that doc-extract-api uses.
3. **`platform_devpost` lib was missing from `index.md` libs enumeration.** Added. (Already present in `hubs/libs.md`; added a one-line description parenthetical there to match the pattern of surrounding libs.)
4. **`qr-api` was cited as the canonical "doesn't need `platform_workers`" example — actually a heavy consumer.** `services/qr-api/src/qr_api/*` imports `redis_for_kv`, `run_rq_worker`, `WorkerConfig`, `readyz_redis_with_workers`, `RedisStrProto` across six modules. Corrected example to `github-stats-api` / `grandma-api` / `opportunity-radar-api` (grep-verified they have zero `platform_workers` or `rq` imports). Added an explicit "services that DO consume it today" list so the boundary is unambiguous.
5. **`readyz_redis` response shape was mis-described.** Wiki claimed "returns `{status: ready}` on success, 503 with a reason on failure". Actual: returns a `ReadyResponse` TypedDict `{"status": "ready"|"degraded", "reason": None|str}`; the consuming route maps `degraded` → HTTP 503. Fixed with the exact success/degraded shapes verified in `libs/platform_workers/src/platform_workers/health.py`.
6. **`platform_workers` RQ surface was under-documented.** The page named only `rq_queue` + `rq_retry` and called them "two thin helpers", but every service with a worker actually goes through `run_rq_worker(config: WorkerConfig)` as the entry point (see qr-api's `worker_entry.py`, etc.). Added a paragraph naming `run_rq_worker`, `WorkerConfig`, `get_current_job`, `rq_fetch_job` — verified as `__all__` / top-level `def`s in `rq_harness.py`. Added `readyz_redis_with_workers` alongside `readyz_redis` with its actual `SCARD`-on-workers-set behaviour.
7. **`hubs/clients.md` looked empty ("0 pages") — but the client `TankpitBot` maintains its own full three-tier wiki at `clients/TankpitBot/wiki/`.** Not a wiki bug per se, but a navigation gap: a new AI reading `api/wiki` would never find 50+ pages of TankpitBot knowledge. Added a section pointing at that wiki as the source of truth for tankpit facts, and defined what this hub SHOULD hold (monorepo-integration surfaces — none yet written).

Verified (no changes needed):
- `platform_workers` typed Redis factories — `redis_for_kv`, `redis_for_rq`, `redis_raw_for_rq`, `redis_for_pubsub` — all present at `libs/platform_workers/src/platform_workers/redis.py:{282,315,324,336}`.
- `rq_queue` + `rq_retry` — present at `rq_harness.py:{185,94}`.
- `monorepo_guards` rule count — the wiki says "20+"; actual is 25 `*_rules.py` files in `libs/monorepo_guards/src/monorepo_guards/`, each containing multiple rules. "20+" is correct and intentionally conservative.
- `monorepo-guards.toml` location — at repo root, path referenced in wiki resolves correctly.
- Every port `8000-8011` matched between README table, root `docker-compose.yml` port-map comment, and the wiki table.
- covenant-radar-api Kafka claim — `confluent-kafka = "^2.12"` in `services/covenant-radar-api/pyproject.toml:33`.
- PostgreSQL claim — root `docker-compose.yml` runs `postgres:16-alpine`; consumed by covenant-radar-api, doc-extract-api, and `covenant_persistence` lib.

Root causes recorded for the misses: (a) the wiki cites the root `README.md` as its primary for service enumeration, which itself omits doc-extract-api — future audits should cross-check `ls services/` against the README, not trust the README alone; (b) the qr-api "doesn't need workers" example was a guess based on the service name, not a grep of its imports.

`fact_checked` on all three content pages bumped to 2026-07-20.

## [2026-07-20] add | ClearGBM histogram split path page
Pages written: cleargbm-histogram-split-path
Hubs updated: libs (+1, 1 -> 2 pages)
Index updated: page count 3 -> 4; libs hub count 1 -> 2
Notes: written alongside the ClearGBM validation + reg_lambda fix session (libs/cleargbm/docs/VALIDATION_REPORT_2026-07-20.md). Page captures the durable knowledge — the two split paths (histogram at runtime, exact retained for tests), the O(K) claim with empirical verification (10k -> 1M samples: 0.92x ratio at K=64), sibling subtraction verified to 1e-15 per bin, the reg_lambda semantic fix that landed in this session's histogram.py + parallel.py, and the Rust vs Python fallback hook indirection. Every claim cites a code path with line-range or function locator. `confidence: high` on the pure math + verified empirical claims; the reg_alpha "not applied to split gain in either path" note is a design-boundary claim, not a bug claim.

## [2026-07-24] add + correct | leaf-normalized benchmarking; retired the Phase-I gap claim
Pages written: cleargbm-leaf-normalized-benchmarking
Pages updated: cleargbm-perf-experiments-2026-07-21 (Phase-I final benchmark marked SUPERSEDED; cnt_factor/counts-drop marked CLOSED)
Hubs updated: libs (+1 = 8 pages)
Index updated: total content pages 9 -> 10; libs hub count 7 -> 8

Notes: three corrections, each traced to a root cause rather than a restated number.

1. **The Phase-I gap (1.10x) does not reproduce and is retired.** Two independent causes. (a) The phase-I harness lived only in a session scratchpad (`scripts/benchmark_vs_lightgbm.py`, never committed) and was lost; every claim made after it disappeared was measured on the noisier phase-E shape, which also hardcoded "LightGBM numbers carry forward from an older manifest". Dividing a fresh cleargbm time by a stale LightGBM time is what produced a reported "1.40x gap" on 2026-07-24. (b) Its canonical statistic was the MIN of 5 repeats. Min is right about slow-side noise but wrong about the fast side: the first fits after idle run with full turbo headroom, a different power regime. Observed LightGBM seed 42 `min/med/mean/max = 0.486/0.828/0.740/0.874` -- one cold-start outlier drove a 1.751x +/- 0.538 gap where the median over the same samples gave 1.43x +/- 0.02. Replacement harness uses median, 2 discarded warmups, and alternates model order across seeds.

2. **counts-drop / cnt_factor is CLOSED, not open.** It was re-proposed as an interleaved-histogram "Phase 4". Rejected on two independent lines: it is already twice-measured (-5% and -16%, the latter with one-sided quality drift), and a max_bins sweep holding allocation count fixed shows cleargbm FLAT from 16->64 bins -- so at the benchmark's max_bins=64 the workload is not bound by histogram bytes, which is exactly what the change relieves. Also noted: cleargbm has no `min_sum_hessian_in_leaf`, so `counts` is the only leaf-size regularizer; LightGBM pairs derived counts with an exact hessian constraint, so adopting the approximation without that backstop is strictly worse than LightGBM's design.

3. **The headline "cleargbm is behind LightGBM" was mostly a measurement artifact.** cleargbm grows depth-wise, LightGBM leaf-wise with num_leaves=31. Measured leaf counts per tree: depth 3/4/5 match within 2%, but at the benchmark's depth 6 cleargbm builds 57.9 leaves vs LightGBM's 31.0 (1.87x). Authoritative 2026-07-24 run: cleargbm 1.2809s +/- 0.0638 vs LightGBM 0.8981s +/- 0.0719 -- raw 1.426x, leaf ratio 1.523x, **per-leaf 0.937x**, i.e. cleargbm is ~6% FASTER at equal tree size, with quality a statistical tie (cleargbm ahead on AUC-PR).

Harness now lives at `libs/covenant_ml/src/covenant_ml/benchmarking/` (layered, DI via Protocols, 100% statement+branch coverage) with entry point `libs/covenant_ml/scripts/benchmark_cleargbm_vs_lightgbm.py`. Placed in covenant_ml rather than cleargbm because covenant_ml already depends on both learners; cleargbm declares only numpy and would otherwise depend on its own competitor. Manifest: `libs/covenant_ml/docs/BENCHMARK_MANIFEST_2026-07-24.json`.

## [2026-07-21] add | cleargbm-perf-experiments-2026-07-21 empirical log
Pages written: cleargbm-perf-experiments-2026-07-21
Hubs updated: libs (+1 = 7 pages)
Index updated: total content pages 8 → 9
Notes: chronicles the 2026-07-21 perf-work session — five shipped commits (6.88s → 1.60s, 4.3× total speedup, gap to LightGBM 8.0× → 2.2×) plus three negative-result experiments (ordered arrays unpooled −4%, counts-backfill −5%, cnt_factor reconstruction −16% with quality drift). Meta-lesson recorded: splitting fused hot-loop work into more passes consistently loses more to overhead than it gains from bandwidth reduction — LightGBM patterns don't transfer 1:1 to cleargbm's context (sorted sample_indices, f64 everywhere, `unsafe_code = "forbid"`). Cross-links to tech-wiki pages that motivated each experiment (`~/PROJECTS/tech-wiki/pages/lightgbm-construct-histogram-inner.md`, `lightgbm-implicit-count-cnt-factor.md`, `lightgbm-score-t-float.md`, `lightgbm-prefetch-t0-macro.md`).

## [2026-07-21] update | cleargbm-perf-experiments phase I (bench harness fix — noise ceiling ±15% → ±1%)
Pages updated: cleargbm-perf-experiments-2026-07-21 (appended phase I section, retired phase-H final-benchmark table)
Notes: Austin flagged that bench noise (±0.15s on ~1s mean) was drowning sub-5% signals. Fixed via best-of-N in `benchmark_vs_lightgbm.py`: REPEATS_PER_MODEL=5 → 5 fits per seed × 3 seeds = 15 fits per model. Report MIN fit_time per seed (physically-correct estimator — CPU-work is bounded below, any slower run is background noise contamination; mean systematically overestimates by avg noise). Per-seed line now shows `fit=0.936s (min of 5: min/med/mean/max = 0.936/0.957/0.956/0.965s)` so noise band is visible. Result: LightGBM 0.83s ± 0.15s → **0.95s ± 0.01s**, cleargbm 0.98s ± 0.12s → **1.05s ± 0.01s**. Discriminable-Δ floor collapsed ~15% → ~1%. Cost: 5× wall-clock per bench. STABLE final: cleargbm **1.05s ± 0.01s** vs LightGBM 0.95s ± 0.01s = gap **1.10×** (10% slower). Cumulative 6.88s → 1.05s = 6.55× speedup, verified at ±1% precision. Wiki now carries an interpretable benchmark table for future experiments.

## [2026-07-21] update | cleargbm-perf-experiments phase H (LTO experiments + u32 sample_indices)
Pages updated: cleargbm-perf-experiments-2026-07-21 (appended phase H section)
Notes: two more experiments after phase G. (7) `[profile.release] lto = "fat"` regressed cleargbm 15% (1.05s → 1.20s); `lto = "thin"` was noise-neutral. Reverted — cargo defaults win here because our hot loop is already tight under `target-cpu=native` and LTO added no exploitable cross-crate boundary. (8) SHIPPED — sample_indices narrowed from usize (8 bytes on x86_64) to u32 (4 bytes) to match LightGBM's `data_size_t = int32` shape. New `crate::narrow::index_widen(u32) -> usize` — the second gated `#[expect(clippy::as_conversions)]` site in the crate (widening is infallible on all targets with usize ≥ 32 bits; try_from would add per-iteration branch in hot loop). Type-level lift across ~15 files: subsampling.rs, rng.rs (shuffle_partial), histogram/mod.rs (build_histogram* + reorder_grad_hess_into), hooks.rs (BuildHistogramFn), tree/{histograms,nodes,builder}.rs (BuildTreeInput.sample_indices, PendingNode, split_samples returns Vec<u32>, leaf_value_per_sample widen-at-write), pyo3_module (i64_slice_to_u32_vec at numpy boundary), all tests. 1133 lib tests pass. Result: cleargbm 1.05s → 0.98s ± 0.12s — mean ~7% better but well inside noise ceiling on this machine (bench stddev climbed 10× during phase H due to background process). Kept for structural cleanliness. Austin flagged the test-code refactor tax (50-100 literal edits per type lift across 5 test files) — testkit fixture-builder lift queued as follow-up (task #38).

## [2026-07-21] update | cleargbm-perf-experiments phase G (f32 gradient/hessian narrowing) — LightGBM parity within 1.13×
Pages updated: cleargbm-perf-experiments-2026-07-21 (appended phase G section)
Notes: SHIPPED — the LightGBM asymmetric-precision shape from `~/PROJECTS/tech-wiki/pages/lightgbm-score-t-float.md`. Narrow inputs (f32 gradients/hessians), wide accumulator (f64 histogram sums), widening via `f64::from(f32)` at write site (Rust equivalent of C++ `double += float`). New gated `crate::narrow::score_narrow` with the ONE `#[expect(clippy::as_conversions, clippy::cast_precision_loss)]` site in the entire crate — the rest still forbids `as` casts and precision-loss lints. Type-level lift across histogram/mod.rs, hooks.rs, tree/{histograms,nodes,builder}.rs, training/train.rs, and pyo3_module (numpy boundary narrows via `.iter().map(score_narrow).collect()`). Every test file that constructed Vec<f64> gradients/hessians updated to Vec<f32>. Result: cleargbm 1.23s → **1.05s ± 0.02s** (+15% faster), quality byte-parity except at f32 precision floor (AUC-ROC 0.6825 → 0.6822; AUC-PR 0.1416 → 0.1413; both within seed noise, still slightly higher AUC-PR than LightGBM's 0.1376). Gap to LightGBM: 1.47× → **1.13×**. Cumulative session: 6.88s → 1.05s (6.55× speedup); gap 8.0× → **1.13×** — within 13% of LightGBM. 1130 lib tests pass. Bench manifest: `libs/cleargbm/docs/BENCHMARK_MANIFEST_2026-07-21.json`.

## [2026-07-21] update | cleargbm-perf-experiments phase F (ordered-arrays-pooled + leaf-cache)
Pages updated: cleargbm-perf-experiments-2026-07-21 (appended phase F section)
Notes: two more experiments landed, BOTH wins. (4) ordered-arrays gather-elimination redone with proper amortization — hook-gated, per-node reorder reused across all n_features histogram builds AND across the smaller-child sibling-subtraction path (`libs/cleargbm_rs/src/histogram/mod.rs::build_histogram_ordered_trusted`, `hooks.rs`, `tree/histograms.rs`). Where experiment 1 lost 4% on unpooled allocation, the pooled version was neutral-to-slightly-positive. (5) LEAF-CACHE — profiling showed `predict_tree` at ~34% of wall-clock; tree builder already computes leaf value for every in-sample row, so `build_tree_with_leaf_assignment(input, hooks) -> (Tree, Vec<f64>)` returns the per-sample leaf assignments as a Vec (NaN sentinel for subsampled-out rows); training loop takes the fast path (direct lookup + add) and only falls back to `predict_tree` for NaN samples. When subsample=1.0 every row skips the tree walk. Result: cleargbm 1.60s → 1.23s (**+23% faster**), quality byte-identical (AUC-ROC/AUC-PR/log-loss/mean_pred/calibration_slope match to 4 decimals). Gap to LightGBM: 2.2× → **1.47×**. Cumulative session win: 6.88s → 1.23s = 5.6× speedup, gap 8.0× → 1.47×. Meta-lesson update: the two shipped wins share a pattern the three negatives lacked — amortization ratio matters (reuse over re-derive), and restructuring who owns already-computed information beats hot-loop micro-optimization once bytes-per-sample is minimized. 1124 lib tests pass; fresh bench at `libs/cleargbm/docs/BENCHMARK_MANIFEST_2026-07-21.json`.

## [2026-07-21] rewrite + add | cleargbm-histogram-split-path refreshed for Rust-only, 4 perf-roadmap pages added
Pages written: cleargbm-perf-column-major-sample-bins, cleargbm-perf-uint8-histogram-bins, cleargbm-perf-simd-histogram-accumulator, cleargbm-perf-leaf-wise-growth
Pages updated: cleargbm-histogram-split-path (rewritten to cite Rust sources; Python-fallback citations retired; fact_checked bumped to 2026-07-21)
Hubs updated: libs (+4, 2 -> 6 pages)
Index updated: total content pages 4 -> 8; libs hub count 2 -> 6
Notes: written after the cleargbm Rust-only refactor landed (commits bcca63a2, 1f20e166, 0696c31e, e8ddca8c, 98d57b2c). Each perf page is implementation-ready — cites the exact Rust file/line + function that needs to change, states expected impact against the 2026-07-21 benchmark baseline (cleargbm 6.88s ± 0.13s, LightGBM 0.87s ± 0.09s), lists prerequisites, and describes the test strategy. Order of implementation intended by the pages: column-major first (highest ROI, unblocks the rest), then uint8 bins, then SIMD, then leaf-wise (LAST — trades interpretability for capacity, do only if the other three don't close the gap). `confidence: high` on the column-major and uint8 pages (structural, cite fixed code); `confidence: medium` on SIMD and leaf-wise (impact estimates depend on ceiling assumptions not directly measured).

## [2026-07-24] correction | balanced-tree premise retracted from two pages
Pages updated: cleargbm-perf-leaf-wise-growth (§ "Interpretability cost" superseded; confidence note rewritten; fact_checked 2026-07-21 -> 2026-07-24; related +cleargbm-leaf-normalized-benchmarking), cleargbm-leaf-normalized-benchmarking (opening paragraph: "a full balanced tree" corrected)
Notes: both pages asserted that ClearGBM's depth-wise growth produces balanced trees with every leaf at roughly the same depth, and the leaf-wise page used that to argue leaf-wise growth would cost interpretability. A tree dump of a trained model refutes the premise: root-to-leaf path lengths range 4-6 at max_depth=5, 13 distinct features appear at depth 5 (so not oblivious either), and the leaf table already on the benchmarking page is itself disconfirming -- a full depth-6 tree has 64 leaves but ClearGBM measures 57.9 (47.15 on the authoritative run), because min_samples_split / min_samples_leaf / no-positive-gain retire branches early. The interpretability objection is therefore withdrawn, not weakened: the shape is already irregular, so leaf-wise changes which branches get deep, not whether any do. Rule-count runs the opposite way from the old assumption -- ClearGBM emits 47-58 leaves/tree against LightGBM's 31 for statistically tied quality, so depth-wise is the less readable of the two on a rules-to-read measure. Also recorded that no interpretability machinery (export_model_json, split-count importance, TreeSHAP, monotonic constraints) depends on tree shape, and that the real interpretability lever is oblivious trees -- a different change from leaf-wise. Meta-lesson: the claim had survived since 2026-07-21 because it was plausible from the algorithm name ("depth-wise") rather than read off a model; the disconfirming number was sitting in a table on a related page the whole time.

## [2026-07-30] add + correct | closed the cleargbm_rs doc gap left by the 2026-07-25 checkpoint
Pages written: cleargbm-f32-score-narrowing-reverted
Pages updated: cleargbm-perf-simd-histogram-accumulator (re-based against current code; fact_checked 2026-07-21 -> 2026-07-30), cleargbm-perf-experiments-2026-07-21 (Experiment 6 marked SUPERSEDED/REVERTED)
Hubs updated: libs (+1 = 9 pages; SIMD page description rewritten)
Index updated: total content pages 10 -> 11; libs hub count 8 -> 9

Notes: the crate's last commit is `8c06e47b` (2026-07-25) "checkpoint in-flight Rust core and ensemble work" — a rescue commit whose own message says the contents were "not authored or verified here." Nothing has touched `libs/cleargbm_rs` or `libs/cleargbm` since. Verified that checkpoint this session: `cargo fmt --check` clean, `cargo clippy --all-targets --all-features -D warnings` clean, 1209 Rust tests + 1 doc-test pass, `make rust-cov` segment gate 4027/4027 = 100.00%, guard 0 violations across 32 rule groups, ruff + strict mypy clean, 22 Python tests at 100% branch coverage. The code was finished; only its documentation was not.

Three fixes, each traced to the checkpoint rather than restated:

1. **`cleargbm-f32-score-narrowing-reverted` did not exist but was cited from code.** `src/narrow.rs:12` and `src/training/train.rs:199` both name that wiki slug as the source for reverting f32 grad/hess narrowing. The page was never written, so the only record of the decision was those two comments. Now written. Deliberately `confidence: medium`: the "8% slower" figure has no benchmark artifact anywhere — grep for f32/narrow/8% across `libs/cleargbm/docs/` returns nothing on this experiment, and the newest manifest (`BENCHMARK_MANIFEST_2026-07-24.json`, timestamp 2026-07-24T18:44:34) predates the revert. Mechanism is plausible (both widths fit in L2 at these node sizes once leaf-cache and ordered-arrays landed, so narrowing buys no bandwidth and adds a per-element widening); magnitude is unreproduced.

2. **A code comment contradicted its own code.** `src/histogram/mod.rs:104-107` still described the hot loop as reading f32 and widening via `f64::from(f32)`, while `HistogramRequest.ordered_gradients` is `&'a [f64]` and the loop adds f64 directly. Rewritten to state the f64-end-to-end reality and point at the new page. Only code change this session.

3. **The SIMD page described deleted code.** It was the last unshipped roadmap item but its "what's wrong today" cited a scalar `build_histogram` loop delegating to `HistogramBuffer::accumulate` with a per-sample bounds check and "zero unrolling" — all three false since `6a2d15b7`/the ordered-arrays work: `accumulate` is now test-only (every caller under `src/**/tests/`), the loop is `build_histogram_ordered_trusted` unrolled 8-wide, and the trusted path has no per-sample bounds check by design. Re-based, and the recommendation inverted: its Approach 1 conceded scatter has no native SIMD form on x86 and reduced to "load vectorized, unroll scalar scatters", which is exactly what the current hand-written loop already does — so only Approach 2 (bin-first reordering) is left, it has the same shape as the three already-measured pass-splitting losses, and the residual 1.426× raw gap is tree-shape (47.15 vs 30.96 leaves/tree) not histogram throughput, with per-leaf already 0.937×. Page now says do not start it as a perf play.

Also fixed, found while verifying the above: `cleargbm-histogram-split-path` cited `tree/builder.rs::compute_child_histograms`, which has lived at `tree/histograms.rs:199` since the `0fdb63f7` builder/nodes/histograms split. Locator corrected and `tree/histograms.rs` added to `source_paths`. `fact_checked` on that page deliberately NOT bumped — only this one citation was re-verified, not the page's O(K) measurement, reg_lambda history, or sigmoid-clip claims. It is still due a full audit.

Root cause for all three: the 2026-07-25 checkpoint committed a concurrent session's working tree to keep it recoverable, but that session never returned to update the wiki or reconcile its own comments — so the code advanced past its documentation and the citation chain broke in the middle. Rule for future rescue commits: a checkpoint of unverified work should carry a log entry naming what its docs still owe.

---
## [2026-07-31] correct | RustedWarfareBot was invisible to every monorepo-level doc
Pages written: none
Pages updated: none
Hubs updated: clients (rewritten — three clients enumerated with their lib surfaces; both game-bot wikis linked; TankpitBot page count corrected 50+ -> 67)
Index updated: clients hub description now names all three clients; libs line spells out covenant_domain/ml/nn/persistence and adds cleargbm_rs

Notes: `clients/RustedWarfareBot/` has its own README, Makefile, `agent/` JVM tree, `doctrines/`, `sweeps/`, and a full wiki, and is under active development — but appeared in **no** monorepo-level doc. Added to root `README.md` clients table, `docs/README.md` service-docs table, `docs/services.md` (a full entry, alongside a rewrite of the TankpitBot entry — which still advertised `RECOVER_FUEL`/`RECOVER_EQUIPMENT` modes and a `tankpit-probe` CLI that no longer exists), the directory trees in `docs/architecture.md` and `docs/development.md`, and the service→library matrix in `docs/architecture.md`.

Its matrix row is deliberately all-blank: RustedWarfareBot depends on `monorepo_guards` alone, no `platform_*` lib. Added a note under the matrix saying so, since a blank row otherwise reads as "not yet filled in" rather than "standalone by design".

The wiki's own structure needed no repair — 4 hubs, 11 pages, hub links resolve 1:1 against `pages/`, and every index count was already correct.

---
## [2026-07-31] audit | service README drift sweep; /healthz + /readyz contract made real
Pages written: none
Pages updated: pages/service-port-map.md (footnote 1 only — the "doc-extract-api is absent from the root README table" claim was stale; it is listed now)
Hubs updated: none
Index updated: none

Audited all 14 service READMEs against their code (routes registered, env vars read, pyproject deps, compose port mappings). Findings and fixes landed in the service docs, not here; the wiki-relevant part:

**The wiki's "every service exposes /healthz and /readyz" claim was false for four services** — grandma-api, transcript-api and opportunity-radar-api registered only `/healthz`, and procart-api registered `/health` and neither of the standard two. Rather than weaken the claim, the endpoints were added, so `pages/service-port-map.md` now describes the real contract: all 14 services expose both. transcript-api's `/readyz` checks Redis and worker presence (it enqueues STT jobs, so a reachable Redis with no worker is not ready); the other three have no queue or database and report ready whenever they serve, with the reasoning recorded in each handler's docstring. procart-api's `/health` was renamed rather than aliased, per the repo's no-backwards-compatibility rule, and its five referencing sites updated.

Also corrected in `transcript-api/src/transcript_api/api/routes/__init__.py:6`: its docstring had documented `GET /readyz — Readiness probe (checks Redis + workers)` for a route that was never registered. The docstring is now true rather than deleted.

Root cause worth recording: the health contract lived only in prose — the root README, this wiki, and each service's own README asserted it, but nothing enforced it. Four services drifted out of it without any check failing. A `monorepo_guards` rule asserting that every FastAPI service registers both routes would make the claim self-enforcing; not written this session.

---
## [2026-07-31] correct | doc-extract-api removed from the api monorepo
Pages written: none
Pages updated: pages/service-port-map.md (8012 row dropped, count 14 -> 13, Traefik router count thirteen -> twelve, footnotes renumbered), pages/platform-workers-rq-pattern.md (dropped from the platform_workers consumer list), hubs/services.md, index.md
Root docs updated: README.md services table

`services/doc-extract-api/` is gone. It was a superseded fork: the live implementation is `~/PROJECTS/MCPs/doc-extract-api`, which runs as the `mcp-doc-extract-api` container on 127.0.0.1:8018, carries 26 source modules to the api copy's 15, and tracks the corvis schema (`tenant_memberships`, `content_hash` dedup via `ON CONFLICT (tenant_id, content_hash)`, a `startup.py` that asserts expected migrations). Nothing in the api monorepo imported `doc_extract_api`, and the MCPs copy has the same pdfplumber + docTR extraction modules, so nothing was lost.

How the drift stayed invisible: the api copy's only link to corvis was through its *tests*, which walked up to `../MCPs/.env` for `DATABASE_TEST_URL` and wrote into `corvis_test`. Its production code was a snapshot of a schema corvis had moved past three times — `users` replaced by `accounts` + `tenant_memberships`, `content_hash` added NOT NULL, and `documents.category` promoted to a foreign key into the `document_categories` taxonomy (`irvine.council.agenda`-style slugs) while the service still validated against a flat 15-value list matching nothing. Those failures never surfaced because `make check` stops at the guard step, and this project's guards were red, so its tests had not run.

Rule this suggests: a service in this monorepo should not reach into another repo's `.env` for a database. That coupling is what let a dead fork keep "passing" while diverging from the schema it depended on.

## [2026-08-17] add+update | cleargbm boundary page, leaf-wise reference-implementation research folded in

Pages written: cleargbm-python-rust-boundary (new)
Pages updated: cleargbm-perf-leaf-wise-growth (design-alternative section: LightGBM's flat best_split_per_leaf_ + ArgMax vs this page's BinaryHeap sketch; the Shi-removal vs LightGBM-gain-poisoning fork a leaf-wise arm must decide; pointer to agent-board task 453c9234 which specifies growth_strategy + the paired-quality gate), cleargbm-histogram-split-path (stale _rust.py blob pin 17d203e5 -> 81518f4d; all 10 pins re-verified against the working tree before bumping fact_checked)
Hubs updated: libs (9 -> 10)
Index updated: total 11 -> 12
Notes: The boundary page records the post-ea7835d2 state: f7c61172 (07-21) deleted the compute path, ea7835d2 (08-17) deleted the packaging shim; maturin top-level module; _rust.py Protocol layer as the only Python mirroring Rust signatures. Primary-source backing for the leaf-wise design (captured serial_tree_learner.cpp, Shi 2007 thesis, XGBoost + Friedman papers) landed the same night in the TECH wiki, gradient-boosting-implementation hub; this wiki carries the codebase-side pointers only.

## [2026-08-18] update | NavProbe listed as a client; growth-policy scripts repointed after a package move

Hubs updated: clients (NavProbe added to the client list and to the dedicated-wiki section; the section header no longer says "both game bots", which was wrong twice over)
Index updated: clients hub description (NavProbe named, and the dedicated-wiki count corrected from two clients to three)
Pages updated: cleargbm-perf-leaf-wise-growth (footnote 11 repointed at the new script and package locations)
Root docs updated: README.md Clients table gains NavProbe
Notes: NavProbe had been on disk since 08-13 with 27 wiki pages of its own and appeared in neither the README nor this wiki, so the monorepo's own surface did not list one of its clients. It is a determinism instrument rather than a user-facing client and the hub says so, because filing it beside three bots without that sentence would misdescribe it.

The growth-policy experiment scripts moved from libs/cleargbm/scripts/ to libs/covenant_ml/scripts/, and the measurement logic they used to inline now lives in the covenant_ml.growth_policy package. The move was not tidying: libs/cleargbm depends only on numpy and cleargbm_rs, so xgboost, lightgbm and scikit-learn are all absent from its environment and the dataset path resolved only from libs/covenant_ml — the scripts could not have run where they were filed. libs/covenant_ml carries all three vendors plus the dataset at the exact relative path they read.

Rule this suggests: a script belongs in the package whose environment can run it, and "which venv has these imports" is worth checking before filing one. Two gates agreed the scripts were misplaced and neither was consulted at the time — libs/cleargbm's make check was red on them for strict-mypy, ruff ANN, the guards' print ban and its 100% coverage gate over scripts/, all at once.

## [2026-09-01] fill | the first two non-ClearGBM pages in a month, and a backend count that drifted

Pages written: `covenant-radar-backend-registry` (services), `determinism-env-read-once-at-library-load` (libs + infrastructure + clients)
Hubs updated: services, libs, infrastructure, clients
Index updated: 25 -> 27 pages; services 1 -> 2, libs 23 -> 24, infrastructure 1 -> 2; a stated coverage-shape paragraph added
Notes: this wiki was 22 ClearGBM pages out of 25. The index now says so in its own header, because a reader finding four hubs and no service pages should be told which silences are deliberate. Client depth IS deliberate — TankpitBot, RustedWarfareBot and NavProbe keep their own full wikis. Service and infrastructure depth is a real gap, and naming it is cheaper than letting the next session rediscover it.

**The clients hub carried stale counts for its own siblings**: TankpitBot 67 (actual 75) and NavProbe 27 (actual 37). Both now carry the date they were counted.

**The backend count.** `services/covenant-radar-api/README.md` says "Eleven model backends behind one interface — seven classifiers … plus four regressors", and elsewhere "all four `*_reg` backends". The seven is right. The four is not: `RegressorBackendName` is a `Literal` of five, so the total is twelve. The declaration is split across two libraries, which is how the miscount survives — `covenant_ml`'s `default_regressor_registry` wires three, and `mlp_reg` / `lstm_reg` come from `covenant_nn`. **Read the `Literal`, not either registry.**

Two commits date the divergence: `7e9b23d0` (2026-08-05) wrote the README's wording, `46b8d4a5` (2026-08-21) put the five-name Literal in `types_regression.py`. That second commit is a 32-file role split, so it dates the declaration's arrival in the file rather than in the codebase — the page says exactly that and no more, after the `no-deferral` rule refused an earlier draft that hedged with "not established here". The rule's message is worth repeating: *cite the source you DID read, or drop the claim*.

Not fixed here, deliberately: the README itself. Editing it to twelve buys agreement until the next backend lands. The durable form is a test asserting the README's count against `len(get_args(RegressorBackendName))`, which is a code change and belongs in its own commit.

**Why the determinism page is here rather than only in NavProbe's wiki.** `platform_core.determinism_env` is where this monorepo's run-comparability story actually lives, and it is unusually well-documented in place — measurements with dates, drivers and torch versions, and an explicit statement of what the control does NOT buy (not one of 72 SDPA digests moves, because the memory-efficient attention kernel is not a cuBLASLt call). NavProbe measures determinism in someone else's simulator; this configures it in ours. The clients hub now names that symmetry and links across, which is why the page declares three hubs.

Verified: both pages `wiki_audit_page` 0 errors / 0 warnings. The determinism page's first pass failed `hubs-membership-consistent` — the cross-link from hubs/clients.md counted as membership while the frontmatter listed only libs + infrastructure. The check reads prose links, not just inclusion-list lines.
