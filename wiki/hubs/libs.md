# Libs

Shared libraries in `libs/` — the reusable core that services and clients build on. Search here BEFORE adding a cross-cutting helper. Two families:

- **platform_*** — infrastructure primitives: `platform_core` (config, logging, HTTP, FastAPI utils, OAuth 2.0 + PKCE), `platform_workers` (RQ + Redis), `platform_ml` (ML artifact storage, device auto-detection), `platform_discord`, `platform_music`, `platform_email` (Outlook Graph + Gmail), `platform_calendar` (Google Calendar), `platform_codebase`, `platform_devpost` (Devpost hackathon discovery + codebase capability matching), `platform_kaggle`, `platform_stt` (Whisper), `platform_langid` (Meta MMS-LID), `platform_translate` (Anthropic + OpenAI backends).
- **domain libs** — narrower vertical stacks: `covenant_domain` / `covenant_ml` / `covenant_nn` / `covenant_persistence` (loan-covenant modeling), `cleargbm` + `cleargbm_rs` (interpretable gradient boosting, numpy + Rust), `procart` (procedural art / neon HDR).
- **instrument_io** — scientific instrument format readers (mass spec, mzML, Excel, PDF).
- **monorepo_guards** — the code-quality rule engine (20+ static analysis checks, Python + Rust).

[platform_workers — RQ + Redis pattern](../pages/platform-workers-rq-pattern.md) -- typed Redis client factories (`redis_for_kv` / `redis_for_rq` / `redis_raw_for_rq` / `redis_for_pubsub`) + RQ harness (`rq_queue`, `rq_retry`) + `readyz_redis` health helper
[ClearGBM histogram-based split finding](../pages/cleargbm-histogram-split-path.md) -- O(K) prefix-sum bin scan, sibling subtraction, reg_lambda semantics, Rust-only compute path
[ClearGBM perf — column-major sample_bins layout](../pages/cleargbm-perf-column-major-sample-bins.md) -- transpose FeatureBins.sample_bins from row-major Vec<Vec<usize>> to column-major flat Vec<usize>; contiguous per-feature scans, 20-40% faster fit (highest ROI perf change)
[ClearGBM perf — uint8 histogram bin dtype](../pages/cleargbm-perf-uint8-histogram-bins.md) -- bin index dtype usize -> u8, cap max_bins <= 255, 8x more values per cache line, 30-60% faster fit; land alongside column-major refactor
[ClearGBM perf — SIMD histogram accumulator](../pages/cleargbm-perf-simd-histogram-accumulator.md) -- vectorize the scalar per-sample += loop via `wide` crate (Approach 1) or bin-first reordering + gather-add (Approach 2); requires column-major + uint8 to hit ceiling
[ClearGBM perf — leaf-wise tree growth](../pages/cleargbm-perf-leaf-wise-growth.md) -- replace depth-first LIFO stack with best-first max-heap over split gains; capacity gain, not speed; do LAST — trades interpretability for capacity

<!-- Add pages here as they're written. Format: [Title](../pages/<slug>.md) -- one-line description -->
