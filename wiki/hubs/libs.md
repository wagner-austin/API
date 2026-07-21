# Libs

Shared libraries in `libs/` — the reusable core that services and clients build on. Search here BEFORE adding a cross-cutting helper. Two families:

- **platform_*** — infrastructure primitives: `platform_core` (config, logging, HTTP, FastAPI utils, OAuth 2.0 + PKCE), `platform_workers` (RQ + Redis), `platform_ml` (ML artifact storage, device auto-detection), `platform_discord`, `platform_music`, `platform_email` (Outlook Graph + Gmail), `platform_calendar` (Google Calendar), `platform_codebase`, `platform_devpost` (Devpost hackathon discovery + codebase capability matching), `platform_kaggle`, `platform_stt` (Whisper), `platform_langid` (Meta MMS-LID), `platform_translate` (Anthropic + OpenAI backends).
- **domain libs** — narrower vertical stacks: `covenant_domain` / `covenant_ml` / `covenant_nn` / `covenant_persistence` (loan-covenant modeling), `cleargbm` + `cleargbm_rs` (interpretable gradient boosting, numpy + Rust), `procart` (procedural art / neon HDR).
- **instrument_io** — scientific instrument format readers (mass spec, mzML, Excel, PDF).
- **monorepo_guards** — the code-quality rule engine (20+ static analysis checks, Python + Rust).

[platform_workers — RQ + Redis pattern](../pages/platform-workers-rq-pattern.md) -- typed Redis client factories (`redis_for_kv` / `redis_for_rq` / `redis_raw_for_rq` / `redis_for_pubsub`) + RQ harness (`rq_queue`, `rq_retry`) + `readyz_redis` health helper
[ClearGBM histogram-based split finding](../pages/cleargbm-histogram-split-path.md) -- O(K) prefix-sum bin scan, sibling subtraction, reg_lambda semantics on the runtime histogram path, Rust vs Python fallback boundary

<!-- Add pages here as they're written. Format: [Title](../pages/<slug>.md) -- one-line description -->
