# Model Trainer — Design Document

## 1. Vision & Goals

A modular, robust, and strictly typed system to train and evaluate language models (GPT‑2, Char-LSTM) and tokenizers (BPE, with optional SentencePiece). The system is API‑first with Discord bot integration for user interaction (via `clients/DiscordBot`). It emphasizes reliability, observability, and maintainability. Supports both CPU and GPU (CUDA) execution with mixed-precision training (FP16/BF16) for faster iteration on GPU hardware.

### Primary Goals
- Pluggable model backends via a standard contract (GPT‑2, Char-LSTM implemented; LLaMA, Qwen planned).
- Pluggable tokenizer backends via a standard contract (BPE implemented; SentencePiece optional when binaries are available).
- GPU acceleration with CUDA support and automatic device selection.
- Mixed-precision training (FP16/BF16) with automatic loss scaling for faster GPU training.
- CPU fallback with multi-threaded preprocessing/tokenization.
- Strict typing end-to-end; no `Any`, no casts, no `type: ignore`.
- Centralized, structured logging via platform_core; zero silent exceptions.
- Explicit dependency injection via a service container.
- Discord bot integration for training control and monitoring (via `clients/DiscordBot`).
- Deterministic, reproducible runs with clear artifacts and manifests.

### Non-Goals (Current)
- Distributed multi-GPU training.
- Fine-grained model parallelism or quantization.
- Advanced hyperparameter tuning or AutoML.

### Constraints & Assumptions
- Supports both CPU and GPU (CUDA) execution; GPU recommended for faster training.
- Model weight access for certain families (e.g., LLaMA) may require gated download.
- Avoid vendor lock-in. Cloud is optional and deferred.

## 1.1 Current Implementation Snapshot (MVP)

- API-first system (FastAPI) exposing tokenizers, runs, artifacts, inference, and health endpoints.
- Tokenizers: BPE implemented; SentencePiece optional if `spm_*` binaries are available.
- Models: GPT‑2 and Char-LSTM backends implemented (CPU configurations) with train, eval, and inference.
- Orchestrators: enqueue training/eval/inference with structured logging via platform_core.
- Workers: RQ worker performs training/eval/inference, emits heartbeats, handles cancellation, and publishes generic job lifecycle events plus trainer metrics to Redis (`trainer:events`).
- Inference: text generation and scoring via async jobs (`/runs/{id}/generate`, `/runs/{id}/score`).
- Logging: Centralized JSON logging using platform_core.logging with structured extra fields (category, service, run_id, etc.).
- Storage: all run manifests live under `artifacts/models/<run_id>/`.

## 1.2 Project Structure (Current)

```
src/
  model_trainer/
    api/
      main.py
      middleware.py
      routes/
        health.py, runs.py, tokenizers.py
      schemas/
      validators/
    core/
      config/settings.py
      contracts/
        compute.py, conversation.py, dataset.py, model.py, queue.py, tokenizer.py
      compute/device_selector.py
      infra/paths.py, redis_utils.py
      logging/types.py, utils.py
      services/
        container.py
        data/corpus.py, corpus_fetcher.py, corpus_cache_cleanup.py
        dataset/local_text_builder.py
        model/backends/, backend_factory.py, unavailable_backend.py
        queue/rq_adapter.py
        storage/artifact_cleanup.py
        tokenizer/bpe_backend.py, char_backend.py, spm_backend.py, tokenizer_cleanup.py
        training/base_trainer.py, dataset_builder.py
    infra/
      storage/run_store.py
      persistence/models.py
    maintenance/cleanup.py
    orchestrators/
      training_orchestrator.py, tokenizer_orchestrator.py, inference_orchestrator.py, conversation_orchestrator.py
    worker/
      train_job.py, eval_job.py, tokenizer_worker.py, generate_job.py, score_job.py, chat_job.py
      job_utils.py, manifest.py, trainer_job_store.py
    worker_entry.py
tests/
scripts/
  cleanup_tokenizers.py
  cleanup_corpus_cache.py
```


## 2. User Flow (MVP)
1. Select a model backend (e.g., GPT‑2 Small) and version (via API).
2. Select a tokenizer backend (e.g., BPE) and configure vocab size.
3. Point to a cleaned text corpus path; configure holdout split for tokenizer and validation.
4. Train tokenizer; validate on holdout (coverage, OOV rate, basic stats).
5. Train model on CPU with basic metrics (loss, perplexity).
6. Monitor logs and progress via API endpoints; view run summary and artifacts.
7. Export artifacts (tokenizer files, checkpoints, training manifest) from `artifacts/`.


## 3. High-Level Architecture

- UI (Web, TypeScript): planned; interacts with API via JSON.
- API Server (Python, FastAPI):
  - Exposes training runs, tokenizer building, model selection, and artifact browsing.
  - Hosts orchestrators that call core services via DI container.
- Core Services (Python):
  - Tokenizer service (pluggable backends via contracts).
  - Model training service (pluggable backends via contracts).
  - Data ingestion and dataset service.
  - Compute service (local CPU now; cloud later).
  - Logging service (centralized JSON logging with categories).
  - Error service (typed errors, standardized error codes, global handlers).
- Adapters / Backends:
  - Model backends: GPT‑2 (Transformers) implemented; LLaMA/Qwen planned (unavailable placeholders present).
  - Tokenizer backends: Hugging Face Tokenizers (BPE) implemented; SentencePiece optional (requires `spm_*` binaries).
- Logging:
  - Centralized structured logging via `platform_core.logging` with `setup_logging()` at startup.
  - Module-level loggers obtained via `get_logger(__name__)`.
  - Structured extra fields: category, service, run_id, tokenizer_id, event, error_code, etc.
  - JSON format for production; text format for development.
- Storage & Artifacts:
  - Local directory structure for artifacts (tokenizers, models) under `artifacts/`.
  - Per‑run manifest for reproducibility at `artifacts/models/<run_id>/manifest.json`.


## 4. Technology Choices

- Language (Core/API): Python 3.11+
  - Strict typing: mypy (strict mode), no `typing.Any`, no `typing.cast`.
  - Pydantic v2 for all data models (configs, DTOs, results). Prefer `BaseModel` with `extra="forbid"` and `validate_assignment=True`. Avoid `@dataclass` for these shapes.
- FastAPI + Hypercorn for API with typed routes and models.
  - Hugging Face: Transformers, Datasets, Tokenizers.
  - PyTorch (CPU) for training; CPU threading configured explicitly.
  - Ruff + mypy in pre-commit; black/ruff format if adopted by repo.
- Language (UI): planned; not yet implemented.
- Logging: Centralized via `platform_core.logging` (JSON formatter, structured fields).
- Testing: pytest + mypy; Playwright or Cypress optional for UI later.
- Queue: Redis + RQ (via `redis-py`) for durable training jobs and status updates.


## 5. Project Structure (Proposed)

```
Model-Trainer/
  DESIGN.md
  README.md (later)
  pyproject.toml
  src/
    model_trainer/
      api/
        routes/               # FastAPI routers (typed)
        schemas/              # Pydantic models (strict)
      core/
        contracts/            # Protocols and typed interfaces
        logging/
        errors/
        config/
        services/
          container.py        # Service container (DI)
          tokenizer/
            backends/         # BPE, SentencePiece adapters
          model/
            backends/         # GPT-2, LLaMA, Qwen adapters
          data/
          training/
          compute/
      infra/
        storage/
        persistence/
      orchestrators/
        tokenizer_orchestrator.py
        training_orchestrator.py
  tests/
  ui/
    package.json
    src/
      app/                    # Routes
      components/
      lib/                    # Typed API client
```


## 6. Contracts (Strict, Stable Interfaces)

Use Python `typing.Protocol` to define implementation-agnostic contracts. No `Any`; use precise generics and TypedDicts/NamedTuples where useful. Example signatures focus on clarity and future extensibility.

### 6.1 Tokenizer Contracts

- `TokenizerConfig`: Pydantic model including `method`, `vocab_size`, `min_frequency`, normalization options, and `special_tokens`.
- `TokenizerTrainRequest`: input corpus path(s), holdout fraction, seed, threads.
- `TokenizerStats`: coverage, OOV rate, token count, char coverage.
- `TokenizerArtifact`: paths to vocab/merges or model files, manifest.

Protocol:
- `TokenizerBackend`:
  - `name() -> str`
  - `supported_methods() -> set[TokenizerMethod]`
  - `train(config: TokenizerConfig, request: TokenizerTrainRequest) -> TokenizerArtifact`
  - `load(artifact_path: str) -> TokenizerHandle`
  - `inspect(handle: TokenizerHandle) -> TokenizerStats`
  - `encode(handle: TokenizerHandle, text: str) -> list[int]`
  - `decode(handle: TokenizerHandle, ids: list[int]) -> str`

Notes:
- Provide BPE via Hugging Face Tokenizers; SentencePiece via `sentencepiece`.
- Multi-threaded training via library options and parallel corpus streaming.

### 6.6 Data Models Policy (Pydantic-first)
- Use `pydantic.BaseModel` for:
  - API schemas, configuration objects, contract types (e.g., `TokenizerTrainConfig`, `ModelTrainConfig`), and result/DTO types (e.g., `TrainOutcome`, `TokenizerTrainStats`).
- Use `TypedDict` for JSON-like manifests persisted to disk (e.g., training manifest structures).
- Use plain classes (or non-frozen dataclasses) for services/containers where runtime validation is not needed and unit tests may monkeypatch attributes.

Guard enforcement:
- Disallow `@dataclass(frozen=True)` and any `@dataclass` in `core/contracts` and `core/config`.
- Disallow `typing.Any`, `typing.cast`, `type: ignore`.
- Disallow `print()` in library code; allowed in tests.

### 6.2 Model Contracts

- `ModelConfig`: model family, size, max_seq_len, optimizer config, LR schedule, batch sizes, seeds, num_epochs, gradient clipping.
- `TrainingDataConfig`: dataset path(s), tokenizer artifact ref, splits, shuffle buffer, num_workers.
- `TrainerConfig`: checkpoint intervals, eval intervals, early stopping, logging cadence, CPU thread limits.
- `TrainingRunManifest`: exact config, versions, seeds, git commit (if any), start/end timestamps, metrics summary.

Protocol:
- `ModelBackend`:
  - `name() -> str`
  - `supported_tokenizers() -> set[TokenizerMethod]`
  - `prepare(config: ModelConfig, tokenizer: TokenizerHandle) -> PreparedModel`
  - `train(prepared: PreparedModel, data_cfg: TrainingDataConfig, trainer_cfg: TrainerConfig) -> TrainingResult`
  - `evaluate(prepared: PreparedModel, split: str) -> EvalResult`
  - `save(prepared: PreparedModel, out_dir: str) -> ModelArtifact`
  - `load(artifact_path: str, tokenizer: TokenizerHandle) -> PreparedModel`

Notes:
- Initial implementation leverages Hugging Face AutoModelForCausalLM, DataCollator, and Trainer on CPU with careful threading.

### 6.3 Data Contracts
- `CorpusProvider` Protocol: streams normalized text; exposes sizes for train/holdout.
- `DatasetBuilder` Protocol: builds tokenized datasets and returns typed dataset handles compatible with backends.

### 6.4 Compute Contracts
- `ComputeProvider` Protocol:
  - `kind() -> Literal['local-cpu', 'cloud']`
  - `threads() -> int`
  - `env() -> dict[str,str]`
  - For future cloud: job submit/cancel/status.


## 7. Dependency Injection (Service Container)

- A small, explicit DI container with:
  - Registration by typed token (Python `Type` or `Literal` keys).
  - Constructor injection via factories; no global singletons.
  - Explicit lifetime scopes (app, request/run, ephemeral).
- Container composes:
  - Settings, Redis client, RQ enqueuer.
  - TokenizerRegistry (maps names -> `TokenizerBackend`).
  - ModelRegistry (maps names -> `ModelBackend`).
  - Orchestrators (TrainingOrchestrator, TokenizerOrchestrator).
  - DatasetBuilder.

Benefits:
- Clear boundaries and mocking in tests; easy backend swaps.
- Logging is global via `platform_core.logging.get_logger(__name__)`, not injected.


## 8. Logging (Centralized, Structured via platform_core)

**Implementation:**
- Uses `platform_core.logging` for all logging (shared with turkic-api, data-bank-api).
- `setup_logging()` called once at application startup (main.py, workers, scripts).
- Module-level loggers: `logger = get_logger(__name__)`.
- No wrapper classes or LoggingService—direct use of platform_core.

**Configuration:**
- `LOGGING__LEVEL`: DEBUG, INFO, WARNING, ERROR, CRITICAL (validated via `narrow_log_level()`).
- Format: JSON for production, text for development (set via `LogFormat`).
- Service name: "model-trainer", "model-trainer-worker", "model-trainer-cleanup".

**Structured Fields:**
- Standard: `timestamp`, `level`, `logger`, `message`, `service`, `instance_id`.
- Extra fields (via `extra={}` dict):
  - `category`: api, orchestrator, tokenizer, training, worker, etc.
  - `service`: specific service context (runs, tokenizers, health, etc.).
  - `event`: semantic event name (enqueued, completed, failed, etc.).
  - `run_id`, `tokenizer_id`: correlation identifiers.
  - `error_code`: typed error codes from AppError.
  - Domain-specific: model_family, vocab_size, loss, perplexity, etc.

**Type Safety:**
- `LoggingExtra` TypedDict defines all allowed extra fields (total=False).
- No `Any` types; all fields strictly typed.
- Extra fields list defined in `LOGGING_EXTRA_FIELDS` tuple.

**Output:**
- Stdout (JSON or text format based on config).
- Third-party libs silenced (urllib3, httpx at WARNING level).

**Logging Policy:**
- No prints in code; only logger calls.
- Include progress markers (epochs, steps, evals) via structured events.
- All exceptions logged with context before propagation.


## 9. Error Handling (Strict, No Silent Failures)

- Typed `AppError` with `ErrorCode` enum (e.g., `DATA_NOT_FOUND`, `TOKENIZER_TRAIN_FAILED`, `MODEL_TRAIN_FAILED`, `CONFIG_INVALID`).
- API layer maps `AppError` to HTTP error responses with structured JSON body.
- Orchestrators catch known exceptions, convert to `AppError`, log with context, re-raise.
- Unknown exceptions bubble to a global handler that logs with stack and returns a sanitized error body.

Policy (enforced by guard):
- Broad catches (`except:`, `except Exception`, `except BaseException`) must both log (error/exception) and re-raise.
- Specific exception catches must at least log or re-raise; prefer structured logging and typed re-throws.


## 10. Configuration & Reproducibility

- Typed config models (Pydantic v2) for:
  - `AppConfig`, `LoggingConfig`, `TokenizerConfig`, `ModelConfig`, `TrainerConfig`, `DataConfig`, `ComputeConfig`.
- Config sources: `TOML` files + env var overrides.
- Every run writes a `manifest.json` including config, library versions, seed, CPU info, durations, artifacts.
- Seed all random sources (Python, NumPy, PyTorch, HF datasets where applicable).


## 11. Data Pipeline & Tokenizer Training

- Input: UTF‑8 text files under a corpus directory or a single combined file.
- Preprocessing: minimal normalization options in MVP (unicode normalization, lowercasing optional).
- Holdout: stratified-like split by file boundary or line boundary; deterministic via seed.
- Tokenizer training:
  - BPE via `tokenizers` with multi-threaded trainer; `vocab_size`, `min_frequency`.
  - SentencePiece optional (Unigram/BPE); use subprocess training with controlled threads.
- Validation metrics: coverage, OOV (if applicable), average tokens per char/word.
- Artifacts: tokenizer files + `tokenizer_manifest.json` with training stats.


## 12. Model Training

- Backend: Hugging Face Transformers + PyTorch (CPU or CUDA).
- Data: HF `datasets` with streaming or on-disk arrow, tokenized with selected tokenizer.
- Collation: causal LM collator with static or dynamic padding to `max_seq_len`.
- Trainer: Custom `BaseTrainer` with AMP (Automatic Mixed Precision) support.

### Device Selection
- `device: "auto"` resolves to CUDA when available, otherwise CPU.
- `device: "cuda"` requires CUDA; raises error if unavailable.
- `device: "cpu"` forces CPU execution.

### Mixed-Precision Training
- `precision: "auto"` resolves based on device:
  - CUDA: defaults to FP16 for faster training
  - CPU: defaults to FP32 (mixed precision not beneficial)
- `precision: "fp16"`: Uses `torch.amp.autocast` with `GradScaler` for loss scaling (CUDA only).
- `precision: "bf16"`: Uses `torch.amp.autocast` without scaler (more numerically stable, CUDA only).
- `precision: "fp32"`: Full precision, works on any device.

### DataLoader Configuration
- CUDA: `num_workers = min(4, cpu_count)`, `pin_memory = true` for faster data transfer.
- CPU: `num_workers = 0`, `pin_memory = false` (lightweight default).

### Batch Size Defaults (CUDA)
- GPT-2: bumps small batches to 32
- Char-LSTM: bumps small batches to 64
- Other: bumps small batches to 16

- Metrics: loss, perplexity (eval), samples/sec (indicative), training time.
- Checkpointing: step/epoch intervals; best checkpoint by eval loss.


## 13. Storage & Artifacts

Directory layout (local):
```
corpus/              # user-provided corpus root (mounted read-only in containers)
artifacts/
  tokenizers/
    <tokenizer_id>/
      tokenizer.json | tokenizer.model
      manifest.json
  models/
    <run_id>/
      manifest.json
      eval/
        metrics.json
```

Notes:
- There is no separate runtime `runs/` directory; per‑run manifests live under `artifacts/models/<run_id>/`.
- Logs are centralized via platform_core (stdout) with structured extra fields for correlation.


### 13.1 Artifact Storage Policy

Upload and cleanup lifecycle:

1. Training completes and the model is saved to `/data/artifacts/models/{run_id}/`.
2. The worker uploads the entire run directory (weights, manifest, logs) as a tarball to `data-bank-api` and receives a `file_id`.
3. The worker persists `runs:artifact:{run_id}:file_id` in Redis and records the pointer in the run manifest.
4. After upload and pointer persistence succeed, the local run directory is deleted by the `ArtifactCleanupService`, subject to `CleanupConfig`.

Evaluation and log access:

- Evaluation jobs download and extract the tarball from `data-bank-api` via `ArtifactDownloader`, recreating the complete directory structure locally.
- Logs are included in the tarball and available after extraction.
- The migration to download-based eval and log access is complete; no service depends on the original training directory after upload.

Storage locations:

- `data-bank-api` owns permanent storage for completed runs. Files are addressed by `file_id` (SHA256) and accessed via the `/files` API.
- Local artifacts under `/data/artifacts/models/{run_id}` are temporary during training and may be recreated on-demand from data-bank for eval/inspection.

Configuration:

- `APP__CLEANUP__ENABLED=true|false` controls whether post-upload cleanup runs.
- `APP__CLEANUP__VERIFY_UPLOAD=true|false` controls whether Redis must contain `file_id` before cleanup.
- `APP__CLEANUP__DRY_RUN=true|false` logs what would be deleted without deleting.
- `APP__CLEANUP__GRACE_PERIOD_SECONDS=N` waits `N` seconds between upload and cleanup.

Observability:

- Cleanup events emit structured logs with fields such as:
  - `event`: `"cleanup_dry_run"`, `"cleanup_completed"`, `"cleanup_failed"`.
  - `run_id`, `path`, `bytes_freed`, `files_deleted`.

Corpus cache and tokenizer artifacts:

- Corpus cache entries live under `Settings.app.data_root / "corpus_cache"` and are treated as pure cache. They can be reclaimed by `CorpusCacheCleanupService` according to size and free-space thresholds in `CorpusCacheCleanupConfig`.
- Tokenizer artifacts live under `artifacts/tokenizers/{tokenizer_id}`. `TokenizerCleanupService` can delete tokenizers that are not referenced in any model manifest and older than `TokenizerCleanupConfig.min_unused_days`.

## 14. API Surface (MVP)

- `POST /tokenizers/train` — start tokenizer training
- `GET /tokenizers/:id` — get tokenizer info/stats
- `POST /runs/train` — start model training
- `GET /runs/:id` — get run status/metrics
- `GET /runs/:id/logs` — tail per‑run logs
- `GET /runs/:id/logs/stream` — SSE stream of logs
- `GET /artifacts/:kind/:id` — download artifact
- `GET /healthz` — lightweight health
- `GET /readyz` — dependency readiness (e.g., Redis reachable)
- `POST /runs/:id/evaluate` — run evaluation on a split/dataset
- `GET /runs/:id/eval` — fetch evaluation results/summary
- `POST /runs/:id/generate` — enqueue text generation job
- `GET /runs/:id/generate/:request_id` — poll generation results
- `POST /runs/:id/score` — enqueue text scoring job
- `GET /runs/:id/score/:request_id` — poll scoring results

All request/response schemas are strictly typed Pydantic models. No `Any`.


## 15. User Interface (Discord Bot)

The user interface is implemented as a Discord bot (`clients/DiscordBot`):

- **Commands** (`cogs/trainer.py`):
  - `/train` — Start a new training run with model/tokenizer/corpus selection.
  - `/status <run_id>` — Check training status and progress.
  - `/cancel <run_id>` — Cancel an in-progress training run.

- **Notifications** (`services/jobs/trainer_notifier.py`):
- Subscribes to `trainer:events` Redis channel.
  - Sends DM updates to users on training started/progress/completed/failed.
  - Rich embeds with loss, perplexity, and progress metrics.

- **Integration**:
- Uses `platform_core.job_events` (lifecycle) and `platform_core.trainer_metrics_events` (metrics) for typed event decoding.
  - Uses `platform_core.model_trainer_client` for HTTP API calls.
  - Correlates events via `user_id` to route notifications to the correct Discord user.


## 16. Testing & Quality Gates

- mypy: `--strict` with rules enforcing:
  - `disallow_any_* = True`, `no_implicit_optional = True`.
  - Disallow untyped defs and calls; require return types.
- pytest unit and integration tests for:
  - Tokenizer backends: training/encode/decode roundtrips on toy corpus.
  - Model backends: minimal steps to ensure loops execute and artifacts are produced.
  - Orchestrators: happy-path flows and error injection.
- Pre-commit: ruff, mypy, black (if used).


## 17. Concurrency & Performance Notes

- Tokenization: use library-native parallelism (threads) for training and encoding.
- Data pipeline: DataLoader `num_workers` tuned to CPU cores; chunked line reading.
- Avoid Python GIL hotspots by relying on native code paths (tokenizers, PyTorch, datasets).
- Limit memory footprint: streamed datasets, small batch sizes; gradient accumulation only if needed.


## 18. Security & Licensing

- Respect gated weights (e.g., LLaMA) and license checks; user supplies access tokens/agreements.
- Cache isolation and path validation for user-supplied corpus paths.


## 19. Roadmap

- MVP (CPU + GPU): ✅ Complete
  - BPE tokenizer (HF Tokenizers) + GPT‑2/Char-LSTM training end-to-end.
  - GPU acceleration with CUDA support and automatic device selection.
  - Mixed-precision training (FP16/BF16) with automatic loss scaling.
  - Strict types, DI container, centralized logging, error service.
  - Discord bot integration for training control and notifications.
- Next:
  - SentencePiece tokenizer backend (optional, requires binaries).
  - LLaMA/Qwen adapters via Transformers with type-safe wrappers.
  - Resume/continue runs and artifact browser.
- Later:
  - Cloud compute provider (job submission + remote logs/artifacts).
  - Advanced metrics, curriculum options.
  - Plugin discovery via entry points for third-party backends.


## 20. Implementation Plan (Concrete Steps)

1. Scaffolding ✅
   - Set up server package with mypy/ruff/pytest and FastAPI skeleton.
   - Integrate platform_core logging (no wrappers, direct usage).
   - Implement DI container and registry patterns.
2. Contracts ✅
   - Define tokenizer/model/data/compute protocols and config models.
3. Tokenizer MVP ✅
   - HF Tokenizers BPE backend + orchestrator + API + tests.
4. Data Pipeline ✅
   - Corpus provider + dataset builder; split/holdout and tokenization.
5. Model MVP ✅
   - GPT‑2 backend via Transformers (CPU) + orchestrator + API + tests.
6. Discord Bot Integration ✅
   - Trainer cog with /train, /status, /cancel commands.
   - Event subscriber for training notifications via DM.
7. Hardening ✅
   - Type coverage review (no Any), test coverage for happy-path, run manifest.
   - Logging integration complete via platform_core.
   - 100% test coverage for statements and branches.


## 21. Type Safety Guarantees

- Python: mypy strict across all modules; no `typing.Any`, no `typing.cast`.
  - For third-party libs lacking types, wrap in thin, typed adapters or create `.pyi` stubs.
- TypeScript: `strict: true`, `noImplicitAny: true`, `noUncheckedIndexedAccess: true`, `exactOptionalPropertyTypes: true`.


## 22. Success Criteria

- Given a small cleaned corpus, user can:
  - Train a BPE tokenizer with a holdout and see coverage stats.
  - Train GPT‑2 or Char-LSTM on CPU or GPU with visible progress and saved checkpoints.
  - Use mixed-precision (FP16/BF16) on GPU for faster training.
  - View logs and metrics via Discord bot or API; export tokenizer/model artifacts and a manifest.
- All code passes type checks and unit tests; no silent exceptions; consistent structured logs.
- 100% test coverage for statements and branches.

## 23. Containerization (Docker & Compose)

Services (compose profiles `dev` and `prod`):
- api: FastAPI + Hypercorn, CPU-only PyTorch, Poetry-managed runtime. Exposes port 8000.
- worker (recommended for multi-day async): same image as api, runs job consumer for training/eval.
- redis (required): durable queue and pub/sub for jobs/log streaming. Image `redis:7-alpine`.

Base images:
- api/worker: `python:3.11-slim` with system deps; install via Poetry (no dev deps in prod).

Note: User interface is provided by the Discord bot (`clients/DiscordBot`), which runs as a separate service and connects to the API and Redis.

Volumes & caches:
- `./corpus:/data/corpus:ro`
- `./artifacts:/data/artifacts`
- `./runs:/data/runs`
- `./logs:/data/logs`
- Named volume `hf_cache:/hf-cache` with `HF_HOME=/hf-cache`

Env (examples):
- `APP_ENV=dev|prod`, `LOG_LEVEL=INFO`, `HF_HOME=/hf-cache`
- `REDIS_URL=redis://redis:6379/0` (dev/prod default) or Upstash URL for cloud
- `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `TOKENIZERS_PARALLELISM=1`
- Secrets via `.env` mounted (FastAPI loads with Pydantic Settings); never bake secrets into images.
 

Compose usage:
- `docker compose --profile dev up` → api + worker + redis
- `docker compose --profile prod up` → api + worker + redis

Resilience:
- Worker runs jobs out-of-process so API can restart without stopping training.
- Checkpoints and manifests on shared volume allow resume after restarts.

Windows note:
- Use Docker Desktop; bind mounts map project paths to containers. Commands are OS-agnostic via Compose.

## 24. Dependency & Tooling (Poetry, Ruff, Mypy)

Poetry:
- Use `pyproject.toml` for dependencies; commit `poetry.lock` for reproducibility.
- Dev dependencies: ruff, mypy, pytest, pre-commit.

Mypy (strict):
- Enforce no `Any` usage and no implicit optional.
- Representative config flags:
  - `disallow_any_generics = True`
  - `disallow_any_unimported = True`
  - `disallow_any_expr = True`
  - `disallow_any_decorated = True`
  - `disallow_any_explicit = True`
  - `no_implicit_optional = True`
  - `warn_redundant_casts = True`
  - `warn_unused_ignores = True`

Ruff:
- Enable `ANN401` to forbid `Any` in annotations.
- Ban `typing.cast` via banned-API rule to enforce “no casts”.
- Example (conceptual) `pyproject.toml` entries:

```
[tool.ruff]
target-version = "py311"

[tool.ruff.lint]
select = ["E","F","B","I","UP","ANN","PERF","S","TID","C90"]

[tool.ruff.lint.flake8-annotations]
allow-star-arg-any = false
mypy-init-return = true

[tool.ruff.lint.flake8-tidy-imports.banned-api]
"typing.cast" = { msg = "Do not use typing.cast; prefer precise types or adapters." }
```

Pre-commit:
- Hooks for ruff, mypy, pytest (fast subset); block merges on failures.

## 25. Config & Secrets (.env)

- Pydantic v2 Settings models for all configs (`AppConfig`, `LoggingConfig`, `TokenizerConfig`, `ModelConfig`, `TrainerConfig`, `DataConfig`, `ComputeConfig`).
- Sources: TOML files (checked-in defaults) + environment variables + `.env` for secrets.
- `.env.example` documents required keys (e.g., `HF_TOKEN` if needed for gated models).
- Secrets loaded only at runtime; do not commit `.env`.

## 26. Asynchronous & Long-Running Jobs

Goals:
- Training may run for days; must survive API/UI restarts and continue asynchronously.

Architecture:
- API enqueues jobs and immediately returns `run_id`.
- Worker consumes from Redis-backed queue (Redis required in all profiles).
- Queue implementation: RQ (Redis Queue) with `redis-py` client; our code depends on a typed JobQueue contract and a small adapter around RQ for strict typing.
- Worker writes logs to JSONL and periodic checkpoints; updates `manifest.json`.
- On startup, Worker reconciles and resumes incomplete runs from latest checkpoint.

Control & Observability:
- `GET /runs/:id` returns status and last heartbeat.
- `GET /runs/:id/logs` streams logs from the file (tail semantics).
- `POST /runs/:id/cancel` sets cancellation flag; worker checks between steps/batches.

Threading/Process model:
- Model training runs in the worker process (synchronous PyTorch loop).
- API remains responsive (async FastAPI) and performs non-blocking IO.

Notes on alternatives:
- Upstash vs redis-py: Upstash is a managed Redis service (deployment choice). We still use `redis-py` as the client. Set `REDIS_URL` to the Upstash URL (TLS) when using it.
- Celery: powerful but heavy for this MVP; adds brokers/results backends and more complexity. We can add a Celery adapter later if needed.
- arq/dramatiq: viable; arq is asyncio-native. We prefer RQ for simplicity and stability, wrapped behind our JobQueue contract for easy swapping.

Retry & failure semantics:
- Classify failures (`USER_INPUT`, `TRANSIENT`, `FATAL`).
- Default training policy: no auto-retry on `USER_INPUT`; bounded retries with backoff for `TRANSIENT`; no retry for `FATAL`.
- Persist failure classification and last exception in the run manifest; log with `error_code`.

Heartbeats & cancellation:
- Worker emits heartbeats to Redis key `runs:hb:<id>`.
- Status key: `runs:status:<id>`; cancellation flag: `runs:<id>:cancelled`.

## 27. Compose Profiles & Commands

- Profiles:
  - `dev`: api, worker, redis.
  - `prod`: api, worker, redis.
- Common commands:
  - `docker compose --profile dev up -d`
  - `docker compose --profile dev logs -f api worker`
  - `docker compose --profile prod up -d`

Note: Discord bot runs separately from `clients/DiscordBot`.

## 28. Borrowed Patterns (Swarm/DiscordBot)

Adopted inspirations to improve reliability and maintainability:
- Logging context (DiscordBot): attach `request_id` and `instance_id` to every log record; dev-friendly console logs + JSONL per run.
- Typed DI container (DiscordBot/Swarm): explicit ServiceContainer for config/services; no global singletons.
- Settings via Pydantic (Swarm): nested env with `env_nested_delimiter="__"`, `.env` support, strict validation.
- Health endpoints (Swarm-inspired): `/healthz` and `/readyz` only (no Prometheus exporter).
- Queue durability (beyond DiscordBot BRPOP): use RQ for ack/retry/visibility; avoid LPUSH/BRPOP loss on crash.
 
- Structured contracts and protocols (both): standard interfaces for queues, tokenizers, models, compute providers.

Non-goals from these repos for MVP:
- Celery workers, autoscalers, and full observability stack (Prometheus/Grafana/Loki) — can be added later.

## 29. Discord Bot Integration (Primary UI)

Purpose:
- User-facing interface for training control and status visibility through Discord.
- Implemented in `clients/DiscordBot` with typed event handling via `platform_core`.

Capabilities:
- Commands (respond in DMs only; ignore guild contexts):
  - `/trainer runs` — list recent runs with IDs and statuses.
  - `/trainer status <run_id>` — fetch current status, last checkpoint, ETA (if available).
  - `/trainer watch <run_id>` — enable push updates for this run to your DMs.
  - `/trainer unwatch <run_id>` — stop updates.
- Push alerts (rate-limited and batched to DM):
  - Run queued, started, checkpoint saved, eval completed, run completed, run failed.
  - Periodic heartbeat summaries (e.g., every 5–10 minutes) with step/epoch progress.

Architecture:
- Run worker publishes JSON events to Redis Pub/Sub channel `runs:<run_id>:events`.
- Discord bot subscribes to Pub/Sub only for runs you “watch”.
- Personal mode storage (single user):
  - `discord:watching` (set of run_ids) indicates which runs to deliver to DM.
  - On startup, the bot opens/validates a DM with the owner and restores Pub/Sub subscriptions for run_ids in `discord:watching`.
- Status queries call the API (`GET /runs/:id`) for source-of-truth state. No corpus data is ever sent to Discord.

Safety & limits:
- Owner-only DMs: require `DISCORD_OWNER_ID`; ignore all events outside the owner’s DM.
- Redact sensitive paths in messages; never display corpus contents or file listings.
- Apply a per-DM rate limit and batch updates to avoid spam and Discord rate limits.

Containerization:
- Optional `discord-status-bot` container using `python:3.11-slim` + `discord.py`.
- Env: `DISCORD_TOKEN`, `DISCORD_OWNER_ID`, `REDIS_URL`, `API_BASE_URL`.

Testing:
- Unit test the event formatter, rate limiter, and subscription storage logic (fake Redis client).
- Integration test mocks API responses and simulates Pub/Sub events.

## 30. Model Evaluation & Testing

Goals:
- Provide quantitative and qualitative checks to validate trained models on CPU.

Evaluation pipeline:
- Trigger: automatically after training, or via `POST /runs/:id/evaluate`.
- Data: use the run’s validation split or a user-provided eval split/path.
- Metrics:
  - Validation loss and perplexity (causal LM cross-entropy).
  - Tokenization stats: average tokens per char/word, sequence length distribution.
  - Optional distinct-n and repetition indicators on generated samples.
- Qualitative: deterministic sample generations from a fixed prompt set (seeded), saved to `samples.jsonl`.

Artifacts:
- `artifacts/models/<run_id>/eval/metrics.json`
- `artifacts/models/<run_id>/eval/samples.jsonl`
- Update run `manifest.json` with eval summary (loss, ppl, dataset info, seed).

Acceptance guidance (CPU‑first, tiny models):
- Sanity target: validation perplexity decreases vs. the first epoch.
- For regression baselines, compare to a previous run’s metrics; flag regressions in the UI.

API:
- `POST /runs/:id/evaluate` — request eval on a specific split or external path.
- `GET /runs/:id/eval` — return latest evaluation summary and artifact pointers.

Testing strategy:
- Tokenizers: encode/decode roundtrip on toy samples; coverage stats on a small corpus.
- Training loop: run a tiny training step on a toy corpus; assert loss decreases between first two evals.
- Evaluation: compute perplexity on a known toy dataset; verify deterministic results with fixed seeds.
- Orchestrators: happy-path and error injection verifying error codes and no silent failures.

## 30.1 Inference (Score & Generate)

Goals:
- Enable users to query trained models for text generation and scoring without retraining.
- Provide async job execution matching training/evaluation patterns.

### Inference Orchestrator

Located at `orchestrators/inference_orchestrator.py`, handles:
- `enqueue_score()` — queue a scoring job, return `request_id` immediately.
- `get_score()` — poll for scoring results by `request_id`.
- `enqueue_generate()` — queue a generation job, return `request_id` immediately.
- `get_generate()` — poll for generation results by `request_id`.

### Text Generation

API:
- `POST /runs/{run_id}/generate` — enqueue generation job.
- `GET /runs/{run_id}/generate/{request_id}` — poll for results.

Request fields:
- `prompt_text` / `prompt_path`: input prompt (mutually exclusive).
- `max_new_tokens`: maximum tokens to generate (1–512).
- `temperature`: sampling temperature (0 = greedy).
- `top_k`, `top_p`: filtering parameters.
- `stop_on_eos`, `stop_sequences`: termination conditions.
- `seed`: reproducibility.
- `num_return_sequences`: batch generation (1–10).

Response fields:
- `request_id`, `status`: job tracking.
- `outputs`: list of generated text sequences.
- `steps`: tokens generated per sequence.
- `eos_terminated`: whether each sequence hit EOS.

Implementation:
- GPT-2: `core/services/model/backends/gpt2/generate.py` uses HuggingFace `model.generate()`.
- Char-LSTM: `core/services/model/backends/char_lstm/generate.py` uses stateful hidden-state passing.

### Text Scoring

API:
- `POST /runs/{run_id}/score` — enqueue scoring job.
- `GET /runs/{run_id}/score/{request_id}` — poll for results.

Request fields:
- `text` / `path`: input text (mutually exclusive).
- `detail_level`: `summary` or `per_char`.
- `top_k`: return top-K predictions per position.
- `seed`: reproducibility.

Response fields:
- `request_id`, `status`: job tracking.
- `loss`, `perplexity`: aggregate metrics.
- `surprisal`: per-token log-probabilities.
- `topk`: top-K token predictions per position.
- `tokens`: tokenized input.

### Worker Jobs

- `worker/generate_job.py`: downloads model from data-bank, runs generation, caches result.
- `worker/score_job.py`: downloads model, computes loss/perplexity/surprisal, caches result.

Results are cached in Redis at `generate:{run_id}:{request_id}` and `score:{run_id}:{request_id}`.

## 31. Security & Correlation (Implemented)

- Request correlation: middleware sets and echoes `X-Request-ID` for every response.
- Error bodies: all errors include `code`, `message`, and `request_id` (see `core/errors/handlers.py`).
- API key auth: if configured, requests must include `X-Api-Key`; invalid/missing key returns HTTP 401 with code `UNAUTHORIZED`.

## 32. Training Events & Discord Integration (Implemented)

- Worker publishes lifecycle via `platform_core.job_events` to `trainer:events`:
  - `trainer.job.started.v1`, `trainer.job.progress.v1`, `trainer.job.completed.v1`, `trainer.job.failed.v1`
- Worker publishes training metrics via `platform_core.trainer_metrics_events` to the same channel:
  - `trainer.metrics.config.v1`, `trainer.metrics.progress.v1`, `trainer.metrics.completed.v1`
- `job_id` equals `run_id` for correlation; `user_id` is included for Discord DM routing. Progress events may include an optional payload; metrics events carry epochs/steps/loss/perplexity/artifact details.
- DiscordBot subscribes and decodes both job and metrics events to render embeds.

## 33. Run Status Messages (Implemented)

- Status keys: `runs:status:<run_id>` and heartbeats `runs:hb:<run_id>`.
- A human-readable message is stored at `runs:msg:<run_id>` on cancellation/failure/completion and exposed via `GET /runs/{id}` as `message`.

## 34. Railway Deployment Notes

- Services: API (hypercorn) and worker (rq), plus Redis addon.
- Ensure start commands execute in the service root (set working directory appropriately).
- Required env: `REDIS_URL`, `SECURITY__API_KEY` (if enforcing auth), and artifacts/log roots.
