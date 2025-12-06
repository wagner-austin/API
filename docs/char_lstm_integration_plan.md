# Character‑Level LSTM Integration Plan

Status: Updated (adds Inference API)
Owners: Model‑Trainer team (Austin), with mentors Connor Mayer, Richard Futrell
Last updated: 2025‑12‑01

## Goals

- Add a first‑class character‑level training path to Model‑Trainer that operates on NFC‑normalized IPA text and supports cross‑language evaluation for mutual intelligibility experiments.
- Keep architecture modular and registry‑driven: new tokenizer backend (`char`) and new model backend (`char_lstm`) without reworking existing orchestrators or job lifecycle.
- Enforce strict typing and repository guardrails (no `Any`, no `cast`, no `type: ignore`, no stubs, no dataclasses in `src/`).
- Reach 100% statements and branch coverage for all new code and touched branches in existing code.
- Provide a first‑class, unified inference surface (score/generate) that works for all model families (GPT‑2 and `char_lstm`) with strict schemas, validators, and queue workers.

Non‑goals:

- No GPU features or CUDA branches in this iteration (CPU‑first).
- No back‑compat shims or best‑effort fallbacks; errors are explicit and propagate.

## Inputs and Preprocessing

- Source corpora: OSCAR or CulturaX via `turkic-api` streamers, filtered by FastText LID.
- Transliteration: `to_ipa(text, lang)` from `services/turkic-api/src/turkic_api/core/translit.py` using language‑specific `*.rules` files. Rules explicitly enforce NFC (see `*_ipa.rules` files containing `NFC`).
- Balanced corpora: use `scripts/build_balanced_corpora.py` to produce per‑language NFC IPA text files with equal IPA‑character budget.

Example command (PowerShell):

```
$env:HUGGINGFACE_HUB_TOKEN=$env:HF_TOKEN; poetry run python scripts/build_balanced_corpora.py --source oscar --threshold 0.90 --langs kk,ky,tr,az,ug,uz --out-dir data/balanced/oscar --progress-every 500 --log-format text
```

Outputs: `data/balanced/oscar/oscar_<lang>_ipa.txt` for `kk, ky, tr, az, ug, uz`.

Normalization policy:

- Keep stress marks, length (ː), and IPA modifiers per rules; do not strip.
- NFC throughout the pipeline; no additional normalization step is needed upstream of Model‑Trainer.

## Tokenizer: `char` backend (shared across 6 languages)

Design:

- Build a vocabulary of unique characters from the union of the six IPA corpora plus specials: `[PAD]=0, [UNK]=1, [BOS]=2, [EOS]=3`.
- Encode: one codepoint → one token id; unknown characters map to `[UNK]`.
- Save artifacts under `artifacts_root/tokenizers/<tokenizer_id>/`:
  - `tokenizer.json` (typed schema: specials, vocab mapping `char -> id`, and a small header identifying type `"char"`).
  - `manifest.json` with training stats (coverage, OOV rate, token_count, char_coverage, holdout_fraction, seed).

Contracts:

- Implement `TokenizerBackend` and `TokenizerHandle` without `Any`.
- Training config uses existing `TokenizerTrainConfig` typed class; `vocab_size` and `min_frequency` are accepted but do not downsample characters by frequency unless specified (default: include all seen chars ≥1). Stats must be computed deterministically using helpers from `core/services/data/corpus.py`.

Planned files:

- `services/Model-Trainer/src/model_trainer/core/services/tokenizer/char_backend.py`

Registry:

- Register under `"char"` in `TokenizerRegistry` alongside `bpe` and optional `sentencepiece`.
- Extend worker loader `_load_tokenizer_for_training` to detect/load char artifacts by the `tokenizer.json` header or manifest marker.

## Model: `char_lstm` backend

Model:

- Unidirectional LSTM language model over character embeddings.
- Architecture: Embedding → LSTM (layers=2) → Linear projection to vocab; output head weight‑tied to embedding.
- Teacher forcing with cross‑entropy loss; next‑char prediction evaluates loss and perplexity.
- Paper‑aligned defaults (CPU‑friendly): `embed=128`, `hidden=256`, `layers=2`, `dropout=0.1`, gradient clip `1.0`.
- Sequence length (`max_seq_len`): 256 by default; tradeoff explained below.

Training & evaluation:

- Use existing `CausalLMDataset` and lightweight `DataLoader` (no padding/colate changes needed).
- Reset hidden state per batch for robustness and simplicity.
- Write `manifest.json` aligned to current format with `model_family = "char_lstm"`.
- Evaluate on validation split using shared tokenizer; write `metrics.json` with `loss` and `perplexity`.

Contracts & layout:

- Follow GPT‑2 backend structure for modularity and tests:
  - `core/services/model/backends/char_lstm/config.py` (TypedDict `CharLSTMTrainConfig`).
  - `core/services/model/backends/char_lstm/model.py` (PyTorch module implementing `LMModelProto`).
  - `core/services/model/backends/char_lstm/io.py` (save/load, token id helpers, encoder adapter).
  - `core/services/model/backends/char_lstm/prepare.py` (construct `PreparedModel` with tokenizer handle -> internal encoder).
  - `core/services/model/backends/char_lstm/train.py` (training loop, manifest writing).
  - `core/services/model/backends/char_lstm/evaluate.py` (validation loop, metrics writeout).
  - `core/services/model/char_lstm_backend_impl.py` (adapter to `ModelBackend` Protocol, mirrors GPT‑2 impl file).

Registry:

- Add to `ModelRegistry` under key `"char_lstm"` in `core/services/container.py`.

## API & Worker integration

- API schema: widen `model_family` literal to include `"char_lstm"` in `services/Model-Trainer/src/model_trainer/api/schemas/runs.py`.
- Model contracts: widen `ModelTrainConfig.model_family` literal to include `"char_lstm"` in `core/contracts/model.py`.
- Tokenizer API: widen `TokenizerTrainRequest.method` to include `"char"` in `api/schemas/tokenizers.py`.
- Worker: extend `_load_tokenizer_for_training` to support char artifacts in `worker/training_worker.py`.
- Evaluation worker: replace the hardcoded GPT‑2 backend selection with family‑based lookup (`container2.model_registry.get(cfg.model_family)`), and honor `path_override` by substituting `cfg.corpus_path` prior to backend evaluation.

### Inference API (Unified for all models)

Endpoints:

- POST `/runs/{run_id}/score` → enqueue scoring job
- GET `/runs/{run_id}/score/{request_id}` → fetch score result
- POST `/runs/{run_id}/generate` → enqueue generation job
- GET `/runs/{run_id}/generate/{request_id}` → fetch generation result

Schemas (strict TypedDict):

- ScoreRequest: `text: str | None`, `path: str | None` (mutually exclusive),
  `detail_level: Literal["summary","per_char"] = "summary"`, `top_k: int | None`, `seed: int | None`.
- ScoreResponse: `request_id: str`, `loss: float`, `perplexity: float`,
  `surprisal: list[float] | None`, `topk: list[list[tuple[str,float]]] | None`, `tokens: list[str] | None`.
- GenerateRequest: `prompt_text | prompt_path` (mutually exclusive),
  `max_new_tokens: int`, `temperature: float`, `top_k: int`, `top_p: float`,
  `stop_on_eos: bool`, `stop_sequences: list[str]`, `seed: int | None`,
  `num_return_sequences: int`.
- GenerateResponse: `request_id: str`, `outputs: list[str]`,
  `meta: { steps: int, eos_terminated: bool | list[bool] }`.

Orchestrator & Queue:

- Add `InferenceOrchestrator` with `enqueue_score`, `get_score`, `enqueue_generate`, `get_generate`.
- Add `ScoreJobPayload` and `GenerateJobPayload` to `core/contracts/queue.py`.
- Add `enqueue_score` and `enqueue_generate` to RQ enqueuer.
- Workers: `process_score_job`, `process_generate_job` in `worker/training_worker.py`, using `container.model_registry.get(cfg.model_family)` and the trained tokenizer.

Redis keys:

- Score cache: `runs:score:<run_id>:<request_id>`
- Generate cache: `runs:gen:<run_id>:<request_id>`

## Shared Libs Alignment

- platform_core
  - Keys: continue to use `heartbeat_key`, `eval_key`, `artifact_file_id_key`, `cancel_key`. Add first‑class inference keys (score/gen) in platform_core to avoid divergence across services.
  - JSON: use `platform_core.json_utils` for all request/response/cache payloads.
  - Errors/Logging: use `platform_core.errors.AppError` and centralized logging helpers with strict typed levels and formats.
- platform_workers
  - Redis: use `RedisStrProto` and `redis_for_kv` for KV and pub/sub; no generic Redis[Any].
  - RQ: use `rq_harness` (`rq_queue`, `rq_retry`) and `RQClientQueue` for enqueue; preserve `TRAINER_QUEUE` alignment.
  - Job context: create `JobContext` via `make_job_context` for score/generate jobs, and publish lifecycle/metrics events under a dedicated domain if needed (e.g., `inference`).
- platform_ml
  - Artifacts: continue to use `ArtifactStore` for download/materialization during evaluation and any load‑on‑demand flows; no service‑local download logic.
- Contracts
  - Queue: strictly typed `ScoreJobPayload` and `GenerateJobPayload` in Model‑Trainer; payloads serialized in enqueuer via `JSONValue` mapper.
  - Validators: strict, service‑edge only; no `Any` across boundaries.

## Discord Usage

- Goal: allow end‑users to interact with trained models (GPT‑2 today, `char_lstm` next) via Discord with no service‑specific hacks.
- Flow (clients/DiscordBot):
  - Configure `MODEL_TRAINER_API_URL` and API key.
  - Expose slash commands:
    - `/mt runs` to list or select a `run_id`.
    - `/mt generate <text> [options]` → POST `/runs/{run_id}/generate` with `prompt_text`; poll GET until done; return generated text.
    - `/mt score <text>` → POST `/runs/{run_id}/score` with `text`; poll GET; return loss/perplexity (and optionally per‑token/char stats to admins).
  - Security: pass API key header; apply rate limits client‑side; cap `max_new_tokens`, `num_return_sequences` for CPU.
- Behavior per model:
  - GPT‑2: token‑level sampling/score via trained BPE tokenizer.
  - char‑LSTM: character‑level sampling/score via shared `char` tokenizer; identical API surface.

## Sequence Length (why it matters)

- Definition: number of characters per sample (a fixed chunk length). Larger values offer longer context for phonotactic/morphological cues but increase CPU time per step and can make optimization harder (longer BPTT).
- Tradeoff: keep `batch_size * seq_len` roughly constant for similar throughput. Recommended default: `max_seq_len = 256`, `batch_size = 64` on CPU.

## Typing, Imports, and Error policy

- Strict typing only: no `Any`, no `cast`, no `type: ignore`, no stubs, no dataclasses in `src/`.
- Use `TypedDict`, `Protocol`, and precise type aliases; validate JSON/TOML via `_decode*` helpers.
- Dynamic imports: use `__import__`, then `getattr`, assigning to a `Protocol`‑typed variable at the point of use to avoid `Any`.
- Error handling: detect and raise clear exceptions; do not soften or recover in core logic; no best‑effort fallbacks.
- Security for file inputs: validators restrict `path`/`prompt_path` to resolve under `settings["app"]["data_root"]` (or a dedicated whitelist). Paths outside are rejected with `INVALID_INPUT`.

## Training regimen (paper‑aligned)

- Phase 1 (pretrain): train `char_lstm` on Kazakh for 5 epochs with early stopping.
- Phase 2 (bilingual continuation): warm‑start from Phase 1, freeze embeddings, train on Kyrgyz and Turkish for 3 epochs with early stopping.
- Advanced train config (NotRequired in TrainRequest):
  - `init_from_run_id: str | None` (warm start)
  - `freeze_embeddings: bool`
  - `early_stopping: { enabled: bool, patience: int, min_delta: float }`

Metrics & outputs:

- Write step/epoch training loss (nats/char) JSONL; compute AUC over the loss curve per language and save summary.
- Eval writes `metrics.json` with `loss` and `perplexity`.

## Tests (100% coverage target)

Tokenizer (`char`):

- Encode/decode round‑trip for IPA samples (with stress and length marks).
- OOV behavior; stats computation on holdout using `sample_lines`.
- Manifest write/read; `inspect()` returns typed `TokenizerTrainStats`.

Model (`char_lstm`):

- Prepare path builds model with correct vocab and special ids.
- Train loop: one tiny corpus epoch runs, loss computed; heartbeat/progress callbacks invoked; cancellation branch covered.
- Save/load round‑trip; forward shapes and loss are stable on a fixed batch.
- Evaluate path writes `metrics.json` and returns finite loss/ppl.

Inference:

- Validators fully cover `text` vs `path`, summary vs per_char, and bounds for `top_k`, `max_new_tokens`, `num_return_sequences`.
- Score worker: summary and per_char branches (including optional top‑k) for GPT‑2 and `char_lstm`.
- Generate worker: sampling with temp/top‑k/top‑p and EOS/stop sequences; deterministic with fixed seed.

Integration:

- Registry includes `char_lstm` and `char`; training orchestrator accepts new family (registry validation passes).
- Worker eval uses family‑based selection; add a regression test targeting the modified line.

Coverage gates:

- Extend `tests/test_coverage_gaps.py` if needed to cover any remaining branches introduced by this work.

## Rollout Plan

1) Eval fixes: family‑based backend selection; honor `path_override`.
2) Inference API: schemas, validators, orchestrator, enqueuer, workers, and tests.
3) Implement `char` tokenizer backend and unit tests.
4) Implement `char_lstm` backend and unit tests (paper‑aligned defaults).
5) Wire registries, widen literals, add integration tests.
6) Documentation: update this plan and `services/Model-Trainer/README.md` with usage.
7) Run `make check` in `services/Model-Trainer` until green (lint + mypy strict + tests, branch coverage 100).

## Acceptance Criteria

- `make check` passes in `services/Model-Trainer` and no regressions in other packages.
- 100% statements and branch coverage for all new and modified code paths.
- End‑to‑end: a training job with `model_family="char_lstm"` runs on a tiny corpus and produces artifacts; eval job writes metrics.
- Inference endpoints (`/score`, `/generate`) enqueue and return results for GPT‑2 and `char_lstm` with strict validation.
- No `Any`, `cast`, `type: ignore`, stubs, or dataclasses introduced.

## Usage Notes (after implementation)

1) Build balanced IPA corpora for `kk,ky,tr,az,ug,uz` via `turkic-api` (see command above).
2) Train shared `char` tokenizer on the union corpus via Model‑Trainer tokenizer API/worker; record `tokenizer_id`.
3) Start a training run on Kazakh with `model_family="char_lstm"`, `tokenizer_id=<shared>`, `max_seq_len=256`, suitable batch/epochs.
4) Evaluate on held‑out sets for all six languages; compare perplexities.
