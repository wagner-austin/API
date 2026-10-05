---
title: Covenant Radar's three processes — API, RQ worker, streaming worker — and the one container that wires them
tags: [services, covenant-radar, architecture, rq, fastapi, dependency-injection]
related:
  - "[[covenant-radar-backend-registry]]"
  - "[[covenant-radar-domain-protocol]]"
  - "[[covenant-radar-kafka-streaming]]"
  - "[[covenant-radar-optuna-optimisation]]"
  - "[[platform-workers-rq-pattern]]"
  - "[[service-port-map]]"
source_paths:
  - services/covenant-radar-api/src/covenant_radar_api/api/main.py
  - services/covenant-radar-api/src/covenant_radar_api/core/container.py
  - services/covenant-radar-api/src/covenant_radar_api/core/model_paths.py
  - services/covenant-radar-api/src/covenant_radar_api/worker_entry.py
  - services/covenant-radar-api/src/covenant_radar_api/api/routes/ml.py
  - services/covenant-radar-api/src/covenant_radar_api/api/routes/ml_analysis.py
  - services/covenant-radar-api/src/covenant_radar_api/api/routes/evaluate.py
  - services/covenant-radar-api/src/covenant_radar_api/worker/evaluate_job.py
  - libs/covenant_domain/src/covenant_domain/rules.py
  - libs/platform_core/src/platform_core/config/covenant_radar.py
  - services/covenant-radar-api/pyproject.toml
  - services/covenant-radar-api/README.md
source_git_blobs:
  "services/covenant-radar-api/src/covenant_radar_api/api/main.py": b9b2dfc969687cc4bb6a578279cf5c017af6fd65
  "services/covenant-radar-api/src/covenant_radar_api/core/container.py": ed8fa72fdb1b5a86b649a6ca2be8ba7896353332
  "services/covenant-radar-api/src/covenant_radar_api/core/model_paths.py": 323fb99967f7affadda00519ef3df6b2d9a76dd3
  "services/covenant-radar-api/src/covenant_radar_api/worker_entry.py": b5efe0cf5061309837c9f991380e8ed2487eb168
  "services/covenant-radar-api/src/covenant_radar_api/api/routes/ml.py": 037a5368517af92277438abe3e211ae5ba8a63aa
  "services/covenant-radar-api/src/covenant_radar_api/api/routes/ml_analysis.py": 8279fa6db5c84444f8d73639e0cd0fdde83f74d5
  "services/covenant-radar-api/src/covenant_radar_api/api/routes/evaluate.py": 091c69d1d73c13b59c04101de8b8c7d82ce514e4
  "services/covenant-radar-api/src/covenant_radar_api/worker/evaluate_job.py": 09cfcc5396bdedfc4b50131f54774655772e049c
  "libs/covenant_domain/src/covenant_domain/rules.py": 9420c46c2e125feb2834be050292dc1e0fa938fa
  "libs/platform_core/src/platform_core/config/covenant_radar.py": 7846daf5cc69c4b2609c9f07e403dd8c6411b2ae
  "services/covenant-radar-api/pyproject.toml": 8f420f4aa5698d2bb311e5dba284bb81acbe72ca
  "services/covenant-radar-api/README.md": 34f48ed308cbe2cbffc9e867e2b4a41748c9a90b
provenance:
  - "every claim read from code at API commit 2e0511ee7 on 2026-10-05; no process was started to observe it"
  - "'nothing enqueues run_batch_evaluation' is a grep over services/ and libs/ for the symbol outside tests/, which found only its definition and __all__"
fact_checked: "2026-10-05"
confidence: high
hubs: [services]
---

# Covenant Radar's three processes

The service is three processes over one codebase, and a session extending it
needs to know which one a change lands in:[^scripts][^app]

| process | entry | what it does |
|---|---|---|
| API | `api/main.py::create_app` | FastAPI; CRUD, rule evaluation, synchronous prediction, and enqueueing |
| RQ worker | `covenant-rq-worker` → `worker_entry:main` | training, Optuna search, explanation jobs |
| streaming worker | `covenant-streaming-worker` → `generic_worker_entry:main` | Kafka inference — see [[covenant-radar-kafka-streaming]] |

The two console scripts are the whole of `[tool.poetry.scripts]`.[^scripts]

## One container, built once, held for the app's lifetime

`create_app` sets up Datadog tracing first (ddtrace must instrument before
anything else imports), then JSON logging, then builds a single
`ServiceContainer.from_settings(cfg, eager_load_model=True)` and holds it open
in the lifespan.[^app] Every router is built by `build_router(container)`, so
routes receive their dependencies rather than reaching for globals.[^app]

`from_settings` constructs Redis, PostgreSQL and the RQ client through
`_test_hooks` factories, runs `ensure_schema` on every start, and picks the
active model by backend.[^container] The RQ worker is thinner: it reads
`REDIS_URL`, binds `COVENANT_QUEUE` and the covenant events channel, and hands
that to the shared harness — the pattern in [[platform-workers-rq-pattern]].[^worker]

**The active-model path has two sources for four backends.**
`settings["app"]["ml_backend"]` is one of `xgboost`, `mlp`, `lstm`,
`lightgbm`;[^mlb] the container takes `active_model_path_xgb` when it is
`xgboost` and `active_model_path_mlp` for **all three others** — so an `lstm`
or `lightgbm` deployment reads its artifact from the setting named for MLP.[^container]
The README states this as "derived from `APP__ML_BACKEND` plus
`APP__ACTIVE_MODEL_PATH_XGB` / `_MLP`", which is accurate and easy to misread
as one path per backend.[^readme] If the model is absent locally and data-bank
is configured, both `load_model_now` and `get_model` try a download into
`/tmp/models` first.[^container]

## Rules are exact; risk is learned — and they never share a code path

`POST /evaluate` runs `covenant_domain.evaluate_all_covenants_for_period`
synchronously and persists the results.[^eval] The status is integer
arithmetic on values scaled by 1,000,000: for a `<=` covenant, `BREACH` above
the threshold, `NEAR_BREACH` above `threshold − threshold·tolerance`, else
`OK`; the `>=` case mirrors it.[^rules] Nothing in that path is probabilistic,
which is the point the README makes about keeping compliance and breach risk
apart.[^readme]

`worker/evaluate_job.py::run_batch_evaluation` is a batch form of the same
evaluation written for RQ — but **no route or script enqueues it**; its only
callers are tests.[^batch] Wiring a batch endpoint means adding the enqueue,
not writing the job.

## Every ML endpoint except two predicts is a queued job

`/ml/train`, `/ml/train-external`, `/ml/train-external-regression`,
`/ml/optimize`, `/ml/optimize-regression`, `/ml/explain` and
`/ml/explain-regression` all `queue.enqueue` a dotted path under
`covenant_radar_api.worker.*`; status is read back from `/ml/jobs/{job_id}`.[^enq]
`/ml/predict` and `/ml/predict-regression` run inline.[^enq] The optimise
jobs are the subject of [[covenant-radar-optuna-optimisation]]; which
backends each job accepts is [[covenant-radar-backend-registry]].

## Request-supplied model paths are confined

`model_path` arrives on explain and regression-predict bodies and flows into
`torch.load` / `joblib.load`, which unpickle. `resolve_model_path` resolves
symlinks and `..` before checking `is_relative_to(models_root)` and raises
`ValueError` otherwise, so a caller cannot choose which file on the host a
loader opens.[^paths]

[^scripts]: `services/covenant-radar-api/pyproject.toml:68-70` — `[tool.poetry.scripts]`, `covenant-rq-worker` and `covenant-streaming-worker`.
[^app]: `services/covenant-radar-api/src/covenant_radar_api/api/main.py` / `create_app` — tracing, `setup_logging`, `ServiceContainer.from_settings(cfg, eager_load_model=True)`, eight `include_router(... build_router(container))` calls.
[^container]: `services/covenant-radar-api/src/covenant_radar_api/core/container.py` / `ServiceContainer.from_settings` (the `backend_model_path` conditional on `MLBackend.XGBOOST`), `load_model_now`, `get_model`.
[^worker]: `services/covenant-radar-api/src/covenant_radar_api/worker_entry.py` / `_build_config` — `REDIS_URL`, `COVENANT_QUEUE`, `default_events_channel(JobDomain.COVENANT)`.
[^mlb]: `libs/platform_core/src/platform_core/config/covenant_radar.py` / `MLBackend` — four members.
[^readme]: `services/covenant-radar-api/README.md` § "Configuration" (the paragraph under the table) and § "The design decisions worth knowing" (rules vs models).
[^eval]: `services/covenant-radar-api/src/covenant_radar_api/api/routes/evaluate.py` / `build_router` → `_evaluate`.
[^rules]: `libs/covenant_domain/src/covenant_domain/rules.py` / `classify_status`.
[^batch]: `services/covenant-radar-api/src/covenant_radar_api/worker/evaluate_job.py` / `run_batch_evaluation` [synthesis] — see provenance for the search that established it has no caller.
[^enq]: `services/covenant-radar-api/src/covenant_radar_api/api/routes/ml.py` / `_register_train`, `_register_train_external`, `_register_train_external_regression`, `_register_predict`; `services/covenant-radar-api/src/covenant_radar_api/api/routes/ml_analysis.py` / `_register_optimize`, `_register_optimize_regression`, `_register_explain`, `_register_explain_regression`, `_register_predict_regression`, `_register_job_status`.
[^paths]: `services/covenant-radar-api/src/covenant_radar_api/core/model_paths.py` / `resolve_model_path`.
