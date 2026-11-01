# API Services

The FastAPI services of this monorepo. Each one is its own Poetry project with
strict typing, 100% statement and branch coverage, and a `make check` that the
fleet runs on every push that touches it.

## Services

| Service | Port | Purpose | Shared libraries |
|---------|------|---------|------------------|
| [covenant-radar-api](./covenant-radar-api/) | 8007 | Multi-domain risk prediction with pluggable ML backends, Optuna optimization and Kafka streaming | platform_core, platform_workers, covenant_domain, covenant_ml, covenant_nn, covenant_persistence, cleargbm_rs |
| [github-stats-api](./github-stats-api/) | 8009 | GitHub stats SVG card generation | platform_core, platform_codebase, platform_kaggle |
| [grandma-api](./grandma-api/) | 8008 | Multi-language audio-to-English translation | platform_core, platform_stt, platform_langid, platform_translate |
| [Model-Trainer](./Model-Trainer/) | 8005 | Language model and tokenizer training, evaluation and inference | platform_core, platform_ml, platform_workers |
| [opportunity-radar-api](./opportunity-radar-api/) | 8010 | Hackathon and competition discovery | platform_core, platform_calendar, platform_codebase, platform_devpost, platform_kaggle |
| [transcript-api](./transcript-api/) | 8003 | YouTube captions and speech-to-text | platform_core, platform_stt, platform_workers |

The ports are the host ports the root `docker-compose.yml` header maps. The
root `Makefile` starts each service with its own target (`up-covenant`,
`up-github-stats`, `up-grandma`, `up-trainer`, `up-opportunity`,
`up-transcript`), each of which runs `make infra` first.

Model-Trainer keeps its corpora and stored model artifacts on the filesystem:
a corpus is staged under `/data/corpus` by its SHA-256 and a finished run is
packed into `/data/artifacts/store`. Until 2026-11-01 both went through
data-bank-api, which was archived that day with the services nothing used any
more (Art-Trainer, handwriting-ai, turkic-api, qr-api, music-wrapped-api,
procart-api); they live on only at the tag `archive/api-packages-2026-11-01`.

## Working on a service

Every service Makefile has the same three targets:

```bash
make lint    # guards, ruff and mypy strict
make test    # pytest with 100% statement and branch coverage
make check   # lint, then test
```

`services/Makefile` fans each target out over every service beneath it and
names the ones that failed at the end.

## Shared rules

- Logging, errors, configuration and health checks come from
  [platform_core](../libs/platform_core/); a service does not read the
  environment or configure stdlib logging itself.
- Data crossing a boundary is a TypedDict with an explicit decoder.
- Every external effect sits behind a `_test_hooks` seam so tests drive the
  real code.
- `scripts/guard.py` in each service runs the
  [monorepo_guards](../libs/monorepo_guards/) rules over its tree.

Error responses share one shape, `{code, message, request_id}`, and every
service answers `GET /healthz` and `GET /readyz`.
