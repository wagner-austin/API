Deploying to Railway (Docker)

Overview
- This repo is containerized via a single-stage `Dockerfile` (FastAPI + RQ).
- We deploy two services from the same image with different start commands.
- Redis is provided by Railway's Redis plugin; artifacts and logs persist on a mounted volume at `/data`.

Prereqs
- Railway project created and connected to this GitHub repo.
- Redis plugin added to the project.

Service: API (model-trainer-api)
- Builder: Dockerfile
- Dockerfile Path: `Dockerfile`
- Start Command: leave empty (Dockerfile CMD runs Hypercorn and respects `$PORT`)
- Exposed Port: Use platform default (image honors `$PORT`, defaults 8000)
- Healthcheck Path: `/readyz`
- Environment Variables:
  - `REDIS__URL` = Use Redis plugin provided URL
  - `APP__ARTIFACTS_ROOT` = `/data/artifacts`
  - `APP__RUNS_ROOT` = `/data/runs`
  - `APP__LOGS_ROOT` = `/data/logs`
  - `HF_HOME` = `/hf-cache` (optional for tokenizer/model cache)
  - Optional: `SECURITY__API_KEY` = `<your-secret>` (send `X-API-Key` header on requests)
- Persistent Volumes:
  - Mount a Railway Volume at `/data`
  - Optionally mount another at `/hf-cache`

Service: Worker (model-trainer-worker)
- Builder: Dockerfile
- Dockerfile Path: `Dockerfile`
- Start Command: `/app/.venv/bin/modeltrainer-rq-worker`
- Environment + Volumes: same as API (especially `REDIS__URL` and `/data` mount)

 Notes on Reliability and Drift
- Single source of truth for container logic is `Dockerfile`.
- API binds to `$PORT` with Hypercorn (see `Dockerfile` default CMD) to work on Railway and similar platforms.
- Strict typing and guard rails are enforced in CI/local via:
  - `mypy --strict`, `ruff` and `scripts/guard.py` (no `Any`, no casts, no `type: ignore`).
  - Run `make check` locally to reproduce CI checks.
- Tests (`tests`) provide broad unit/integration coverage with `--cov-branch`.

Local Validation
- `copy .env.example .env`
- `make start` to run Redis, API, and Worker locally via Docker Compose.
- Health: `GET http://localhost:8000/readyz`
- Run checks: `make check` (ruff + mypy + guard + pytest with branch coverage)

API Quick Reference
- Health: `GET /healthz`, `GET /readyz`
- Tokenizers: `POST /tokenizers/train`, `GET /tokenizers/{id}`
- Training Runs: `POST /runs/train`, `GET /runs/{id}`, `POST /runs/{id}/evaluate`, `GET /runs/{id}/eval`
- Artifacts: `GET /runs/{id}/artifact`
- Logs: `GET /runs/{id}/logs`, `GET /runs/{id}/logs/stream`
- Cancellation: `POST /runs/{id}/cancel`
- Inference: `POST /runs/{id}/generate`, `GET /runs/{id}/generate/{request_id}`, `POST /runs/{id}/score`, `GET /runs/{id}/score/{request_id}`
- Chat: `POST /runs/{id}/chat`, `GET /runs/{id}/chat/{session_id}/{request_id}`, `GET /runs/{id}/chat/{session_id}`, `DELETE /runs/{id}/chat/{session_id}`

Security
- If `SECURITY__API_KEY` is set, all endpoints require header: `X-API-Key: <value>`.
