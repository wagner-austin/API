# music-wrapped-api

Typed FastAPI service for Music Wrapped. Follows platform patterns:
- Strict types, no Any/cast/type: ignore
- Protocols at boundaries, TypedDict models
- Redis + RQ via platform_workers
- Health endpoints and worker entrypoint

## Environment

- Required variables (loaded with `platform_core.config._require_env_str`):
  - `REDIS_URL`
  - `LASTFM_API_KEY`, `LASTFM_API_SECRET`
  - `SPOTIFY_CLIENT_ID`, `SPOTIFY_CLIENT_SECRET`
  - `APPLE_DEVELOPER_TOKEN`
  - Optional: `PORT` (defaults to 8000 inside container)

Copy `.env.example` to `.env` and fill in values:

```
cp .env.example .env
```

## Docker

- Build multi-stage image and run API:
  - `docker compose up -d` from `services/music-wrapped-api`
  - Creates `api` and `worker` services; uses `.env` for configuration
- Network: compose expects `platform-network` to exist (shared across services):
  - `docker network create platform-network`
- Redis: set `REDIS_URL=redis://platform-redis:6379/0` and run a Redis container on the network, or point to an existing instance.

## Endpoints

- `GET /healthz`, `GET /readyz`
- `POST /v1/wrapped/auth/youtube/store`
- `POST /v1/wrapped/auth/apple/store`
- `GET /v1/wrapped/auth/lastfm/start`, `GET /v1/wrapped/auth/lastfm/callback`
- `GET /v1/wrapped/auth/spotify/start`, `GET /v1/wrapped/auth/spotify/callback`
- `POST /v1/wrapped/generate` → enqueues job
- `GET /v1/wrapped/status/{job_id}` → status, progress
- `GET /v1/wrapped/result/{result_id}` → JSON result
- `GET /v1/wrapped/download/{result_id}` → PNG

## Notes

- Config access is centralized via `platform_core.config._require_env_str`, failing fast when a required var is missing.
- Worker entrypoint: `music_wrapped_api.worker_entry:main` (`music-wrapped-worker` console script)
- 100% statements and branch coverage enforced via `make check`.
