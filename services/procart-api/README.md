# procart-api

Strict, typed FastAPI service orchestrating procart rendering.

## Installation

```bash
cd services/procart-api
poetry install --with dev
```

## Run

```bash
# Development
poetry run hypercorn procart_api.main:app --bind 0.0.0.0:8000 --reload

# Production
poetry run hypercorn procart_api.main:app --bind [::]:${PORT:-8000}
```

## Endpoints

- GET /healthz — liveness
- GET /readyz — readiness
- GET /registries/modules — list available visual modules
- GET /registries/camera-paths — list camera path types
- GET /registries/tone-mappers — list tone mapping types
- GET /registries/post-effects — list post-processing effects
- GET /registries/composite-ops — list composite operations
- POST /render/preview — render a single frame preview
- POST /render/frames — render all frames to disk
- POST /render/video — encode frames to video via ffmpeg

See docs/api.md for request/response schemas.

## Development

```bash
make lint   # guards + ruff + mypy
make test   # pytest with --cov-branch
make check  # lint + test (fail_under=100)
```

## Project Structure

```
src/procart_api/
  app.py         # FastAPI factory with platform_core exception handlers
  _test_hooks.py # injectable hooks (ffmpeg runner)
  main.py        # production entrypoint; sets real hooks
  routes/
    health.py    # /healthz + /readyz via add_api_route
scripts/
  guard.py       # monorepo guard harness
  _test_hooks.py # injectable hooks (http client, display, event source)
  demo_video.py  # renders a demo scene through a running service
  live_preview.py # interactive pygame tuner for the neon-orb parameters
tests/            # ASGI tests via httpx.ASGITransport
docs/PLAN.md      # service integration plan
```

## Scripts

```bash
# Render a demo scene end to end against a running service
poetry run python -m scripts.demo_video --base-url http://127.0.0.1:8000 --out demo_output

# Interactive parameter tuner (opens a window; keys are listed in its docstring)
poetry run python -m scripts.live_preview
```

## Hooks

- FFMPEG_RUNNER Protocol provided by libs/procart; production sets the real runner in main.py and tests set fakes.
- No conditionals in core paths; call the hook directly.

## Standards

- Hypercorn ASGI server in dependencies; no Uvicorn in code
- Guards, mypy --strict, Ruff over src/tests/scripts; --cov-branch with fail_under=100
- No Any, cast, type: ignore, .pyi, or dataclasses in src; no try/except in core



## API Schema

See docs/api.md for request/response examples and typed schema notes.

