"""Model Trainer API routes.

Endpoints:
    Health:
        GET  /healthz                      - Liveness probe (always returns ok)
        GET  /readyz                       - Readiness probe (checks Redis + workers)

    Training Runs (/runs prefix, API key required):
        POST /runs/train                   - Start a new training run
        GET  /runs/{run_id}                - Get run status
        POST /runs/{run_id}/evaluate       - Enqueue evaluation for a run
        GET  /runs/{run_id}/eval           - Get evaluation results
        GET  /runs/{run_id}/artifact       - Get artifact storage pointer
        GET  /runs/{run_id}/logs           - Get training logs (tail)
        GET  /runs/{run_id}/logs/stream    - Stream training logs (SSE)
        POST /runs/{run_id}/cancel         - Request run cancellation

    Tokenizers (/tokenizers prefix, API key required):
        POST /tokenizers/train             - Start tokenizer training
        GET  /tokenizers/{tokenizer_id}    - Get tokenizer info and stats
"""

from __future__ import annotations

from .health import build_router as build_health_router
from .runs import build_router as build_runs_router
from .tokenizers import build_router as build_tokenizers_router

__all__ = [
    "build_health_router",
    "build_runs_router",
    "build_tokenizers_router",
]
