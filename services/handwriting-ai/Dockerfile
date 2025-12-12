# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11.9-slim-bookworm
ARG APP_DIR=services/handwriting-ai

# ---------- Builder: resolve deps and build wheel ----------
FROM python:${PYTHON_VERSION} AS builder

ARG APP_DIR=services/handwriting-ai

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.3 \
    POETRY_VIRTUALENVS_CREATE=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl libjpeg62-turbo-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    && pip install "poetry==${POETRY_VERSION}"

WORKDIR /workspace

# Copy project and shared libs in monorepo layout
COPY ${APP_DIR}/pyproject.toml ${APP_DIR}/poetry.lock ${APP_DIR}/README.md ./services/handwriting-ai/
COPY ${APP_DIR}/src ./services/handwriting-ai/src
COPY ${APP_DIR}/config ./services/handwriting-ai/config
COPY ${APP_DIR}/scripts ./services/handwriting-ai/scripts
COPY ${APP_DIR}/seed ./services/handwriting-ai/seed
COPY libs ./libs

WORKDIR /workspace/services/handwriting-ai

RUN poetry build -f wheel

# ---------- Runtime base: install the wheel ----------
FROM python:${PYTHON_VERSION} AS runtime-base

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN addgroup --system app && adduser --system --home /home/app --ingroup app app

WORKDIR /workspace

COPY --from=builder /workspace/services/handwriting-ai/dist/*.whl /workspace/dist/
COPY --from=builder /workspace/libs /workspace/libs
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        /workspace/dist/*.whl /workspace/libs/platform_core /workspace/libs/platform_workers \
    && rm -rf /workspace/dist /workspace/libs /root/.cache

# Runtime assets (config/scripts/seed) live under /app
WORKDIR /app
COPY --from=builder /workspace/services/handwriting-ai/scripts ./scripts
COPY --from=builder /workspace/services/handwriting-ai/config ./config
COPY --from=builder /workspace/services/handwriting-ai/seed ./seed

USER app

# ---------- API target ----------
FROM runtime-base AS api

EXPOSE 8000

# Launch hypercorn (PORT may be injected by platform)
CMD ["sh", "-c", "exec hypercorn 'handwriting_ai.api.main:create_app()' --bind [::]:${PORT:-8000}"]

# ---------- Worker target ----------
FROM runtime-base AS worker
CMD ["sh", "-c", "exec handwriting-rq-worker"]
