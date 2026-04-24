# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11.9-slim-bookworm
ARG APP_DIR=services/doc-extract-api
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

# ---------- Builder: resolve deps and build wheel ----------
FROM python:${PYTHON_VERSION} AS builder

ARG APP_DIR=services/doc-extract-api

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.3 \
    POETRY_VIRTUALENVS_CREATE=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    && pip install "poetry==${POETRY_VERSION}"

WORKDIR /workspace

# Copy project and shared libs in monorepo layout
COPY ${APP_DIR}/pyproject.toml ${APP_DIR}/poetry.lock ${APP_DIR}/README.md ./services/doc-extract-api/
COPY ${APP_DIR}/src ./services/doc-extract-api/src
COPY ${APP_DIR}/scripts ./services/doc-extract-api/scripts
COPY libs ./libs

WORKDIR /workspace/services/doc-extract-api

RUN poetry build -f wheel

# ---------- Runtime base: install the wheel ----------
FROM python:${PYTHON_VERSION} AS runtime-base

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_INDEX=${TORCH_INDEX}

RUN addgroup --system app && adduser --system --home /home/app --ingroup app app

# Create docTR model cache directory with correct ownership
RUN mkdir -p /doctr-cache && chown -R app:app /doctr-cache

WORKDIR /workspace

COPY --from=builder /workspace/services/doc-extract-api/dist/*.whl /workspace/dist/
COPY --from=builder /workspace/libs /workspace/libs
RUN pip install --default-timeout=1000 --extra-index-url ${TORCH_INDEX} /workspace/dist/*.whl /workspace/libs/platform_core /workspace/libs/platform_workers \
    && rm -rf /workspace/dist /workspace/libs /root/.cache

WORKDIR /app

USER app

# ---------- API target ----------
FROM runtime-base AS api
EXPOSE 8000
CMD ["sh", "-c", "exec hypercorn 'doc_extract_api.asgi:app' --bind [::]:${PORT:-8000}"]

# ---------- Worker target ----------
FROM runtime-base AS worker
CMD ["sh", "-c", "exec doc-extract-rq-worker"]
