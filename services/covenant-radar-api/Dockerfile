# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11.9-slim-bookworm
ARG APP_DIR=services/covenant-radar-api

# ---------- Builder: resolve deps and build wheel ----------
FROM python:${PYTHON_VERSION} AS builder

ARG APP_DIR=services/covenant-radar-api

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
COPY ${APP_DIR}/pyproject.toml ${APP_DIR}/poetry.lock ${APP_DIR}/README.md ./services/covenant-radar-api/
COPY ${APP_DIR}/src ./services/covenant-radar-api/src
COPY ${APP_DIR}/scripts ./services/covenant-radar-api/scripts
COPY libs ./libs

WORKDIR /workspace/services/covenant-radar-api

RUN poetry build -f wheel

# ---------- Runtime base: install the wheel ----------
FROM python:${PYTHON_VERSION} AS runtime-base

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN addgroup --system app && adduser --system --home /home/app --ingroup app app

WORKDIR /workspace

COPY --from=builder /workspace/services/covenant-radar-api/dist/*.whl /workspace/dist/
COPY --from=builder /workspace/libs /workspace/libs
RUN pip install /workspace/dist/*.whl \
    /workspace/libs/platform_core \
    /workspace/libs/platform_workers \
    /workspace/libs/platform_ml \
    /workspace/libs/covenant_domain \
    /workspace/libs/covenant_ml \
    /workspace/libs/covenant_persistence \
    && rm -rf /workspace/dist /workspace/libs /root/.cache

WORKDIR /app

USER app

# ---------- API target ----------
FROM runtime-base AS api

EXPOSE 8000

CMD ["sh", "-c", "exec hypercorn 'covenant_radar_api.main:create_app()' --bind [::]:${PORT:-8000}"]

# ---------- Worker target ----------
FROM runtime-base AS worker

CMD ["sh", "-c", "exec covenant-rq-worker"]
