# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11.9-slim-bookworm
ARG APP_DIR=clients/DiscordBot

# ---------- Builder: resolve deps and build wheel ----------
FROM python:${PYTHON_VERSION} AS builder

ARG APP_DIR=clients/DiscordBot

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
COPY ${APP_DIR}/pyproject.toml ${APP_DIR}/poetry.lock ${APP_DIR}/README.md ./clients/DiscordBot/
COPY ${APP_DIR}/src ./clients/DiscordBot/src
COPY ${APP_DIR}/scripts ./clients/DiscordBot/scripts
COPY libs ./libs

WORKDIR /workspace/clients/DiscordBot

RUN poetry build -f wheel

# ---------- Runtime base: install the wheel ----------
FROM python:${PYTHON_VERSION} AS runtime-base

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN addgroup --system app && adduser --system --ingroup app app

WORKDIR /workspace

COPY --from=builder /workspace/clients/DiscordBot/dist/*.whl /workspace/dist/
COPY --from=builder /workspace/libs /workspace/libs
RUN pip install /workspace/dist/*.whl /workspace/libs/platform_core /workspace/libs/platform_workers /workspace/libs/platform_discord \
    && rm -rf /workspace/dist /workspace/libs /root/.cache

WORKDIR /app
COPY --from=builder /workspace/clients/DiscordBot/scripts ./scripts

USER app

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python scripts/healthcheck.py || exit 1

# ---------- Bot target ----------
FROM runtime-base AS bot

CMD ["sh", "-c", "exec python -m clubbot.main"]

# ---------- Worker target ----------
FROM runtime-base AS worker

CMD ["sh", "-c", "exec rq worker transcript --with-scheduler"]
