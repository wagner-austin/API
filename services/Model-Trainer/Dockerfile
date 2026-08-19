# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11.9-slim-bookworm
ARG APP_DIR=services/Model-Trainer
# Must match the default docker-compose.yml passes. It previously defaulted to
# the CPU wheel index while compose always overrode it to cu124, so a plain
# `docker build` produced a silently CPU-only image that looked identical to
# the one the stack runs and would have trained without the GPU.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu124

# ---------- Builder: resolve deps and build wheel ----------
FROM python:${PYTHON_VERSION} AS builder

ARG APP_DIR=services/Model-Trainer

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

# Only the three path dependencies this service declares in pyproject.toml.
# `COPY libs ./libs` pulled all 23 packages in the monorepo -- 5.0GB reached
# the builder and was then baked into the runtime image by a COPY layer that
# the later `rm -rf` could not reclaim. Adding a lib here is deliberate: it
# must also be declared as a path dependency below.
COPY ${APP_DIR}/pyproject.toml ${APP_DIR}/poetry.lock ${APP_DIR}/README.md ./services/Model-Trainer/
COPY ${APP_DIR}/src ./services/Model-Trainer/src
COPY ${APP_DIR}/scripts ./services/Model-Trainer/scripts
COPY ${APP_DIR}/config ./services/Model-Trainer/config
COPY libs/platform_core ./libs/platform_core
COPY libs/platform_workers ./libs/platform_workers
COPY libs/platform_ml ./libs/platform_ml

WORKDIR /workspace/services/Model-Trainer

RUN poetry build -f wheel

# Third-party dependencies, resolved from the lock file and written without
# the `-e file://` entries for the three local libs. Those are installed from
# source in the application layer instead, so this file changes only when a
# dependency changes -- never when application source changes. BuildKit keys a
# COPY on file content, so an unchanged requirements.txt keeps the runtime's
# multi-gigabyte torch install cached across ordinary source edits.
RUN poetry export -f requirements.txt --without-hashes -o /tmp/requirements.full.txt \
    && grep -v '^-e file://' /tmp/requirements.full.txt > /workspace/requirements.txt

# ---------- Runtime base: install the wheel ----------
FROM python:${PYTHON_VERSION} AS runtime-base

# Redeclared because a stage does not inherit a global ARG's value. This is
# the declaration the install below actually reads, so it carries the same
# cu124 default as the global one and as docker-compose.yml.
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu124

# The service commit, baked at build time so every TrainingManifest can name
# the code that trained the run; the image carries no .git for rev-parse.
# Empty when the build did not pass it, which the manifest records as null.
ARG GIT_COMMIT=

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_INDEX=${TORCH_INDEX} \
    GIT_COMMIT=${GIT_COMMIT}

RUN addgroup --system app && adduser --system --home /home/app --ingroup app app

# Create HF cache directory with correct ownership before switching to app user
RUN mkdir -p /hf-cache && chown -R app:app /hf-cache

WORKDIR /workspace

# ---- Dependency layer ----
# Invalidated only by a dependency change. This is the layer that installs
# torch + the CUDA runtime, so keeping it off the source-edit path is what
# makes an incremental rebuild take seconds instead of reinstalling ~6GB.
COPY --from=builder /workspace/requirements.txt /workspace/requirements.txt
RUN pip install --default-timeout=1000 --extra-index-url ${TORCH_INDEX} -r /workspace/requirements.txt \
    && rm -rf /workspace/requirements.txt /root/.cache

# ---- Application layer ----
# Rebuilt on every source change, and small because every third-party
# dependency is already satisfied above. `--no-deps` is what holds that
# guarantee: if this wheel ever needs something the lock did not resolve, the
# install fails here rather than silently pulling a second resolution.
COPY --from=builder /workspace/services/Model-Trainer/dist/*.whl /workspace/dist/
COPY --from=builder /workspace/libs /workspace/libs
RUN pip install --no-deps /workspace/dist/*.whl /workspace/libs/platform_core /workspace/libs/platform_workers /workspace/libs/platform_ml \
    && rm -rf /workspace/dist /workspace/libs /root/.cache

WORKDIR /app

USER app

# ---------- API target ----------
FROM runtime-base AS api
EXPOSE 8000
CMD ["sh", "-c", "exec hypercorn 'model_trainer.api.main:create_app()' --bind [::]:${PORT:-8000}"]

# ---------- Worker target ----------
FROM runtime-base AS worker
CMD ["sh", "-c", "exec modeltrainer-rq-worker"]
