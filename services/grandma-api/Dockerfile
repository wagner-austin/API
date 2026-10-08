# syntax=docker/dockerfile:1

ARG PYTHON_VERSION=3.11.9-slim-bookworm
ARG APP_DIR=services/grandma-api

# ---------- Builder: resolve deps and build wheel ----------
FROM python:${PYTHON_VERSION} AS builder

ARG APP_DIR=services/grandma-api

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
COPY ${APP_DIR}/pyproject.toml ${APP_DIR}/poetry.lock ${APP_DIR}/README.md ./services/grandma-api/
COPY ${APP_DIR}/src ./services/grandma-api/src
COPY ${APP_DIR}/scripts ./services/grandma-api/scripts
COPY libs ./libs

WORKDIR /workspace/services/grandma-api

RUN poetry build -f wheel

# ---------- Runtime base: install the wheel ----------
FROM python:${PYTHON_VERSION} AS runtime-base

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ffmpeg is a runtime dependency: core/audio/converter.py shells out to it for
# every audio conversion. Found missing 2026-09-08 by CI's real-ffmpeg test
# (FileNotFoundError: 'ffmpeg'), which was the honest verdict -- the runtime
# image never carried the binary the converter calls.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN addgroup --system app && adduser --system --home /home/app --ingroup app app

WORKDIR /workspace

# torch comes from the PyTorch CPU index, as in handwriting-ai: the wheel's
# platform_langid needs torch, and from PyPI pip takes the CUDA build and its
# nvidia libraries, for a service whose language detector runs on
# platform_langid's DEFAULT_DEVICE "cpu" and whose compose file reserves no
# GPU. Measured on lavender-wsl's rootless daemon (MCPs board task
# 90c135ac): the install stage rebuilt cold in 668 s with the CUDA build and
# 292 s with this index, and the image holds torch 2.14.1+cpu at 2.47 GB.
#
# platform_langid's model stack gets its own layer, before the wheel and the
# libs are copied in, so a change to the service's source or to any lib
# reinstalls only those and never this layer: every lib change made the
# whole install a cold one, the slowest case of tools/maketools-execution.
# Each requirement is the constraint its lib declares (platform_langid for
# torch, torchaudio and transformers, platform_stt for numpy), which
# tests/test_dockerfile_model_layer.py pins to their pyproject files: with
# bare names this layer took transformers 5 and numpy 2, and the install
# below spent 106 s replacing them.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.0.0,<3.0.0" "torchaudio>=2.0.0,<3.0.0" "transformers>=4.30.0,<5.0.0" "numpy<2.0" \
    && rm -rf /root/.cache

COPY --from=builder /workspace/services/grandma-api/dist/*.whl /workspace/dist/
COPY --from=builder /workspace/libs /workspace/libs
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        /workspace/dist/*.whl /workspace/libs/platform_core /workspace/libs/platform_stt \
    && rm -rf /workspace/dist /workspace/libs /root/.cache

WORKDIR /app

USER app

# ---------- API target ----------
FROM runtime-base AS api
EXPOSE 8080
CMD ["sh", "-c", "exec hypercorn 'grandma_api.asgi:app' --bind [::]:${PORT:-8080}"]
