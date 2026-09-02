# syntax=docker/dockerfile:1
#
# TankpitBot in a container, so a live session is not tied to one
# workstation.
#
# Build context is the REPO ROOT, not this directory. `pyproject.toml`
# takes `monorepo-guards` and `platform-core` as path dependencies at
# `../../libs/`, which a context rooted here cannot see:
#
#     docker build -f clients/TankpitBot/Dockerfile -t tankpit-bot .
#
# Slim rather than the official Playwright image. `mcr.microsoft.com/
# playwright/python` ships Chromium, Firefox and WebKit plus their
# system deps and lands near 2 GB; this bot drives Chromium only.
# Installing the single browser onto `python:3.11-slim` gets the same
# capability at roughly a quarter of that.
#
# CREDENTIALS ARE NEVER BAKED, and the container takes them by ENV
# rather than by file. `browser/accounts.py` resolves in three steps and
# the first is `TANKPIT_USERNAME` + `TANKPIT_PASSWORD`, so a
# single-account run needs no `accounts.json` at all.
#
# That is not merely tidier — the checkout-relative accounts path does
# not survive installation (four parents up from the installed module is
# site-packages). FLEET MODE therefore names the pool explicitly:
# `TANKPIT_ACCOUNTS_FILE` points at a read-only mount (see the `fleet`
# service in docker-compose.yml), which is the env-overridable path this
# header once recorded as the missing piece. The pool is never baked;
# single-account runs still need no file at all.
#
# The same image serves both shapes: CMD is the single bounded bot;
# the fleet service overrides the command to `tankpit-fleet`, whose
# spawner forks `sys.executable -c` children — no poetry, no shell,
# so it works inside the installed image unchanged.

ARG PYTHON_VERSION=3.11.9-slim-bookworm
ARG APP_DIR=clients/TankpitBot

# ---------- Builder: resolve deps and build the wheel ----------
FROM python:${PYTHON_VERSION} AS builder

ARG APP_DIR=clients/TankpitBot

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_VERSION=1.8.3 \
    POETRY_VIRTUALENVS_CREATE=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip \
    && pip install "poetry==${POETRY_VERSION}"

WORKDIR /workspace

# Monorepo layout, mirroring the service Dockerfiles: the package's own
# manifest and sources, then `libs/` for the two path dependencies.
COPY ${APP_DIR}/pyproject.toml ${APP_DIR}/poetry.lock ${APP_DIR}/README.md ./clients/TankpitBot/
COPY ${APP_DIR}/src ./clients/TankpitBot/src
COPY ${APP_DIR}/scripts ./clients/TankpitBot/scripts

# ONLY the two path dependencies this package declares, never `libs/`
# wholesale. `pyproject.toml` takes `monorepo-guards` and
# `platform-core` from `../../libs/`; the rest of that tree is
# cleargbm_rs, the covenant libs and friends, none of which this bot
# imports.
#
# Measured: copying all of `libs/` cost 1.54 GB of the first image's
# 3.54 GB. The `rm -rf` in the install step below does not reclaim it —
# a COPY layer survives a later delete. That is the exact failure the
# repo's own `.dockerignore` records against Model-Trainer ("baked into
# a COPY --from=builder layer that survived the later rm -rf"), and it
# is inherited by copying the service Dockerfiles' shape.
COPY libs/monorepo_guards ./libs/monorepo_guards
COPY libs/platform_core ./libs/platform_core

WORKDIR /workspace/clients/TankpitBot

RUN poetry build -f wheel

# ---------- Runtime ----------
FROM python:${PYTHON_VERSION} AS runtime

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Playwright resolves browsers here. Set before the install so the
    # download lands somewhere the non-root user can read, rather than
    # in root's home where it would be unreadable at run time.
    PLAYWRIGHT_BROWSERS_PATH=/opt/playwright

RUN addgroup --system app && adduser --system --home /home/app --ingroup app app

WORKDIR /workspace

COPY --from=builder /workspace/clients/TankpitBot/dist/*.whl /workspace/dist/
COPY --from=builder /workspace/libs /workspace/libs

# `platform_core` is named explicitly; `monorepo_guards` is not, and must
# still be present. poetry-core writes path dependencies into the wheel
# as PEP 508 direct references (`monorepo-guards @ file:///workspace/
# libs/monorepo_guards`), so pip resolves it from that path rather than
# from an index. Delete the directory and the install fails; publish the
# wheel without the directory and pip would reach for an index instead,
# which for a private lib name is the interesting failure.
RUN pip install /workspace/dist/*.whl /workspace/libs/platform_core \
    && rm -rf /workspace/dist /workspace/libs /root/.cache

# Chromium only, with its system deps. `--with-deps` runs apt itself, so
# the browser and the libraries it dlopens stay in step; pinning that
# list by hand is how a Playwright upgrade turns into a blank page.
RUN playwright install --with-deps chromium \
    && rm -rf /var/lib/apt/lists/* /root/.cache \
    && chmod -R a+rX /opt/playwright

WORKDIR /app

# The tracked runtime data a checkout serves from its project root,
# baked because it VERSIONS WITH THE CODE: the static XOR key (without
# which no session decodes a single wire byte — the checkout-relative
# default resolves into site-packages once installed, so the ENV below
# names the baked copy) and the 44 field terrain GIFs (found relative
# to the working directory, which is exactly what /app is). Mutable
# state — accounts.json, data/tank_registry.json, .env, runs/ — is
# NEVER baked; the fleet service mounts it (see docker-compose.yml).
ARG APP_DIR=clients/TankpitBot
COPY ${APP_DIR}/xor_static_key.txt /app/
COPY ${APP_DIR}/field*.gif /app/
ENV TANKPIT_XOR_KEY_FILE=/app/xor_static_key.txt

# Session output. Declared so `runs/` survives the container and a
# capture can be read from the host afterwards — the event log is the
# artifact a session exists to produce.
RUN mkdir -p /app/runs && chown -R app:app /app
VOLUME ["/app/runs"]

USER app

# The fleet manager's control page, when the fleet service runs this
# image with `command: tankpit-fleet`. Documentation only — publishing
# is compose's decision.
EXPOSE 27300

# A bounded session, not a daemon. The bot is driven by
# TANKPIT_BOT_SESSION_SECONDS and exits; `docker run` inherits that
# lifetime rather than needing a stop signal. (The fleet override runs
# until drained instead; its children ignore this default because the
# manager states each child's environment explicitly.)
ENV TANKPIT_BOT_SESSION_SECONDS=30

CMD ["tankpit-bot"]
