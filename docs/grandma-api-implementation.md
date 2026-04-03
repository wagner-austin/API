# Grandma API - Implementation Plan

**Purpose:** Multi-language audio-to-English translation API. Originally built for Vietnamese-to-English communication, now supports 57 input languages with automatic language detection.

**Status:** Core implementation complete. Frontend shipped as companion TypeScript app (not GitHub Pages).

---

## Overview

A service that:
1. Receives audio recordings
2. Detects the spoken language automatically via `platform_langid` (Meta MMS-LID)
3. Transcribes speech via `platform_stt` (OpenAI Whisper)
4. Translates to English via `platform_translate` (Anthropic Claude)
5. Returns English transcript

---

## Architecture

The service is a FastAPI backend on port 8008 with a companion vanilla TypeScript frontend on port 8091.

**Translation pipeline:**

```
Audio → [Language ID] → [Transcription] → [Translation] → English
         platform_langid   platform_stt      platform_translate
```

**Dependencies:**
- `platform-core` — logging, errors, config
- `platform-stt` — OpenAI Whisper client
- `platform-langid` — spoken language identification (Meta MMS-LID)
- `platform-translate` — text translation (Anthropic Claude backend)

---

## API Endpoints

- `GET /healthz` — Liveness probe
- `POST /translate` — Translate audio to English text (multipart/form-data: `audio` file + `token` auth)

Supported audio formats: WebM, MP3, WAV, M4A, OGG.

Error responses use structured format: `{code, message, request_id}`.

---

## Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | OpenAI API key for Whisper |
| `API_TOKEN` | Yes | - | Simple auth token |
| `PORT` | No | `8080` | Server port (Railway sets this) |
| `LOG_LEVEL` | No | `INFO` | Logging level |

---

## Frontend

Companion browser app in `services/grandma-api/web/` (vanilla TypeScript, Vitest, ESLint). Provides audio recording via MediaRecorder API with real-time translation display. Uses `ts-ebml` for WebM audio handling.

---

## Deployment

Deployed to Railway. See `Dockerfile` and `railway.toml` in `services/grandma-api/`.

---

## Implementation Phases

| Phase | Status |
|-------|--------|
| 1. Service setup (directory, pyproject, Makefile, guard) | Done |
| 2. Core implementation (config, types, routes, DI hooks) | Done |
| 3. Testing (100% coverage) | Done |
| 4. Multi-language support (57 languages, auto-detection) | Done |
| 5. TypeScript frontend | Done |
| 6. Railway deployment | Pending |
