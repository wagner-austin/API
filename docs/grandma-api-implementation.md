# Grandma API - Implementation Plan

**Purpose:** Simple Vietnamese-to-English audio translation for communicating with grandmother who only speaks Vietnamese.

**Status:** Core Implementation Complete (Phases 1-3 done, Phases 4-5 pending deployment)

---

## 1) Overview

A minimal API service that:
1. Receives audio recordings (30-second chunks)
2. Translates Vietnamese speech to English text using OpenAI Whisper
3. Returns English transcript

Paired with a simple static frontend on GitHub Pages.

---

## 2) Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  GitHub Pages - Static Frontend                                 │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  index.html (~150 lines)                                    ││
│  │  - Password unlock (simple token)                           ││
│  │  - Record button (MediaRecorder API)                        ││
│  │  - Sends 30s audio chunks                                   ││
│  │  - Displays English transcript                              ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ POST /translate (audio + token)
┌─────────────────────────────────────────────────────────────────┐
│  grandma-api (Railway)                                          │
│                                                                 │
│  FastAPI + Hypercorn                                            │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  POST /translate                                            ││
│  │  1. Verify token                                            ││
│  │  2. Receive audio file                                      ││
│  │  3. Call OpenAI Whisper translate (vi → en)                 ││
│  │  4. Return {"text": "English transcript"}                   ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                 │
│  Dependencies:                                                  │
│  - platform_stt (OpenAISttClient)                               │
│  - platform_core (logging, errors)                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3) Service Structure

```
services/grandma-api/
├── Dockerfile
├── Makefile
├── pyproject.toml
├── README.md
├── src/grandma_api/
│   ├── __init__.py
│   ├── asgi.py              # ASGI entry point
│   ├── config.py            # Settings TypedDict with encode/decode/require
│   ├── health.py            # Health check utilities
│   ├── types.py             # TranslationResult TypedDict
│   └── api/
│       ├── __init__.py
│       ├── _test_hooks.py   # DI for STT client factory
│       ├── main.py          # FastAPI app factory
│       └── routes/
│           ├── __init__.py
│           ├── health.py    # /healthz endpoint
│           └── translate.py # /translate endpoint
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_api_main.py
│   ├── test_api_routes_health.py
│   ├── test_api_routes_translate.py
│   ├── test_api_test_hooks.py
│   ├── test_asgi.py
│   ├── test_config.py
│   ├── test_health.py
│   ├── test_script_guard_entrypoint.py
│   └── test_types.py
└── scripts/
    └── guard.py             # Standard guard checks
```

---

## 4) Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Yes | - | OpenAI API key for Whisper |
| `API_TOKEN` | Yes | - | Simple auth token |
| `PORT` | No | `8080` | Server port (Railway sets this) |
| `LOG_LEVEL` | No | `INFO` | Logging level |

### Config TypedDict

```python
# src/grandma_api/config.py

class GrandmaApiSettings(TypedDict):
    """Configuration for grandma-api."""
    openai_api_key: str
    api_token: str
    port: int
    log_level: str


def load_settings() -> GrandmaApiSettings:
    """Load settings from environment variables."""
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    if not openai_key:
        raise ValueError("OPENAI_API_KEY environment variable required")

    api_token = os.environ.get("API_TOKEN", "")
    if not api_token:
        raise ValueError("API_TOKEN environment variable required")

    return GrandmaApiSettings(
        openai_api_key=openai_key,
        api_token=api_token,
        port=int(os.environ.get("PORT", "8080")),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
    )
```

---

## 5) API Endpoints

### GET /healthz

Health check (liveness probe) for Railway.

**Response:**
```json
{"status": "ok"}
```

### POST /translate

Translate Vietnamese audio to English text.

**Request:**
- Content-Type: `multipart/form-data`
- Body:
  - `audio`: Audio file (webm, mp3, wav, etc.)
  - `token`: Auth token (form field or query param)

**Response (200):**
```json
{
  "text": "Hello grandmother, how are you today?"
}
```

**Response (401):**
```json
{
  "detail": "Invalid token"
}
```

**Response (400):**
```json
{
  "detail": "No audio file provided"
}
```

---

## 6) Implementation

### main.py

```python
"""Grandma API - Vietnamese to English audio translation."""

from __future__ import annotations

import os
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from platform_core import get_logger, setup_logging

from grandma_api import _test_hooks
from grandma_api.config import load_settings

setup_logging()
logger = get_logger(__name__)

app = FastAPI(
    title="Grandma API",
    description="Vietnamese to English audio translation",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # GitHub Pages domain in prod
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/translate")
async def translate(
    audio: Annotated[UploadFile, File(description="Audio file to translate")],
    token: Annotated[str, Form(description="Auth token")] = "",
) -> dict[str, str]:
    """Translate Vietnamese audio to English text.

    Args:
        audio: Audio file (webm, mp3, wav supported).
        token: Authentication token.

    Returns:
        Dictionary with "text" key containing English translation.

    Raises:
        HTTPException: 401 if token invalid, 400 if no audio.
    """
    settings = load_settings()

    # Verify token
    if token != settings["api_token"]:
        logger.warning("Invalid token attempt")
        raise HTTPException(status_code=401, detail="Invalid token")

    # Read audio
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="No audio file provided")

    logger.info(
        "Translating audio",
        extra={"filename": audio.filename, "size_bytes": len(audio_bytes)},
    )

    # Get client and translate
    client = _test_hooks.get_stt_client(settings["openai_api_key"])

    result = client.translate(
        file=audio_bytes,
        filename=audio.filename or "audio.webm",
    )

    logger.info("Translation complete", extra={"text_length": len(result["text"])})

    return {"text": result["text"]}
```

### _test_hooks.py

```python
"""Dependency injection hooks for testing."""

from __future__ import annotations

from platform_stt import OpenAISttClient

# Production implementation - tests override this
_client_cache: dict[str, OpenAISttClient] = {}


def get_stt_client(api_key: str) -> OpenAISttClient:
    """Get or create OpenAI STT client.

    Args:
        api_key: OpenAI API key.

    Returns:
        OpenAISttClient instance (cached by api_key).
    """
    if api_key not in _client_cache:
        _client_cache[api_key] = OpenAISttClient(api_key=api_key)
    return _client_cache[api_key]


def reset_hooks() -> None:
    """Reset hooks to defaults. Called by tests."""
    _client_cache.clear()
```

---

## 7) Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install ffmpeg (required by platform_stt for audio processing)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# Copy libs (dependencies)
COPY libs/platform_core /app/libs/platform_core
COPY libs/platform_stt /app/libs/platform_stt

# Copy service
COPY services/grandma-api /app/services/grandma-api

# Install poetry and dependencies
RUN pip install --no-cache-dir poetry && \
    cd /app/libs/platform_core && poetry install --only main --no-interaction && \
    cd /app/libs/platform_stt && poetry install --only main --no-interaction && \
    cd /app/services/grandma-api && poetry install --only main --no-interaction

WORKDIR /app/services/grandma-api

# Hypercorn for production
CMD ["poetry", "run", "hypercorn", "grandma_api.main:app", "--bind", "0.0.0.0:8080"]
```

---

## 8) pyproject.toml

```toml
[build-system]
requires = ["poetry-core>=1.3.0"]
build-backend = "poetry.core.masonry.api"

[tool.poetry]
name = "grandma-api"
version = "0.1.0"
description = "Vietnamese to English audio translation API using OpenAI Whisper."
authors = ["Austin Wagner <austinwagner@msn.com>"]
packages = [
  { include = "grandma_api", from = "src" },
  { include = "scripts" },
]

[tool.poetry.dependencies]
python = "^3.11"
fastapi = "^0.124"
hypercorn = "^0.18"
python-multipart = "^0.0.18"
typing-extensions = "^4.12.2"
platform-core = { path = "../../libs/platform_core", develop = true }
platform-stt = { path = "../../libs/platform_stt", develop = true }

[tool.poetry.group.dev.dependencies]
pytest = "^9.0.0"
pytest-asyncio = "^1.3.0"
pytest-cov = "^7.0.0"
pytest-xdist = "^3.6.1"
mypy = "^1.13.0"
ruff = "^0.14.4"
httpx = "^0.28.0"

[tool.mypy]
python_version = "3.11"
strict = true
warn_unused_ignores = true
disallow_any_unimported = true
disallow_any_expr = true
disallow_any_decorated = true
disallow_any_explicit = true
disallow_any_generics = true
files = ["src", "tests", "scripts"]

[tool.ruff]
line-length = 100
target-version = "py311"
src = ["src", "tests", "scripts"]

[tool.ruff.lint]
select = ["E", "F", "I", "B", "BLE", "UP", "N", "C4", "SIM", "RET", "C90", "RUF", "ANN"]

[tool.coverage.run]
source = ["src", "scripts"]
omit = []
branch = true

[tool.coverage.report]
precision = 2
show_missing = true
fail_under = 100

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
addopts = "-v -n auto --dist loadscope"
```

---

## 9) Frontend (GitHub Pages)

Single `index.html` file - separate repo or `gh-pages` branch:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Grandma Translator</title>
  <style>
    * { box-sizing: border-box; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      max-width: 500px;
      margin: 0 auto;
      padding: 20px;
      background: #f5f5f5;
    }
    h1 { text-align: center; }
    .card {
      background: white;
      border-radius: 12px;
      padding: 24px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.1);
      margin-bottom: 16px;
    }
    input[type="password"], input[type="text"] {
      width: 100%;
      padding: 12px;
      font-size: 16px;
      border: 1px solid #ddd;
      border-radius: 8px;
      margin-bottom: 12px;
    }
    button {
      width: 100%;
      padding: 16px;
      font-size: 18px;
      border: none;
      border-radius: 8px;
      cursor: pointer;
      transition: transform 0.1s;
    }
    button:active { transform: scale(0.98); }
    .btn-primary { background: #007AFF; color: white; }
    .btn-record {
      background: #FF3B30;
      color: white;
      font-size: 24px;
      padding: 32px;
      border-radius: 50%;
      width: 120px;
      height: 120px;
      margin: 20px auto;
      display: block;
    }
    .btn-record.recording { background: #34C759; }
    #app { display: none; }
    #login.hidden { display: none; }
    #app.visible { display: block; }
    #status {
      text-align: center;
      color: #666;
      margin: 12px 0;
    }
    #transcript {
      background: #f9f9f9;
      border-radius: 8px;
      padding: 16px;
      min-height: 150px;
      white-space: pre-wrap;
      font-size: 18px;
      line-height: 1.5;
    }
    .error { color: #FF3B30; }
  </style>
</head>
<body>
  <div id="login" class="card">
    <h1>Grandma Translator</h1>
    <input type="password" id="token" placeholder="Enter password" autocomplete="off">
    <button class="btn-primary" onclick="unlock()">Enter</button>
  </div>

  <div id="app">
    <div class="card" style="text-align: center;">
      <button id="recordBtn" class="btn-record" onclick="toggleRecording()">
        🎤
      </button>
      <div id="status">Tap to record</div>
    </div>

    <div class="card">
      <div id="transcript">Translations will appear here...</div>
    </div>

    <button class="btn-primary" onclick="clearTranscript()" style="background: #8E8E93;">
      Clear
    </button>
  </div>

  <script>
    // CONFIG - Update with your Railway URL
    const API_URL = 'https://grandma-api-production.up.railway.app';

    let token = '';
    let mediaRecorder = null;
    let audioChunks = [];
    let isRecording = false;

    function unlock() {
      token = document.getElementById('token').value;
      if (token) {
        document.getElementById('login').classList.add('hidden');
        document.getElementById('app').classList.add('visible');
      }
    }

    async function toggleRecording() {
      const btn = document.getElementById('recordBtn');
      const status = document.getElementById('status');

      if (!isRecording) {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });
          audioChunks = [];

          mediaRecorder.ondataavailable = (e) => audioChunks.push(e.data);
          mediaRecorder.onstop = sendAudio;

          mediaRecorder.start();
          isRecording = true;
          btn.classList.add('recording');
          btn.textContent = '⏹';
          status.textContent = 'Recording... Tap to stop';
        } catch (err) {
          status.textContent = 'Microphone access denied';
          status.classList.add('error');
        }
      } else {
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(t => t.stop());
        isRecording = false;
        btn.classList.remove('recording');
        btn.textContent = '🎤';
        status.textContent = 'Processing...';
      }
    }

    async function sendAudio() {
      const status = document.getElementById('status');
      const transcript = document.getElementById('transcript');

      const blob = new Blob(audioChunks, { type: 'audio/webm' });
      const formData = new FormData();
      formData.append('audio', blob, 'recording.webm');
      formData.append('token', token);

      try {
        const res = await fetch(`${API_URL}/translate`, {
          method: 'POST',
          body: formData,
        });

        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || 'Translation failed');
        }

        const data = await res.json();

        // Append new translation
        if (transcript.textContent === 'Translations will appear here...') {
          transcript.textContent = '';
        }
        transcript.textContent += data.text + '\n\n';
        status.textContent = 'Tap to record';
        status.classList.remove('error');
      } catch (err) {
        status.textContent = err.message;
        status.classList.add('error');
      }
    }

    function clearTranscript() {
      document.getElementById('transcript').textContent = 'Translations will appear here...';
    }

    // Auto-focus password field
    document.getElementById('token').focus();
  </script>
</body>
</html>
```

---

## 10) Railway Deployment

### Setup

```bash
# Install Railway CLI
npm i -g @railway/cli

# Login
railway login

# Init new project (from repo root)
cd /path/to/API
railway init

# Set root directory to services/grandma-api in Railway dashboard
# Or use railway.toml
```

### railway.toml (in services/grandma-api/)

```toml
[build]
dockerfilePath = "Dockerfile"

[deploy]
healthcheckPath = "/health"
healthcheckTimeout = 30
restartPolicyType = "ON_FAILURE"
```

### Environment Variables

Set in Railway dashboard or CLI:

```bash
railway variables set OPENAI_API_KEY="sk-..."
railway variables set API_TOKEN="your-secret-password"
```

### Deploy

```bash
railway up
```

---

## 11) Implementation Phases

### Phase 1: Service Setup ✅
- [x] Create `services/grandma-api/` directory structure
- [x] Create `pyproject.toml` with dependencies
- [x] Create `Makefile` with standard targets
- [x] Create `scripts/guard.py`

### Phase 2: Core Implementation ✅
- [x] Implement `config.py` with settings TypedDict (encode/decode/require pattern)
- [x] Implement `types.py` with TranslationResult TypedDict
- [x] Implement `api/_test_hooks.py` for STT client DI
- [x] Implement `api/main.py` with app factory
- [x] Implement `api/routes/health.py` (/healthz endpoint)
- [x] Implement `api/routes/translate.py` (/translate endpoint)
- [x] Create `asgi.py` entry point
- [ ] Create Dockerfile (pending)

### Phase 3: Testing ✅
- [x] Add tests for config loading (test_config.py)
- [x] Add tests for /healthz endpoint (test_api_routes_health.py)
- [x] Add tests for /translate endpoint with fake client (test_api_routes_translate.py)
- [x] Add tests for _test_hooks (test_api_test_hooks.py)
- [x] Add tests for types (test_types.py)
- [x] Add tests for guard script (test_script_guard_entrypoint.py)
- [x] Achieve 100% coverage
- [x] Run `make check` - all passing

### Phase 4: Deployment (Pending)
- [ ] Create Railway project
- [ ] Configure environment variables
- [ ] Deploy and verify /healthz
- [ ] Test /translate with real audio

### Phase 5: Frontend (Pending)
- [ ] Create GitHub repo for frontend
- [ ] Add index.html
- [ ] Enable GitHub Pages
- [ ] Update API_URL in frontend
- [ ] Test end-to-end

---

## 12) Testing Strategy

### Fake STT Client

```python
# tests/conftest.py
from grandma_api import _test_hooks
from platform_stt import VerboseResponse

class FakeSttClient:
    def __init__(self, response_text: str = "Hello from grandmother") -> None:
        self.response_text = response_text
        self.calls: list[tuple[bytes, str]] = []

    def translate(self, file: bytes, filename: str) -> VerboseResponse:
        self.calls.append((file, filename))
        return VerboseResponse(text=self.response_text, segments=[])

@pytest.fixture
def fake_client() -> FakeSttClient:
    client = FakeSttClient()
    original = _test_hooks.get_stt_client
    _test_hooks.get_stt_client = lambda _: client
    yield client
    _test_hooks.get_stt_client = original
```

### Test Cases

| Test | Description |
|------|-------------|
| `test_health` | GET /health returns {"status": "ok"} |
| `test_translate_success` | Valid token + audio returns translation |
| `test_translate_invalid_token` | Wrong token returns 401 |
| `test_translate_missing_audio` | No audio file returns 400 |
| `test_translate_empty_audio` | Empty audio returns 400 |

---

## 13) Code Standards

All code must follow project standards:
- No `Any`, no `cast`, no `type: ignore`
- TypedDict for all structured data
- `_test_hooks.py` for dependency injection
- 100% test coverage
- Google-style docstrings

---

## 14) Future Enhancements (Out of Scope)

- Bidirectional translation (English → Vietnamese for replies)
- Conversation history persistence
- Multiple language support
- Audio playback of translations
- Mobile app wrapper
