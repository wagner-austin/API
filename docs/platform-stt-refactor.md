# RFC: Platform STT Library Extraction

## Status
- **Complete**: All phases implemented and tested with 100% coverage
- Scope: `libs/platform_stt` (new package), `services/transcript-api` (migration pending)
- Non-goals: backwards compatibility layers, partial migrations

## Goals
- Extract a reusable, strictly-typed speech-to-text library from `transcript-api`.
- Centralize Whisper API client, audio chunking, parallel transcription, segment merging, and language detection.
- Enforce strict typing with no `Any`, no `cast`, no `type: ignore`, no mocks.
- Achieve and sustain 100% statement and branch coverage.
- Keep transcript-api thin: domain-specific API routes only.

## Principles (Hard Requirements)
- **Strict typing only**:
  - No `Any`, no `cast`, no `type: ignore`, no `.pyi`, no `noqa`.
  - Use `TypedDict`, `Protocol`, type aliases. No dataclasses in `src/`.
- **Parse/validate at edges**:
  - All types have `encode_*`, `decode_*`, and `require_*` functions.
  - Boundary validation using `JSONTypeError` from `platform_core`.
- **Dependency injection via hooks**:
  - `_test_hooks.py` with subprocess and ffmpeg hooks for testability.
  - `testing.py` exposes public test utilities for downstream services.
- **Fail loud and early**:
  - No best-effort fallbacks; exceptions propagate.
  - Explicit error types for all failure modes.

## High-Level Architecture

### libs/platform_stt (new)
```
libs/platform_stt/
├── src/platform_stt/
│   ├── __init__.py           # Public exports
│   ├── _test_hooks.py        # Internal dependency injection
│   ├── chunker.py            # AudioChunker - silence-based splitting
│   ├── langid.py             # FastText language detection
│   ├── merger.py             # TranscriptMerger - segment merging
│   ├── parallel.py           # ParallelTranscriber - bounded concurrency
│   ├── testing.py            # Public test utilities
│   ├── types.py              # TypedDicts with encode/decode/require
│   ├── whisper_client.py     # OpenAISttClient - Whisper API
│   └── whisper_parse.py      # SDK response parsing
└── tests/                    # 100% coverage tests
```

### services/transcript-api (to be updated)
- Thin API layer importing from `platform_stt`
- FastAPI routes for transcription/translation endpoints
- Job handlers for async processing

## Component Designs

### 1. OpenAISttClient
HTTP client for OpenAI Whisper API with typed request/response.

```python
class OpenAISttClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 120.0,
    ) -> None: ...

    def transcribe(
        self,
        file: BinaryIO,
        language: str,
        filename: str = "audio.mp3",
        prompt: str = "",
    ) -> VerboseResponse: ...

    def translate(
        self,
        file: BinaryIO,
        filename: str = "audio.mp3",
        prompt: str = "",
    ) -> VerboseResponse: ...
```

### 2. AudioChunker
Split large audio files at silence points using ffmpeg/ffprobe.

```python
class AudioChunker:
    def __init__(
        self,
        target_chunk_mb: float = 20.0,
        max_chunk_duration_seconds: float = 600.0,
        silence_threshold_db: float = -40.0,
        silence_duration_seconds: float = 0.5,
        ffmpeg_path: str = "ffmpeg",
        ffprobe_path: str = "ffprobe",
    ) -> None: ...

    def chunk_audio(self, audio_path: str) -> list[AudioChunk]: ...
```

**Chunking algorithm**:
1. Check file size - if under 25MB, return single chunk
2. Run ffprobe to detect container format and codec
3. Run silence detection filter to find split points
4. Calculate optimal split points based on size and duration limits
5. Split audio at silence points using stream copy (fallback to re-encode)

### 3. ParallelTranscriber
Transcribe chunks with bounded concurrency and retry logic.

```python
class ParallelTranscriber:
    def __init__(
        self,
        client: TranscribeProtocol,
        max_workers: int = 4,
        max_retries: int = 3,
    ) -> None: ...

    def transcribe_chunks(
        self,
        chunks: list[AudioChunk],
        language: str,
    ) -> list[tuple[AudioChunk, VerboseResponse]]: ...
```

Uses ThreadPoolExecutor with exponential backoff on failure.

### 4. TranscriptMerger
Merge segments from multiple chunks with time offset correction.

```python
class TranscriptMerger:
    def merge(
        self,
        segments_list: list[list[TranscriptSegment]],
        chunk_start_times: list[float] | None = None,
    ) -> list[TranscriptSegment]: ...

def merge_segment_text(segments: list[TranscriptSegment]) -> str: ...
```

### 5. Language Detection
FastText-based language identification using lid.218e or lid.176 models.

```python
def load_langid_model(data_dir: str, prefer_218e: bool = True) -> LangIdModelProtocol: ...
def detect_language(text: str, model: LangIdModelProtocol) -> LanguageDetectionResult: ...
def is_language(text: str, lang: str, model: LangIdModelProtocol, threshold: float = 0.8) -> bool: ...
```

### 6. Type Definitions
All types use TypedDict with full encode/decode/require pattern:

```python
class TranscriptSegment(TypedDict):
    text: str
    start: float
    duration: float

class VerboseSegment(TypedDict):
    text: str
    start: float
    end: float

class VerboseResponse(TypedDict):
    text: str
    segments: list[VerboseSegment]

class AudioChunk(TypedDict):
    path: str
    start_seconds: float
    duration_seconds: float
    size_bytes: int

class ChunkerConfig(TypedDict):
    target_chunk_mb: float
    max_chunk_duration_seconds: float
    silence_threshold_db: float
    silence_duration_seconds: float

class LanguageDetectionResult(TypedDict):
    language: str
    confidence: float
    script: str | None
```

## Testing Strategy

### Test Patterns
- No mocks - use `_test_hooks.py` for dependency injection
- `FakeSubprocessResult` for subprocess testing
- `FakeLangIdModel` for language detection testing
- All assertions must be specific (no `assert x is not None`)

### Coverage Requirements
- 100% statement coverage
- 100% branch coverage
- Coverage exclusions only for:
  - Protocol ellipsis (`...`)
  - Specific defensive type checks at boundaries

## Migration Plan

### Phase 1: Library Creation ✅
- [x] Create package structure
- [x] Define TypedDicts with encode/decode/require
- [x] Extract OpenAISttClient
- [x] Extract AudioChunker with silence detection
- [x] Extract ParallelTranscriber
- [x] Extract TranscriptMerger
- [x] Add FastText language detection
- [x] Create testing utilities
- [x] Achieve 100% coverage

### Phase 2: Integration (Pending)
- [ ] Update transcript-api to import from platform_stt
- [ ] Remove duplicated code from transcript-api
- [ ] Verify all existing functionality preserved
- [ ] Update transcript-api tests

### Phase 3: Cleanup (Pending)
- [ ] Remove old internal modules from transcript-api
- [ ] Update documentation
- [ ] Final integration testing

## Dependencies

### Runtime
- `platform-core`: Error handling, logging, JSON utilities
- `numpy < 2.0`: Audio processing (pinned for fasttext compatibility)
- `openai ^2.8.0`: Whisper API client
- `fasttext-wheel ^0.9.2`: Language detection models

### External Tools
- `ffmpeg`: Audio splitting and re-encoding
- `ffprobe`: Audio format detection

### Dev
- `pytest`: Testing
- `pytest-cov`: Coverage
- `mypy`: Type checking
- `ruff`: Linting
