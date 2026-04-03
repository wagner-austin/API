# Speech Translation Pipeline

Modular architecture for reliable speech-to-English translation.

## Problem

Whisper's built-in translation is unreliable:
- Language auto-detect fails on noisy/far-field audio
- Translation sometimes just transcribes
- Single model doing too many things

## Solution

Best-in-class model for each task:

```
Audio → [Language ID] → [Transcription] → [Translation] → English
         platform_langid   platform_stt      platform_translate
```

## Architecture

### Module Structure

```
libs/
├── platform_langid/           # NEW - Spoken language detection
│   ├── src/platform_langid/
│   │   ├── __init__.py        # Public exports
│   │   ├── _test_hooks.py     # DI hooks for testing
│   │   ├── detector.py        # Core detection logic
│   │   ├── testing.py         # Public test utilities
│   │   └── types.py           # TypedDict definitions
│   └── tests/
│       └── test_*.py          # Full coverage tests
│
├── platform_translate/        # NEW - Text translation
│   ├── src/platform_translate/
│   │   ├── __init__.py        # Public exports
│   │   ├── _test_hooks.py     # DI hooks for testing
│   │   ├── backends/          # Pluggable translation backends
│   │   │   ├── __init__.py
│   │   │   ├── anthropic.py   # Claude API backend
│   │   │   └── protocol.py    # Backend protocol
│   │   ├── testing.py         # Public test utilities
│   │   ├── translator.py      # Core translation service
│   │   └── types.py           # TypedDict definitions
│   └── tests/
│       └── test_*.py          # Full coverage tests
│
└── platform_stt/              # EXISTS - No changes needed
```

### Pattern Compliance

Each library follows monorepo patterns:

| Pattern | Description |
|---------|-------------|
| TypedDict | All data structures use TypedDict, not dataclasses |
| encode/decode/require | Every TypedDict has encode_, decode_, require_ functions |
| _test_hooks.py | Internal DI hooks, underscore prefix = private |
| testing.py | Public test utilities exported for consumers |
| Protocol | All dependencies use Protocol types |
| No Any/cast | Strict typing throughout |
| 100% coverage | Statements and branches |

## Components

### 1. platform_langid (NEW)

Spoken language detection from audio waveforms.

**Model:** Meta MMS-LID (4017 languages)
- HuggingFace: `facebook/mms-lid-4017`
- Input: audio waveform (resampled to 16kHz)
- Output: SpokenLanguageResult TypedDict

**Files:**
- `types.py` - SpokenLanguageResult TypedDict with encode/decode/require
- `_test_hooks.py` - ModelProtocol, ProcessorProtocol, hooks
- `detector.py` - detect_spoken_language() using hooks
- `testing.py` - FakeSpokenLangIdModel for consumers

**Dependencies:** `transformers`, `torch`, `torchaudio`

### 2. platform_stt (EXISTS)

Speech-to-text with explicit language. No changes needed.

```python
from platform_stt import OpenAISttClient

result = client.transcribe(file=audio, language="vi")
```

### 3. platform_translate (NEW)

Text translation with pluggable backends.

**Backends:**
- `AnthropicBackend` - Claude API (default, already have API access)
- Future: DeepL, NLLB-200

**Files:**
- `types.py` - TranslationResult TypedDict with encode/decode/require
- `backends/protocol.py` - TranslationBackendProtocol
- `backends/anthropic.py` - Claude API implementation
- `_test_hooks.py` - Backend factory hooks
- `translator.py` - translate_text() using configured backend
- `testing.py` - FakeTranslationBackend for consumers

**Dependencies:** `anthropic` (already in monorepo)

### 4. grandma-api Updates

Update ServiceContainer and translate route to use pipeline.

**Files to modify:**
- `core/container.py` - Add langid and translate service factories
- `api/routes/translate.py` - Three-step pipeline
- `api/schemas/translate.py` - Add detected_language to response

## Updated grandma-api Flow

```python
# 1. Detect language from audio
result = langid.detect_spoken_language(audio_bytes)
lang = result["language"]
confidence = result["confidence"]

# 2. Transcribe with explicit language
source_text = stt.transcribe(audio, language=lang)["text"]

# 3. Translate if not English
if lang != "en":
    translation = translate.translate_text(source_text, source=lang, target="en")
    english_text = translation["text"]
else:
    english_text = source_text
```

## Implementation Order

1. `platform_langid` - MMS-LID integration with transformers
2. `platform_translate` - Claude API backend
3. Update `grandma-api` ServiceContainer and route
4. Full test coverage for all components
5. Integration testing

## Data Requirements for Fine-Tuning (Future)

| Task | Data Needed |
|------|-------------|
| Language ID | Audio + language label |
| Transcription | Audio + source text |
| Translation | Source text + English text |

**Existing datasets:**
- Common Voice (ASR, 100+ languages)
- CoVoST 2 (speech translation, 21 languages)
- FLEURS (ASR + translation, 102 languages)
- NLLB dataset (text translation, 200 languages)
