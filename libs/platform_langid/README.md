# platform-langid

Spoken language identification from audio using Meta's MMS-LID model.

## Installation

```toml
[tool.poetry.dependencies]
platform-langid = { path = "../libs/platform_langid", develop = true }
```

## Quick Start

```python
from platform_langid import detect_spoken_language, SpokenLanguageResult

# Detect language from audio bytes (16-bit PCM)
result = detect_spoken_language(audio_bytes, sample_rate=16000)
print(result["language"])    # "vi"
print(result["confidence"])  # 0.94
print(result["model_id"])    # "facebook/mms-lid-4017"
```

### Repeated Detection

For repeated detections, create a detector instance to avoid reloading the model:

```python
from platform_langid import create_detector, default_detector_config

config = default_detector_config()
detector = create_detector(config)

for audio in audio_files:
    result = detector.detect(audio, sample_rate=16000)
    print(result["language"])
```

### Custom Configuration

```python
from platform_langid import DetectorConfig, create_detector

config = DetectorConfig(
    model_id="facebook/mms-lid-4017",
    device="cuda",  # Use GPU
    confidence_threshold=0.5,  # Return "und" if below threshold
)
detector = create_detector(config)
```

## Type Definitions

All types use TypedDict with strict typing.

### SpokenLanguageResult

```python
from platform_langid import (
    SpokenLanguageResult,
    encode_spoken_language_result,
    decode_spoken_language_result,
    require_spoken_language_result,
)

result = SpokenLanguageResult(
    language="vi",
    confidence=0.94,
    model_id="facebook/mms-lid-4017",
)

# Encode/decode for JSON serialization
encoded = encode_spoken_language_result(result)
decoded = decode_spoken_language_result(encoded)
```

### DetectorConfig

```python
from platform_langid import (
    DetectorConfig,
    encode_detector_config,
    decode_detector_config,
    require_detector_config,
    default_detector_config,
)

config = default_detector_config()
# or
config = DetectorConfig(
    model_id="facebook/mms-lid-4017",
    device="cpu",
    confidence_threshold=0.0,
)
```

### AudioInput

```python
from platform_langid import (
    AudioInput,
    encode_audio_input,
    decode_audio_input,
    require_audio_input,
)

audio = AudioInput(
    waveform=audio_bytes,
    sample_rate=16000,
    format="pcm_s16le",
)
```

## Constants

```python
from platform_langid import (
    DEFAULT_MODEL_ID,           # "facebook/mms-lid-4017"
    DEFAULT_DEVICE,             # "cpu"
    DEFAULT_CONFIDENCE_THRESHOLD,  # 0.0
    TARGET_SAMPLE_RATE,         # 16000
)
```

## Testing Utilities

Public test utilities for downstream services:

```python
from platform_langid import _test_hooks
from platform_langid.testing import (
    FakeSpokenLanguageDetector,
    FakeModel,
    FakeProcessor,
    FakeAudioLoader,
    make_fake_detector_factory,
    reset_hooks,
)

# Set up fakes for testing
_test_hooks.detector_factory = make_fake_detector_factory(
    language="vi",
    confidence=0.95,
)

# Run tests...

# Reset to production after test
reset_hooks()
```

## Development

```bash
make lint   # guard checks, ruff, mypy
make test   # pytest with coverage
make check  # lint + test
```

## Requirements

- Python 3.11+
- torch ^2.0.0
- torchaudio ^2.0.0
- transformers ^4.30.0
- platform-core (monorepo shared library)

## Quality Standards

- 100% test coverage enforced (statements and branches)
- mypy strict mode with all `disallow_any_*` flags
- No `Any`, `cast`, `type: ignore`, `.pyi` stubs
- Guard checks on all src, tests
- Google-style docstrings
