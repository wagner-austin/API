# platform-stt

Speech-to-text library with Whisper API, audio chunking, parallel transcription, segment merging, and language detection.

## Installation

```toml
[tool.poetry.dependencies]
platform-stt = { path = "../libs/platform_stt", develop = true }
```

## Quick Start

```python
from pathlib import Path
from platform_stt import OpenAISttClient, format_srt, write_srt

# Transcribe audio and generate SRT subtitles
client = OpenAISttClient(api_key="...")
with open("audio.mp3", "rb") as f:
    result = client.transcribe(file=f, language="en")

# Generate SRT file
srt_content = format_srt(result["segments"])
write_srt(srt_content, Path("subtitles.srt"))

# Translation to English
with open("audio.mp3", "rb") as f:
    result = client.translate(file=f)
```

### Additional Features

```python
from platform_stt import AudioChunker, TranscriptMerger

# Chunk large audio files (for files > 25MB)
chunker = AudioChunker()
chunks = chunker.chunk_audio("large_file.mp3", total_duration=3600.0, estimated_mb=100.0)

# Merge transcripts from chunks
merger = TranscriptMerger()
merged = merger.merge(segments_list, chunk_start_times)
```

## OpenAI Whisper Client

HTTP client for OpenAI Whisper API with transcription and translation support.

```python
from platform_stt import OpenAISttClient, VerboseResponse

client = OpenAISttClient(
    api_key="sk-...",
    base_url="https://api.openai.com/v1",  # Optional
    timeout_seconds=120.0,  # Optional
)

# Transcribe with verbose output
with open("audio.mp3", "rb") as f:
    response: VerboseResponse = client.transcribe(
        file=f,
        language="vi",  # Source language
        filename="audio.mp3",  # Optional filename
        prompt="",  # Optional prompt
    )
print(response["text"])
print(response["segments"])

# Translate to English
with open("audio.mp3", "rb") as f:
    response = client.translate(file=f)
```

## Audio Chunking

Split large audio files at silence points for parallel transcription.

```python
from platform_stt import AudioChunker, AudioChunk

chunker = AudioChunker(
    target_chunk_mb=20.0,  # Target chunk size in MB
    max_chunk_duration_seconds=600.0,  # Max 10 minutes per chunk
    silence_threshold_db=-40.0,  # Silence detection threshold
    silence_duration_seconds=0.5,  # Minimum silence duration
    ffmpeg_path="ffmpeg",  # Path to ffmpeg
    ffprobe_path="ffprobe",  # Path to ffprobe
)

# Chunk audio file
chunks: list[AudioChunk] = chunker.chunk_audio("/path/to/audio.mp3")

for chunk in chunks:
    print(f"{chunk['path']}: {chunk['start_seconds']}s - {chunk['duration_seconds']}s")
```

### ChunkerConfig

Use `ChunkerConfig` TypedDict for passing config as data:

```python
from platform_stt import ChunkerConfig, decode_chunker_config

config: ChunkerConfig = decode_chunker_config(
    {
        "target_chunk_mb": 20.0,
        "max_chunk_duration_seconds": 600.0,
        "silence_threshold_db": -40.0,
        "silence_duration_seconds": 0.5,
    }
)
```

## Parallel Transcription

Transcribe chunks in parallel with bounded concurrency.

```python
from platform_stt import ParallelTranscriber, OpenAISttClient

client = OpenAISttClient(api_key="...")
transcriber = ParallelTranscriber(
    client=client,
    max_workers=4,  # Max parallel workers
    max_retries=3,  # Retry on failure
)

# Transcribe all chunks
results = transcriber.transcribe_chunks(
    chunks=chunks,
    language="vi",
)

for chunk, response in results:
    print(f"{chunk['start_seconds']}s: {response['text']}")
```

## Segment Merging

Merge transcript segments from multiple chunks with time offset correction.

```python
from platform_stt import TranscriptMerger, TranscriptSegment, merge_segment_text

merger = TranscriptMerger()

# Merge segments from chunked transcripts
all_segments: list[list[TranscriptSegment]] = [
    # Segments from chunk 1 (starts at 0s)
    [{"text": "Hello", "start": 0.0, "duration": 1.0}],
    # Segments from chunk 2 (starts at 30s)
    [{"text": "World", "start": 0.0, "duration": 1.0}],
]
chunk_start_times = [0.0, 30.0]

merged = merger.merge(all_segments, chunk_start_times)

# Simple text concatenation
text = merge_segment_text(merged)
```

## Language Detection

Detect language using FastText lid.218e or lid.176 models.

```python
from platform_stt import detect_language, is_language, load_langid_model

# Load model (auto-downloads if not present)
model = load_langid_model("/path/to/data", prefer_218e=True)

# Detect language
result = detect_language("Xin chao the gioi", model)
print(result["language"])  # "vi"
print(result["confidence"])  # 0.95

# Check specific language
if is_language("Hello world", "en", model, threshold=0.8):
    print("Text is English")
```

## Whisper Parse Utilities

Convert OpenAI SDK responses to typed structures.

```python
from platform_stt import (
    to_verbose_response,
    convert_verbose_to_segments,
    VerboseResponse,
    TranscriptSegment,
)

# Convert SDK response to typed VerboseResponse
verbose: VerboseResponse = to_verbose_response(sdk_response)

# Convert to TranscriptSegments
segments: list[TranscriptSegment] = convert_verbose_to_segments(verbose)
```

## SRT Subtitle Generation

Generate SRT subtitle files from Whisper transcription segments.

```python
from pathlib import Path
from platform_stt import OpenAISttClient, format_srt, write_srt

# Transcribe audio
client = OpenAISttClient(api_key="...")
with open("video_audio.mp3", "rb") as f:
    response = client.transcribe(file=f, language="en")

# Generate SRT content
srt_content = format_srt(response["segments"])

# Write to file
write_srt(srt_content, Path("subtitles.srt"))
```

### SRT Output Format

```
1
00:00:00,000 --> 00:00:04,000
First sentence of your video.

2
00:00:04,500 --> 00:00:08,000
Second sentence here.
```

### SRT Functions

| Function | Description |
|----------|-------------|
| `format_srt(segments)` | Convert `VerboseSegment[]` to full SRT string |
| `format_srt_entry(entry)` | Format single `SrtEntry` block |
| `format_timestamp(seconds)` | Convert seconds to `HH:MM:SS,mmm` |
| `write_srt(content, path)` | Write SRT content to file |
| `segments_to_srt_entries(segments)` | Convert segments to `SrtEntry[]` |

### SrtEntry TypedDict

```python
from platform_stt import (
    SrtEntry,
    encode_srt_entry,
    decode_srt_entry,
    require_srt_entry,
)

entry = SrtEntry(
    index=1,
    start_seconds=0.0,
    end_seconds=2.5,
    text="Hello world",
)

# Encode/decode for JSON serialization
encoded = encode_srt_entry(entry)
decoded = decode_srt_entry(encoded)
```

## Type Definitions

All types use TypedDict with strict typing. No `Any` types allowed.

### Core Types

| Type | Description |
|------|-------------|
| `TranscriptSegment` | Segment with text, start, duration |
| `VerboseSegment` | Whisper segment with text, start, end |
| `VerboseResponse` | Full Whisper response with segments |
| `AudioChunk` | Chunk with path, start, duration, size |
| `ChunkerConfig` | Chunker configuration |
| `LanguageDetectionResult` | Language, confidence, script |
| `SrtEntry` | SRT subtitle entry with index, timestamps, text |

### Encode/Decode Functions

Each type has corresponding functions:

```python
from platform_stt import (
    # Encode to dict for JSON serialization
    encode_transcript_segment,
    encode_audio_chunk,
    encode_verbose_response,
    # Decode from dict with validation
    decode_transcript_segment,
    decode_audio_chunk,
    decode_verbose_response,
    # Validate arbitrary JSONValue
    require_transcript_segment,
    require_audio_chunk,
    require_verbose_response,
)
```

### Validation Functions

```python
from platform_stt import validate_whisper_language, validate_whisper_task

# Validate language code
lang = validate_whisper_language("vi")  # Returns "vi" or raises ValueError

# Validate task
task = validate_whisper_task("transcribe")  # Returns "transcribe" or raises ValueError
```

## Constants

```python
from platform_stt import WHISPER_SUPPORTED_LANGUAGES

if "vi" in WHISPER_SUPPORTED_LANGUAGES:
    print("Vietnamese is supported")
```

## Testing Utilities

Public test utilities for downstream services. Uses the hook pattern for dependency injection.

```python
from platform_stt import _test_hooks
from platform_stt.testing import (
    FakeAudioChunker,
    FakeLangIdModel,
    FakeSTTClient,
    FakeSubprocessResult,
    FakeSubprocessRun,
    FakeWriteTextFile,
    make_fake_audio_chunker_factory,
    make_fake_langid_model_factory,
    make_fake_subprocess_run,
    reset_hooks,
    set_production_hooks,
)

# Create fake STT client for testing
fake_client = FakeSTTClient(
    response=VerboseResponse(text="Test", language="en", segments=[]),
)

# Create fake language ID model
fake_model = FakeLangIdModel(label="__label__vi", confidence=0.95)

# Install fake subprocess runner
fake_subprocess = make_fake_subprocess_run(FakeSubprocessResult(returncode=0, stdout=b"output"))

# Reset hooks after tests
reset_hooks()
```

### Hook Pattern

Production code uses hooks from `_test_hooks`, tests override with fakes:

```python
from platform_stt import _test_hooks
from platform_stt.testing import FakeSTTClient

# Override hook in test
_test_hooks.openai_client_factory = lambda **kw: FakeSTTClient()

# Reset to production after test
from platform_stt.testing import reset_hooks

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
- ffmpeg and ffprobe for audio chunking
- numpy < 2.0 (for fasttext compatibility)
- openai ^2.8.0 (for Whisper API)
- fasttext-wheel ^0.9.2 (for language detection)
- platform-core (monorepo shared library)

## Quality Standards

- 100% test coverage enforced (statements and branches)
- mypy strict mode with all `disallow_any_*` flags
- No `Any`, `cast`, `type: ignore`, `.pyi` stubs
- Guard checks on all src, tests, scripts
- Google-style docstrings
