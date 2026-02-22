# platform-translate

Text translation with pluggable backends (Anthropic Claude, DeepL, NLLB).

## Installation

```toml
[tool.poetry.dependencies]
platform-translate = { path = "../libs/platform_translate", develop = true }
```

## Quick Start

```python
from platform_translate import translate_text, TranslatorConfig

config = TranslatorConfig(
    backend="anthropic",
    api_key="sk-ant-...",
    model="claude-3-haiku-20240307",
)

result = translate_text("Xin chào", "vi", "en", config)
print(result["text"])  # "Hello"
```

### Repeated Translations

For repeated translations, create a translator instance:

```python
from platform_translate import create_translator, TranslatorConfig

config = TranslatorConfig(
    backend="anthropic",
    api_key="sk-ant-...",
    model="claude-3-haiku-20240307",
)

translator = create_translator(config)
result1 = translator.translate("Xin chào", "vi", "en")
result2 = translator.translate("Cảm ơn", "vi", "en")
```

### Default Configuration

```python
from platform_translate import default_translator_config

config = default_translator_config(api_key="sk-ant-...")
# Uses anthropic backend with claude-3-haiku
```

## Backends

### Anthropic (Default)

Uses Claude API for high-quality translation.

```python
from platform_translate.backends import AnthropicBackend

config = TranslatorConfig(
    backend="anthropic",
    api_key="sk-ant-...",
    model="claude-3-haiku-20240307",
)
```

### Future Backends

- **DeepL** - Coming soon
- **NLLB-200** - Coming soon (local, 200 languages)

## Type Definitions

All types use TypedDict with strict typing.

### TranslationRequest

```python
from platform_translate import (
    TranslationRequest,
    encode_translation_request,
    decode_translation_request,
    require_translation_request,
)

request = TranslationRequest(
    text="Hello",
    source_language="en",
    target_language="es",
)
```

### TranslationResult

```python
from platform_translate import (
    TranslationResult,
    encode_translation_result,
    decode_translation_result,
    require_translation_result,
)

# Result from translation
result = TranslationResult(
    text="Hola",
    source_language="en",
    target_language="es",
    backend="anthropic",
)
```

### TranslatorConfig

```python
from platform_translate import (
    TranslatorConfig,
    encode_translator_config,
    decode_translator_config,
    require_translator_config,
)

config = TranslatorConfig(
    backend="anthropic",
    api_key="sk-...",
    model="claude-3-haiku-20240307",
)
```

## Constants

```python
from platform_translate import (
    DEFAULT_BACKEND,  # "anthropic"
    DEFAULT_MODEL,    # "claude-3-haiku-20240307"
)
```

## Testing Utilities

Public test utilities for downstream services:

```python
from platform_translate import _test_hooks
from platform_translate.testing import (
    FakeTranslationBackend,
    FakeAnthropicClient,
    make_fake_backend_factory,
    reset_hooks,
)

# Set up fakes for testing
_test_hooks.backend_factory = make_fake_backend_factory(
    translated_text="Hello",
    backend_id="fake",
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
- anthropic ^0.39.0
- platform-core (monorepo shared library)

## Quality Standards

- 100% test coverage enforced (statements and branches)
- mypy strict mode with all `disallow_any_*` flags
- No `Any`, `cast`, `type: ignore`, `.pyi` stubs
- Guard checks on all src, tests
- Google-style docstrings
