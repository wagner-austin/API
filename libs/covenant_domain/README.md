# covenant-domain

Pure business logic for covenant monitoring: TypedDict models, rule engine, and feature extraction.

## Overview

This library contains the core domain logic for Covenant Radar with zero IO dependencies:

- **models.py** - TypedDict definitions for Deal, Covenant, Measurement, CovenantResult
- **types.py** - Type aliases and Literal types
- **decode.py** - JSON dict to TypedDict decoders
- **encode.py** - TypedDict to JSON dict encoders
- **formula_parser.py** - Safe arithmetic expression evaluator (shunting-yard)
- **rules.py** - Covenant evaluation and status classification
- **features.py** - ML feature extraction

## Design Principles

- All types are `TypedDict` (no dataclasses, no classes with state)
- All monetary values stored as scaled integers (`* 1_000_000`)
- No `Any`, `cast`, `type: ignore`, or `.pyi` stubs
- No `try/except` in core logic - exceptions propagate
- 100% test coverage (statement + branch)

## Usage

```python
from covenant_domain.models import Deal, Covenant, Measurement
from covenant_domain.rules import evaluate_covenant_for_period
from covenant_domain.decode import decode_deal
```

## Development

```bash
make check  # lint + test with 100% coverage
```
