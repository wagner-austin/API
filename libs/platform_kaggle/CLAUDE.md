# AI Instructions for platform_kaggle

## What This Library Does

- Fetches competitions from Kaggle API
- Filters by tags, categories, and interests
- Matches competitions against codebase capabilities
- Scores fit (strong_fit, good_fit, stretch, new_territory)

## Do NOT Use This Directly

**Use the opportunity-radar-api instead:**

```bash
# Find competitions matching codebase
curl "https://opportunity-radar-api-production.up.railway.app/kaggle/competitions"

# Filter by tags
curl "https://opportunity-radar-api-production.up.railway.app/kaggle/competitions?tags=tabular&tags=classification"

# Exclude tags
curl "https://opportunity-radar-api-production.up.railway.app/kaggle/competitions?exclude=image&exclude=computer-vision"

# Active only (default is true)
curl "https://opportunity-radar-api-production.up.railway.app/kaggle/competitions?active_only=true"

# Without codebase matching
curl "https://opportunity-radar-api-production.up.railway.app/kaggle/competitions?match_codebase=false"

# Get specific competition
curl "https://opportunity-radar-api-production.up.railway.app/kaggle/competitions/titanic"
```

## When To Use This Library Directly

Only use this library directly when:
1. Writing tests for opportunity-radar-api
2. Extending competition matching logic
3. The API is unavailable

---

# Competition Analysis Guide

When analyzing competitions, use this guide to assess fit.

## Codebase Capabilities

The monorepo has these ML/data capabilities (auto-detected from pyproject.toml):

### ML Backends
- **LightGBM** - Large-scale tabular data, gradient boosting
- **XGBoost** - Tabular classification/regression
- **PyTorch** - Deep learning, neural networks
- **scikit-learn** - General ML algorithms
- **TorchVision** - Computer vision, image classification
- **Hugging Face Transformers** - NLP, LLMs, text processing

### Data Processing
- **Pandas/CSV/Parquet** - Tabular data handling
- **Hugging Face Datasets** - ML dataset loading

### Specialized
- **OpenAI Whisper** - Speech-to-text transcription
- **FastText** - Language identification
- **Optuna** - Hyperparameter optimization

### Infrastructure (detected)
- **Datadog APM** - Distributed tracing
- **Confluent Kafka** - Event streaming
- **Redis** - Caching/messaging
- **OpenAI API** - LLM integration
- **Google Cloud** - GCS, BigQuery

## How to Analyze a Competition

### Step 1: Get Competition Details

```bash
curl "https://opportunity-radar-api-production.up.railway.app/kaggle/competitions/{ref}"
```

### Step 2: Assess Capability Match

**Strong fit (70-100%)** - Direct capabilities:
- Tabular classification/regression -> LightGBM, XGBoost
- Text classification/NLP -> Transformers
- Image classification -> TorchVision
- Speech tasks -> Whisper

**Moderate fit (40-70%)** - Related capabilities:
- Time series -> PyTorch (would need models)
- Object detection -> TorchVision (need specific models)

**Stretch (20-40%)** - Foundations only:
- 3D segmentation -> PyTorch exists but no 3D experience
- Reinforcement learning -> No RL code

**New territory (<20%)** - Missing core:
- Mobile app development
- Edge deployment
- Proprietary models required

### Step 3: Format Recommendation

```
**[Competition Name]**
- Fit: X% (strong_fit/good_fit/stretch/new_territory)
- Deadline: YYYY-MM-DD
- Prize: $X

Why this score:
- [What we have that matches]
- [What gaps exist]

Recommendation: [Compete/Skip/Consider if...]
```

## Testing Fakes (for lib development)

```python
from platform_kaggle import (
    FakeKaggleClient,
    make_fake_competition,
    make_fake_profile,
    make_fake_capability,
    hooks,
    reset_hooks,
)

# Override client for testing
hooks.kaggle_client = lambda: FakeKaggleClient(competitions=(...))
```
