# Kaggle Competition Analysis Guide

This document helps Claude Code agents analyze Kaggle competitions against the codebase capabilities.

## Quick Start

Run the competition finder script:
```bash
cd libs/platform_kaggle
poetry run python scripts/find_competitions.py
```

This outputs:
1. Codebase capabilities (what we can do)
2. Active competitions with full descriptions

Then analyze each competition for fit.

## Codebase Capabilities

The monorepo has these ML/data capabilities:

### ML Backends
- **LightGBM** - Large-scale tabular data, gradient boosting (libs/cleargbm)
- **XGBoost** - Tabular classification/regression
- **PyTorch** - Deep learning, neural networks
- **scikit-learn** - General ML algorithms
- **TorchVision** - Computer vision, image classification
- **Hugging Face Transformers** - NLP, LLMs, text processing

### Data Processing
- **Pandas/CSV/Parquet** - Tabular data handling
- **Hugging Face Datasets** - ML dataset loading
- **Tokenization** - Fast tokenizers, SentencePiece

### Specialized
- **OpenAI Whisper** - Speech-to-text transcription (platform_stt)
- **FastText** - Language identification (turkic-api)
- **Transliteration** - Script conversion between writing systems
- **Optuna** - Hyperparameter optimization

### Domain Expertise
- **Loan covenant monitoring** - Financial compliance, breach prediction
- **Multilingual NLP** - Turkic languages, translation

## How to Analyze a Competition

### Step 1: Read the Description
Look for:
- **Task type**: Classification, regression, segmentation, generation, etc.
- **Data format**: Tabular, images, text, audio, 3D, time series
- **Evaluation metric**: Custom metrics may need special handling
- **Hardware requirements**: GPU, TPU, memory constraints
- **Explicit tool requirements**: "Must use X model"

### Step 2: Assess Capability Match

**Strong fit (70-100%)** - We have direct capabilities:
- Tabular classification/regression → LightGBM, XGBoost, sklearn
- Text classification/NLP → Transformers, tokenizers
- Image classification → TorchVision, PyTorch
- Speech tasks → Whisper

**Moderate fit (40-70%)** - We have related capabilities:
- Time series → PyTorch (would need to build models)
- Object detection → TorchVision base (would need specific models)
- LLM fine-tuning → Transformers (have infra, need compute)

**Stretch (20-40%)** - We have foundations only:
- 3D segmentation → PyTorch exists but no 3D experience
- Reinforcement learning → PyTorch exists but no RL code
- Video processing → No current capabilities

**New territory (<20%)** - Missing core capabilities:
- Mobile app development
- Edge/on-device deployment
- Specific proprietary models required

### Step 3: Identify Gaps

Be specific about what's missing:
- "Need 3D segmentation experience - have 2D only"
- "Need domain knowledge in X"
- "Would need to learn Y framework"
- "Compute requirements exceed typical resources"

### Step 4: Give Recommendation

Format:
```
**[Competition Name]**
- Fit: X% (strong_fit/good_fit/stretch/new_territory)
- Deadline: YYYY-MM-DD
- Prize: $X

Why this score:
- [What we have that matches]
- [What gaps exist]
- [Effort estimate to bridge gaps]

Recommendation: [Compete/Skip/Consider if...]
```

## Example Analysis

**AI Mathematical Olympiad - Progress Prize 3**
- Fit: 65% (good_fit)
- Deadline: 2026-04-15
- Prize: $2.2M

Why this score:
- HAVE: Hugging Face Transformers for running open-source LLMs
- HAVE: Tokenization and text processing infrastructure
- HAVE: PyTorch for any custom model work
- GAP: Mathematical reasoning is the actual challenge, not tooling
- GAP: May need significant prompt engineering / fine-tuning expertise

Recommendation: Compete - infrastructure is solid, challenge is algorithmic

---

**Vesuvius Challenge - Surface Detection**
- Fit: 35% (stretch)
- Deadline: 2026-02-13
- Prize: $200K

Why this score:
- HAVE: PyTorch and TorchVision as foundation
- HAVE: General deep learning experience
- GAP: No 3D CT scan segmentation experience
- GAP: No medical imaging domain knowledge
- GAP: Would need to learn volumetric segmentation from scratch

Recommendation: Skip unless specifically interested in learning 3D segmentation

## Key Questions to Answer

1. **Can we load and process the data?** (format compatibility)
2. **Do we have models for this task type?** (or close enough to adapt)
3. **Do we understand the domain?** (or can we learn quickly)
4. **Do we have the compute?** (GPU requirements)
5. **Is there a hard blocker?** (specific model required, mobile app, etc.)
