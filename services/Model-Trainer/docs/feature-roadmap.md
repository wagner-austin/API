# Model-Trainer Feature Roadmap

## Priority Features

### 1. Training Duration in Manifest
**Status:** Not Implemented

Add elapsed training time to `manifest.json`:
- `training_duration_sec`: Total training time in seconds
- `started_at`: ISO timestamp when training began
- `completed_at`: ISO timestamp when training finished

**Files to modify:**
- `src/model_trainer/worker/train_job.py` - capture start/end times
- `src/model_trainer/worker/manifest.py` - add fields to manifest schema
- `src/model_trainer/infra/persistence/models.py` - update TrainingRunRecord

---

### 2. Throughput/Memory Profiling
**Status:** Not Implemented

Add performance metrics to manifest:
- `peak_gpu_memory_mb`: Maximum GPU memory used
- `avg_samples_per_sec`: Average throughput during training
- `total_tokens_processed`: Total tokens seen during training

**Files to modify:**
- `src/model_trainer/core/services/training/base_trainer.py` - collect metrics
- `src/model_trainer/worker/train_job.py` - aggregate and save
- `src/model_trainer/worker/manifest.py` - add fields

---

### 3. Model Size / Parameter Count
**Status:** Not Implemented

Add model metadata to manifest:
- `param_count`: Total trainable parameters
- `model_size_mb`: Size of saved model on disk
- `vocab_size`: Tokenizer vocabulary size

**Files to modify:**
- `src/model_trainer/worker/train_job.py` - compute after model creation
- `src/model_trainer/worker/manifest.py` - add fields

---

## Current Training Limitations

| Issue | Current | Target |
|-------|---------|--------|
| Context length | 256 tokens | 512-1024 tokens |
| Corpus | 274MB cached file | OSCAR/CulturaX streaming |
| Chat understanding | None (trained from scratch) | Fine-tune from pretrained |

## Corpus Sources Available (via turkic-api)

- `stream_oscar(lang)` - OSCAR-2301 web crawl
- `stream_culturax(lang)` - CulturaX (mC4 + OSCAR deduplicated)
- `stream_wikipedia_xml(lang)` - Wikipedia dumps

## Notes

- Distributed GPU training not needed for current use case
- BPE tokenizer decode fix completed (Metaspace pre-tokenizer + decoder)
- Chat infrastructure works; model just needs better training data
