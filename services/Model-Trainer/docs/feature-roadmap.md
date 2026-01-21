# Model-Trainer Feature Roadmap

## Priority Features

### 1. Training Duration in Manifest
**Status:** Implemented

Add elapsed training time to `manifest.json`:
- `training_duration_sec`: Total training time in seconds
- `started_at`: ISO timestamp when training began
- `completed_at`: ISO timestamp when training finished

**Files modified:**
- `src/model_trainer/core/services/training/base_trainer.py` - capture start/end times
- `src/model_trainer/worker/manifest.py` - add fields to manifest schema
- `src/model_trainer/infra/persistence/models.py` - added `TrainingManifestTiming` TypedDict
- `src/model_trainer/core/_test_hooks.py` - added `time_monotonic` and `datetime_utcnow_iso` hooks

---

### 2. Throughput/Memory Profiling
**Status:** Implemented

Add performance metrics to manifest:
- `peak_gpu_memory_mb`: Maximum GPU memory used (null for CPU training)
- `avg_samples_per_sec`: Average throughput during training
- `total_tokens_processed`: Total tokens seen during training

**Files modified:**
- `src/model_trainer/core/services/training/base_trainer.py` - collect metrics
- `src/model_trainer/worker/manifest.py` - add fields
- `src/model_trainer/infra/persistence/models.py` - added `TrainingManifestPerformance` TypedDict
- `src/model_trainer/core/_test_hooks.py` - added `gpu_max_memory_allocated` and `gpu_reset_peak_memory_stats` hooks

---

### 3. Model Size / Parameter Count
**Status:** Implemented

Add model metadata to manifest:
- `param_count`: Total trainable parameters
- `model_size_mb`: Size of saved model on disk
- `vocab_size`: Tokenizer vocabulary size

**Files modified:**
- `src/model_trainer/core/services/training/base_trainer.py` - compute after model save
- `src/model_trainer/worker/manifest.py` - add fields
- `src/model_trainer/infra/persistence/models.py` - added `TrainingManifestModelInfo` TypedDict
- `src/model_trainer/core/_test_hooks.py` - added `count_model_parameters` and `get_directory_size_bytes` hooks

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
