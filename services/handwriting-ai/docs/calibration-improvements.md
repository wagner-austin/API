# Calibration System Improvements

## Document Status
- **Status**: Partially Implemented
- **Author**: System Architecture Team
- **Date**: 2025-11-13
- **Last Updated**: 2025-11-27
- **Version**: 1.1

## Executive Summary

This document proposes architectural improvements to the calibration system to maximize memory utilization and reduce calibration time. Current production deployments show only ~19% memory utilization during training on 7.6GB containers, indicating the calibration system is overly conservative and not discovering optimal batch sizes.

### Implementation Status

| Feature | Status | Location |
|---------|--------|----------|
| Result streaming | ✅ Implemented | `runner.py:139` `_emit_result_file()`, `measure.py` `on_improvement` callback |
| Dataset pickling | ✅ Implemented | `runner.py:454+` `_MNISTRawDataset.__reduce__()` serializes only `(root, train)` |
| Stable multiprocessing context | ✅ Implemented | `measure.py:361` `_resolve_worker_context()` returns forkserver/spawn |
| Headroom-based expansion (simplified) | ✅ Implemented | `measure.py:192` `_expand_upper_bound_if_headroom()`, `HEADROOM_TARGET=60.0` |
| Phase 1A/B/C (full exponential expansion) | ❌ Pending | Described below - full three-phase adaptive algorithm |
| Phase 2 (memory-aware sample counts) | ❌ Pending | `calibration_samples` is static at 100, no tier-based computation |

**Current state**: The simplified headroom expansion doubles the upper bound when memory usage is below 60% threshold. The full three-phase algorithm (cap verification → exponential expansion → precision binary search) and tier-based sample counts remain as future improvements.

## Problem Statement

### Current Behavior

The calibration system currently exhibits three critical inefficiencies:

1. **Conservative Batch Size Discovery**: Binary search starts from batch_size=1 and searches upward to a computed cap (e.g., 256). With only 8 sample batches, the search terminates prematurely, finding batch_size=128 at 20% memory usage instead of continuing to discover the actual maximum.

2. **Fixed Sample Count**: The calibration_samples parameter is hardcoded to 8 batches regardless of memory tier. This provides insufficient measurements for larger memory environments to reach steady-state memory usage and discover true capacity.

3. **Inefficient Pickle Serialization**: When DataLoader spawns worker processes, it pickles the entire MNISTRawDataset containing 60,000 raw images (approximately 47MB). This exceeds pipe buffer limits (64KB), causing pickle truncation errors and worker spawn failures.

### Impact Analysis

**Production Evidence** (Railway 7.6GB container):
- Calibration finds: batch_size=128, memory=19.4%
- Training runs at: batch_size=256, memory=19.4%
- Wasted capacity: 73% of available memory unused
- Theoretical maximum: batch_size could be 800+ based on linear extrapolation

**Performance Implications**:
- Training throughput: 165 samples/sec (current) vs ~600+ samples/sec (theoretical)
- Training time: 15 epochs takes 2.5 hours vs potentially <45 minutes
- Resource utilization: $7.45/month for underutilized container

## Root Cause Analysis

### Issue 1: Algorithmic Inefficiency

The current binary search algorithm in measure.py:_measure_candidate:

- Initializes search range: bs_lo=1, bs_hi=computed_cap
- Uses classic binary search: mid = (lo + hi) // 2
- Problem: With samples=8, only 3-4 iterations complete before time limit
- Result: Search terminates at first successful mid value, never exploring upper range

**Timeline Example** (simplified):
1. Try bs=128 (mid of 1-256): Success at 20% memory → new range [129,256]
2. Try bs=192 (mid of 129-256): Success at 25% memory → new range [193,256]
3. Try bs=224 (mid of 193-256): Success at 28% memory → new range [225,256]
4. Calibration timeout after 45s → return bs=128 as "safe"

The algorithm never tests if 256 (the computed cap) even works, let alone if higher values are viable.

### Issue 2: Static Configuration

The calibration_samples field in train_config.py is statically set to 8. This value:

- Was chosen for <1GB containers where fast calibration is critical
- Does not scale with available memory
- Provides insufficient batches for optimizer state initialization (Adam requires 2x model params for momentum/variance)
- Insufficient for DataLoader worker warm-up and prefetch buffer filling

### Issue 3: Pickle Data Inefficiency (Resolved)

The PreprocessDataset.__reduce__ method in dataset.py:147-151 pickles self._base directly. For MNIST calibration, _base is _MNISTRawDataset containing:

- 60,000 x 784-byte images = ~47MB raw data
- Pickle overhead adds ~10-15%
- Total pickle size: ~52MB

DataLoader worker spawn via multiprocessing.spawn uses a pipe with 64KB buffer on most systems. The 52MB pickle:
- Exceeds buffer by 800x
- Causes UnpicklingError: pickle data was truncated
- Requires retry logic, slowing calibration
- May fail intermittently under load

## Proposed Solution Architecture

### Phase 1: Adaptive Batch Size Search Algorithm

Replace the current binary search with a three-phase adaptive algorithm:

**Phase 1A: Cap Verification**
- Test the computed cap (e.g., 256) immediately
- If successful: proceed to expansion phase
- If failed: fall back to binary search [1, cap-1]
- Rationale: Validate memory-based estimate before exploration

**Phase 1B: Exponential Expansion**
- Starting from verified cap, test 2x, 4x, 8x multiples
- Continue while peak_memory_percent < headroom_threshold (default: 60%)
- Stop expansion on first failure or memory threshold breach
- Rationale: Discover true maximum using available headroom

**Phase 1C: Precision Binary Search**
- Search range: [last_success, first_failure]
- Use traditional binary search for final precision
- Rationale: Optimize exact batch size within discovered bounds

**Algorithm Properties**:
- Worst-case iterations: O(log cap) + O(log expansion_factor) = O(log n)
- Best-case: 1 iteration (cap is optimal)
- Average-case: 4-6 iterations vs current 3-4, but finds 2-4x larger batch size
- Memory safety: Maintains threshold-based guards throughout

### Phase 2: Memory-Aware Sample Configuration

Add compute_calibration_samples function to safety.py:

**Tier-Based Sample Counts**:
- <1GB: 8 samples (preserve fast calibration for constrained environments)
- 1-2GB: 12 samples (light exploration)
- 2-4GB: 16 samples (standard exploration)
- >=4GB: 32 samples (thorough exploration with ample time/memory budget)

**Rationale by Tier**:
- <1GB: Minimize calibration overhead; OOM risk high; conservative necessary
- 1-2GB: Balance speed vs accuracy; modest expansion viable
- 2-4GB: Standard workload tier; more samples improve accuracy
- >=4GB: Enterprise tier; maximize resource utilization

**Sample Count Purpose**:
- Batch 1: Cold start (module imports, memory allocation)
- Batches 2-4: Optimizer state initialization
- Batches 5-8: DataLoader worker warm-up, prefetch buffer fill
- Batches 9+: Steady-state measurement, accurate memory profiling

### Phase 3: Efficient Dataset Pickling

Add custom __reduce__ method to _MNISTRawDataset in runner.py:

**Pickle Strategy**:
- Serialize only MNISTSpec (path: Path, train: bool) instead of raw image data
- Spec pickle size: ~100 bytes
- Workers reload MNIST data from disk on unpickle
- Disk read cost: 250-400ms (one-time per worker spawn)

**Trade-offs Analysis**:
- Pickle size: 52MB → 100 bytes (99.9% reduction)
- Worker spawn time: +250ms per worker (disk I/O)
- Reliability: Eliminates truncation errors
- Net benefit: Faster overall calibration despite I/O cost (no retries)

**Implementation Strategy**:
- Add _MNISTRawDataset.__reduce__ returning (rebuild_func, (spec,))
- Add _rebuild_mnist_raw_dataset(spec) function for unpickling
- Pass spec to _MNISTRawDataset.__init__ and store as self._spec
- Update PreprocessDataset.__reduce__ to use spec-based rebuild

## Implementation Plan

### Module Changes

**src/handwriting_ai/training/calibration/measure.py**:
- Replace _measure_candidate implementation with three-phase algorithm
- Add _test_batch_size helper for single batch size measurement
- Add _exponential_expansion helper for phase 1B
- Add headroom_threshold parameter (default: 60.0)
- Maintain existing type signatures for compatibility
- Add structured logging for each phase

**src/handwriting_ai/training/safety.py**:
- Add compute_calibration_samples(memory_bytes: int | None) -> int
- Add to __all__ exports
- Update module docstring

**src/handwriting_ai/training/train_config.py**:
- Keep calibration_samples field as int (backward compatible)
- Document that default=8 is for <1GB; larger tiers use computed values
- Or: Make calibration_samples: int | None with None meaning "auto-compute"

**src/handwriting_ai/training/calibration/runner.py**:
- Add spec: MNISTSpec | None parameter to _MNISTRawDataset.__init__
- Add _MNISTRawDataset.__reduce__ method
- Add _rebuild_mnist_raw_dataset function (module-level)
- Update _build_mnist_dataset to pass spec to _MNISTRawDataset
- Update PreprocessDataset.__reduce__ in dataset.py to use MNISTSpec

**src/handwriting_ai/training/dataset.py**:
- Update PreprocessDataset.__reduce__ to serialize MNISTSpec instead of raw dataset
- Add _rebuild_preprocess_dataset to accept spec
- Requires coordination with runner.py changes

### Type Safety Requirements

**Strict Type Compliance**:
- All functions maintain -> return type annotations
- No use of Any, cast, or type: ignore
- All parameters have explicit types
- Use Protocol for callbacks (Callable[[CalibrationResult], None])
- dataclass(frozen=True) for all configuration structures

**New Type Additions**:
- BatchSizeSearchPhase = Literal["expansion", "binary_search", "complete"]
- SearchState dataclass to track algorithm state
- Ensure all numeric types are explicitly int | float, not Any

### Testing Strategy

**Unit Tests** (tests/test_calibration_measure.py):
- test_exponential_expansion_finds_maximum: Verify expansion phase discovers 4x cap
- test_expansion_stops_at_threshold: Verify stops when memory > 60%
- test_expansion_stops_on_failure: Verify stops on OOM/RuntimeError
- test_binary_search_precision: Verify final search finds optimal within range
- test_cap_verification_failure: Verify fallback when cap fails immediately
- test_headroom_threshold_configurable: Verify threshold parameter respected

**Unit Tests** (tests/test_safety.py):
- test_compute_calibration_samples_tiers: Verify each tier returns correct value
- test_compute_calibration_samples_none: Verify None returns default
- test_compute_calibration_samples_boundaries: Test tier boundaries exactly

**Unit Tests** (tests/test_calibration_runner.py):
- test_mnist_dataset_pickle_spec: Verify spec-only serialization
- test_mnist_dataset_unpickle_reload: Verify workers reload from disk
- test_preprocess_dataset_pickle_mnist: Verify PreprocessDataset uses spec

**Integration Tests** (tests/test_calibration_runner_subprocess_integration.py):
- test_calibration_with_expansion: Full subprocess test with expansion phase
- test_calibration_memory_headroom: Verify uses >60% memory when available
- test_calibration_no_pickle_truncation: Verify worker spawn succeeds with 2+ workers

**Coverage Requirements**:
- All new functions: 100% statement coverage
- All new branches: 100% branch coverage
- All error paths: Explicit tests with mock failures
- Maintain existing 92%+ overall coverage

### Backward Compatibility

**Configuration**:
- calibration_samples field remains optional with default=8
- Existing configs continue to work unchanged
- Opt-in to auto-compute via calibration_samples=None or remove field

**Behavior**:
- Users with calibration_samples=8 see identical behavior (phase 1A only)
- Users with calibration_samples>=16 benefit from expansion phase
- No breaking changes to calibration API

**Cache Invalidation**:
- Calibration cache signature includes samples count
- Changing samples invalidates cache (expected behavior)
- Calibration re-runs once, then caches for 7 days

### Performance Implications

**Calibration Time**:
- Current: 45s timeout, 3-4 measurements, finds bs=128
- Proposed: 45-60s, 5-8 measurements, finds bs=512+
- Net time: +15s calibration for 3-4x batch size improvement

**Training Time** (15 epochs, 60k samples):
- Current: 165 samples/sec → 91 minutes
- Projected: 600 samples/sec → 25 minutes
- Savings: 66 minutes per training run

**Amortized Cost**:
- Calibration cache TTL: 7 days
- Training frequency: 10-20x per week
- Net savings: 10 hours/week for +1 minute calibration overhead

**Memory Safety**:
- No change to memory guard thresholds
- Expansion phase respects 60% headroom limit
- Binary search maintains existing safety margins
- No increased OOM risk

## Risks and Mitigations

### Risk 1: Expansion Phase OOM

**Risk**: Exponential expansion may OOM before memory guard triggers

**Mitigation**:
- Wrap each expansion attempt in try/except (RuntimeError, MemoryError)
- Immediately stop expansion on exception
- Fall back to last successful batch size
- Log exception type for diagnostics

**Residual Risk**: Low - guard threshold provides 30-40% margin

### Risk 2: Longer Calibration Time

**Risk**: More measurements increase calibration time, delaying training start

**Mitigation**:
- Implement on_improvement callback to stream results
- Parent can start training with intermediate result if timeout approaching
- Stage A timeout remains 45s; expansion phase prioritized over precision
- User can force calibration_samples=8 to restore old behavior

**Residual Risk**: Low - most deployments benefit from better batch size

### Risk 3: Pickle Changes Break Workers

**Risk**: Custom __reduce__ may cause issues with DataLoader worker spawn

**Mitigation**:
- Add _MNISTRawDataset.spec: MNISTSpec | None = None (backward compatible)
- If spec is None, raise RuntimeError("Cannot pickle without spec")
- Integration tests verify worker spawn with 2+ workers
- Test on Windows (spawn) and Linux (fork) multiprocessing contexts

**Residual Risk**: Medium - multiprocessing pickling is notoriously fragile

### Risk 4: Disk I/O Latency in Workers

**Risk**: Workers reloading MNIST from disk may slow batch iteration

**Mitigation**:
- Disk read is one-time per worker spawn (250-400ms)
- Workers are persistent across batches
- MNIST files are small (9.9MB train images, 28.9KB train labels)
- OS filesystem cache makes subsequent reads instant
- Monitor calibration logs for slow _build_mnist_dataset

**Residual Risk**: Low - I/O overhead is negligible vs training time

## Success Criteria

### Quantitative Metrics

1. **Memory Utilization**: Production deployments reach 70-85% memory during training (vs current 19%)
2. **Batch Size Discovery**: 7.6GB tier discovers batch_size >= 512 (vs current 256)
3. **Training Throughput**: Achieve 500+ samples/sec (vs current 165)
4. **Calibration Reliability**: Zero pickle truncation errors in logs
5. **Test Coverage**: Maintain 92%+ overall coverage, 100% for new code

### Qualitative Metrics

1. **Code Quality**: All mypy strict checks pass, no Any types
2. **Maintainability**: Algorithm phases clearly separated with helper functions
3. **Documentation**: Comprehensive inline docstrings explaining each phase
4. **Observability**: Structured logging for each calibration decision
5. **Backward Compatibility**: Existing configs work without changes

## Future Enhancements

### Phase 4: GPU Memory Calibration

Extend algorithm to support CUDA memory calibration:
- Use torch.cuda.memory_allocated() instead of cgroup metrics
- Add GPU memory headroom threshold (default: 70%)
- Test mixed CPU/GPU configurations

### Phase 5: Multi-Stage Batch Size

Support different batch sizes for training vs validation:
- Validation can use larger batches (no gradients)
- Calibration discovers both independently
- Improves overall throughput

### Phase 6: Adaptive Batch Sizing

Real-time batch size adjustment during training:
- Monitor memory pressure per-batch
- Increase batch size if memory < 50% for 10+ batches
- Decrease batch size if memory > 85% for 3+ batches
- Requires gradient accumulation for consistency

## References

### Related Documentation
- src/handwriting_ai/training/calibration/README.md (if exists)
- src/handwriting_ai/training/safety.py docstrings
- Python multiprocessing documentation: pickle protocol

### Production Evidence
- Railway logs: 2025-11-13 showing 19% memory usage
- Calibration logs: batch_size=128 discovery at 20% memory
- Training logs: stable 165 samples/sec throughput

### Algorithm Research
- Binary search: O(log n) complexity analysis
- Exponential search: Optimal for unbounded search spaces
- Memory profiling: PyTorch memory management patterns

## Approval and Sign-off

This design document requires review and approval before implementation begins. Implementation will proceed in phases with testing at each stage.

**Reviewers**: Architecture team, Training pipeline owners
**Implementation Timeline**: 2-3 days for full implementation + testing
**Deployment**: Staged rollout via feature flag or calibration_samples config
