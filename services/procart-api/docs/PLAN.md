Implementation Plan (Service)

Scope
- FastAPI endpoints for health, scenes list, preview frame, render frames, and video encode.
- Module-level hooks for ffmpeg; production sets RealFfmpegRunner.
- Outbound HTTP (if added) goes through platform_core.http_client builders only.

Standards Alignment
- Hypercorn as the ASGI server in dependencies (no Uvicorn usage in code).
- Guards, mypy --strict, Ruff over src/tests/scripts; coverage branch mode fail_under=100.
- No Any, cast, type: ignore, .pyi, or dataclasses in src. No try/except in core.

Routes (to be added post-core)
- GET /health — liveness.
- GET /scenes — registry listing (scene_id, description).
- GET /render/{scene_id}/preview?frame_index=0 — PNG bytes (small resolution override).
- POST /render/{scene_id}/frames?output_dir=...&overrides... — render frames to disk.
- POST /render/{scene_id}/video?output_dir=... — encode frames via FFMPEG_RUNNER.

Integration
- Use Python scene registry from libs/procart/configs.
- Validate overrides via internal decoders (no Pydantic models).
- No production httpx usage unless required; then use platform_core.http_client.

Pluggability Surface (Service Awareness)
- Expose available registries via endpoints (optional): cameras, tone mappers, post-effects, modules, composite ops.
- Accept strict selector values in requests; dispatch via library registries.
- Unknown selectors or invalid payloads return 422 via ValueError → handler.

Strict Standards & Testing
- No Any, cast, type: ignore, .pyi, or dataclasses in src. No try/except in core.
- Decode request params into strict TypedDicts using internal _decode_* functions.
- Outbound HTTP (if added) must use platform_core.http_client builders and Protocol-typed dynamic import.
- 100% statements + branches coverage; tests assert precise failure modes (e.g., 422 for bad selectors).
- Validation via root run: bash C:\\Users\\Test\\PROJECTS\\API\\make check | tail -100

Testing
- httpx.AsyncClient ASGI tests; no mocks; inject FakeFfmpegRunner via _test_hooks.
- 100% branch + statement coverage, including error paths (unknown scene, invalid overrides).
