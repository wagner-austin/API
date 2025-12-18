Implementation Plan

Scope
- Core library providing typed procedural rendering building blocks and modules.
- No external config formats; Python-returned TypedDicts validated internally.
- Full tests with 100% branch + statement coverage.

Standards Alignment
- Guards, mypy --strict, Ruff over src/tests/scripts; coverage branch mode fail_under=100.
- No Any, cast, type: ignore, .pyi, or dataclasses in src. No try/except in core.
- Dynamic imports must assign to Protocol-annotated variables to avoid Any.

Compute Backend
- Introduce a MathBackend Protocol (sin, cos, exp, clip, grid, distance, basic ops).
- Default NumPyBackend implementation; injectable into render functions/Scene. Production requires NumPy; we hide it behind a Protocol to satisfy typing without Any/ignores.
- Optional future GPU: add a Torch/CuPy backend behind dynamic import using Protocol typing.

Configuration
- Python scene/job builders return TypedDicts; validate via internal _decode_* functions.
- Registry mapping scene_id -> builder in src/procart/configs/scenes_registry.py.
- No Pydantic; no TOML; no framework validation.

Hooks
- External tools (ffmpeg) via Protocol and set by consumers at service startup.
- No conditionals in core; call hook directly.

Dependencies
- numpy, Pillow. No httpx usage in core library. No GPU deps by default.
- Camera builder: integrate `build_camera_from_config(CameraConfig)` to compute (x,y) with amplitude/phase.

Determinism & Drift
- Seed derivation from scene_id + layer.id for deterministic outputs.
- Golden-frame tests to catch unintended drift.

Milestones
1) Scaffolding: pyproject, Makefile, guard, docs, tests harness.
2) Types & Exceptions: TypedDicts and ConfigError.
3) Math Utils: color and geometry pure functions.
4) Module Interface: VisualModule Protocol.
5) Modules: background, neon_orbs, recursive_rects.
6) Scene: layers, camera/parallax, alpha-over composition with resolution override.
7) IO & Runner: images_io, ffmpeg Protocol + RealFfmpegRunner.
8) Configs: scenes_registry and sample scenes; config_decode validation.
9) Render pipeline: supersampling, tone map, downsample, frame writing. Use typed camera config via `registry_camera.build_camera_from_config`.
10) Tests: unit, integration, golden-image checks to hold 100% coverage.

Quality Gates
- Mypy strict; Ruff formatting and lint; monorepo guards.
- No Any/cast/type:ignore/.pyi/dataclasses in src.
- No try/except in core; explicit exceptions propagate.

Output Conventions
- Frames: {output_dir}/{scene_id}/frames/frame_{index:06}.png
- Video:  {output_dir}/{scene_id}/{scene_id}.mp4

Pluggable Architecture

- Visual Modules
  - Protocol: VisualModule with render_frame(...)-> np.ndarray (RGBA, float32)
  - Registry: VISUAL_MODULES: dict[str, VisualModuleFactory]
  - Config: LayerConfig union; module chooses config branch by name

- Camera Paths
  - Protocol: CameraPath with position(t: float) -> tuple[float, float]
  - Registry: CAMERA_PATHS: dict[str, CameraPathFactory]
  - Built-ins: circular, figure_eight; typed configs with {type, amplitude, phase}
  - Builder: `build_camera_from_config(cfg: CameraConfig)` returning a scaled/phase-shifted path

- Tone Mappers
  - Protocol: ToneMapper with apply(rgb_linear: np.ndarray) -> np.ndarray
  - Registry: TONE_MAPPERS: dict[str, ToneMapperFactory]
  - Built-ins: exposure_gamma (default), reinhard_simple, filmic_simple

- Post Effects
  - Protocol: PostEffect with apply(rgba_hdr, resolution, t, seed) -> np.ndarray
  - Registry: POST_EFFECTS: dict[str, PostEffectFactory]
  - Pipeline: 0..N effects applied pre-tone-map (HDR domain)

- Composite Ops
  - Protocol: CompositeOp with composite(back, front) -> np.ndarray
  - Modes: normal (alpha-over), add, screen, lighten, darken
  - Config: per-layer selection; unknown → ValueError

- Noise Sources
  - Protocol: NoiseSource with generate(resolution, seed, t) -> np.ndarray
  - Built-ins: value, perlin, simplex, fbm (stacked)
  - Used by modules for fields/warps/masks

- Palettes
  - Protocol: Palette with sample(h: float) -> np.ndarray (RGB linear)
  - Built-ins: neon sets, gradients, cosine palettes
  - Modules map phase → hue → palette.sample()

- Math Backend
  - Protocol: MathBackend (sin, cos, exp, clip, grid, distance, blur)
  - Default: NumpyBackend; future: Torch/CuPy via dynamic import with Protocol-typed assignment

Registry and Decoding Rules
- Each registry maps string name → factory from TypedDict config
- Scene and layer configs include typed selectors (e.g., tone_mapping.type)
- Internal _decode_* functions validate names and payloads; unknown names or bad types raise ValueError

Algorithms & Presets

- Algorithm Families (as VisualModules)
  - fractal_mandelbrot: max_iter, bailout, zoom, pan_x, pan_y, coloring, palette
  - fractal_julia: c_re, c_im, zoom, pan, coloring, palette
  - spiral_flow: turns, radial_gain, angular_gain, falloff, palette
  - neon_orbs, background, recursive_rects (initial set)
  - Future: kaleidoscope_warp, ifs_flame, lsystem_branches (add as separate modules)

- Presets Registry
  - PRESETS: dict[str, Callable[[], LayerConfig]]
  - Stable, named presets per algorithm (e.g., "mandelbrot_neon_zoom_v1")
  - Decoders expand preset → concrete LayerConfig (then validate); unknown preset → ValueError

Param Schedules

- Protocol: ParamSchedule with value(t: float) -> float | int | tuple[float, ...]
- Built-ins: constant, linear, ease_in_out, periodic(sin/cos), piecewise, noise1D (deterministic)
- Config: ScheduleConfig union; modules may embed schedule fields (e.g., `turns_schedule` for SpiralFlow)
- Validation: discriminated unions validated at edge; unknown names raise ValueError

Scenes With Multiple Algorithms

- A Scene composes an ordered list of LayerConfig entries; each can choose a distinct algorithm
- Layer-level controls: opacity, parallax_depth, composite mode; optional palette/noise
- PostEffects and ToneMapper applied after composition (HDR domain first, then tone map)

Testing Requirements

- Unit: shape, dtype=float32, alpha∈[0,1], RGB≥0, no NaN/inf; determinism with fixed seeds
- Time variance: frames at t=0 vs t=0.5 must differ (module-specific invariant)
- Decoders: invalid names/keys/types → ValueError with precise messages
- Golden images: small deterministic presets for drift detection
- Coverage: 100% statements and branches across src and scripts

Docstrings & API Contracts

- Google-style docstrings for all public functions/classes with Args/Returns/Raises
- Explicit failure points documented; no try/except in core logic

Guards & Monorepo Integration

- scripts/guard.py runs monorepo guard orchestrator; banned: Any, cast, TypeAlias, type: ignore
- Ruff and mypy strict configured for src/tests/scripts; branch coverage fail_under=100
- Root validation: bash C:\\Users\\Test\\PROJECTS\\API\\make check | tail -100


