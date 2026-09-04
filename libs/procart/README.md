# procart

Strict, typed procedural-art core library for generating looping neon visual scenes.

## Installation

```bash
poetry add procart
```

## Quick Start

```python
from procart.registry import build_module
from procart.scene import Layer, Scene
from procart.types import Resolution, ToneMappingConfigExposureGamma, RenderTimingConfig
from procart.images_io import write_frame_png

res: Resolution = {"width": 256, "height": 256}
timing: RenderTimingConfig = {"duration_seconds": 2.0, "fps": 30, "supersample_factor": 1}
tone: ToneMappingConfigExposureGamma = {"type": "exposure_gamma", "exposure": 1.0, "gamma": 2.2}

neon_cfg = {
    "module": "neon_orbs",
    "id": "orbs",
    "opacity": 1.0,
    "parallax_depth": 1.0,
    "composite": "normal",
    "count": 3,
    "glow": {"core_intensity": 1.4, "halo_intensity": 0.2, "core_radius": 0.02, "halo_radius": 0.2},
}

layer = Layer(id="orbs", module=build_module(neon_cfg), opacity=1.0, parallax_depth=1.0)
scene = Scene(
    id="demo",
    description="neon demo",
    resolution=res,
    timing=timing,
    tone_mapping=tone,
    layers=[layer],
)

rgba_hdr = scene.render_frame(0)
write_frame_png("frame_000.png", rgba_hdr)
```

## Features

 - Pluggable modules: background, neon_orbs, recursive_rects, fractal_mandelbrot, spiral_flow
- Strict TypedDict configs and internal decoders (no Pydantic/TOML)
- HDR linear RGBA pipeline with exposure/gamma tone mapping
- Supersampling and deterministic seeds to prevent drift
- External tools behind Protocol hooks (FFmpeg runner)

## Development

```bash
make lint   # guards + ruff + mypy
make test   # pytest with --cov-branch
make check  # lint + test (fail_under=100)
```

## Project Structure

```
src/procart/
  color.py        # HSVâ†’RGB, tone mapping
  geometry.py     # grids & distances
  math_backend.py # FloatArray + MathBackend Protocol, NumPy implementation
  modules/        # background, neon_orbs, recursive_rects
  registry.py     # selector->factory mapping
  registry_camera.py # camera paths + typed builder
  scene.py        # Layer/Scene composition and camera
  images_io.py    # write PNG, resize via Pillow
  ffmpeg_runner.py# FfmpegRunner Protocol + RealFfmpegRunner
  config_decode.py# strict decoders for TypedDict configs
  types.py        # all type aliases and TypedDicts
```

## Pluggability

- Visual modules via `VisualModule` Protocol and registry
- Tone mappers via registry (exposure_gamma as default; reinhard/filmic variants covered in tests)
- Camera paths, post-effects, composite ops, palettes, noise sources: see `docs/PLAN.md`
- Math backend swappable via `MathBackend` Protocol (NumPy default)

## Standards

- Python 3.11, mypy `--strict`, Ruff, 100% statements + branches
- No Any, no cast, no `type: ignore`, no dataclasses in `src/`
- No try/except in core; failures are explicit and propagate

See `docs/PLAN.md` for the detailed architecture and pluggability roadmap.

## Cameras & Schedules

Camera selection is typed via `CameraConfig` and the render pipeline applies it using
`build_camera_from_config()`. `SpiralFlow` demonstrates schedule-driven parameters
through `ScheduleConfig` (constant | linear) to evolve values deterministically over t ∈ [0,1].

