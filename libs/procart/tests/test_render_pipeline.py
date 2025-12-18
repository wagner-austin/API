from __future__ import annotations

import os

import pytest

from procart.registry import build_module
from procart.render import render_frame_at_resolution, render_scene_to_frames
from procart.scene import Layer
from procart.types import (
    BlackBackgroundLayerConfig,
    RenderJobConfig,
    RenderTimingConfig,
    Resolution,
    SceneConfig,
    ToneMappingConfigExposureGamma,
)


def _scene_for_render(supersample: int = 2) -> SceneConfig:
    res: Resolution = {"width": 12, "height": 10}
    timing: RenderTimingConfig = {
        "duration_seconds": 1.0,
        "fps": 2,
        "supersample_factor": int(supersample),
    }
    tone: ToneMappingConfigExposureGamma = {
        "type": "exposure_gamma",
        "exposure": 1.0,
        "gamma": 2.2,
    }
    layer: BlackBackgroundLayerConfig = {
        "module": "black_background",
        "id": "bg",
        "opacity": 1.0,
        "parallax_depth": 0.0,
        "composite": "normal",
    }
    return {
        "id": "sceneA",
        "description": "bg only",
        "resolution": res,
        "timing": timing,
        "tone_mapping": tone,
        "camera": {"type": "circular", "amplitude": 0.0, "phase": 0.0},
        "layers": [layer],
    }


def test_render_frame_at_resolution_basic() -> None:
    scene = _scene_for_render(supersample=1)
    # Convert configs to runtime layers
    mod = build_module(scene["layers"][0])
    rt_layers = [
        Layer(
            id="bg",
            module=mod,
            opacity=1.0,
            parallax_depth=0.0,
        )
    ]
    out = render_frame_at_resolution(
        layers=rt_layers,
        frame_index=0,
        base_scene=scene,
        resolution=scene["resolution"],
    )
    assert out.shape == (scene["resolution"]["height"], scene["resolution"]["width"], 4)


def test_render_scene_to_frames_supersample_and_output(tmp_path: str | bytes) -> None:
    scene = _scene_for_render(supersample=2)
    mod = build_module(scene["layers"][0])
    rt_layers = [Layer(id="bg", module=mod, opacity=1.0, parallax_depth=0.0)]
    job: RenderJobConfig = {"output_dir": str(tmp_path), "scene": scene}
    frames_dir = render_scene_to_frames(rt_layers, job)
    # Expect frame_000000.png exists
    expected = os.path.join(frames_dir, "frame_000000.png")
    assert os.path.exists(expected)


def test_render_scene_to_frames_invalid_supersample_raises(tmp_path: str | bytes) -> None:
    scene = _scene_for_render(supersample=0)
    mod = build_module(scene["layers"][0])
    rt_layers = [Layer(id="bg", module=mod, opacity=1.0, parallax_depth=0.0)]
    job: RenderJobConfig = {"output_dir": str(tmp_path), "scene": scene}
    with pytest.raises(ValueError):
        _ = render_scene_to_frames(rt_layers, job)


def test_render_frame_at_resolution_invalid_timing_and_index() -> None:
    # fps zero triggers timing error; and index out of range path as well
    res: Resolution = {"width": 8, "height": 6}
    timing_bad: RenderTimingConfig = {"duration_seconds": 1.0, "fps": 0, "supersample_factor": 1}
    tone: ToneMappingConfigExposureGamma = {"type": "exposure_gamma", "exposure": 1.0, "gamma": 2.2}
    layer_cfg: BlackBackgroundLayerConfig = {
        "module": "black_background",
        "id": "bg",
        "opacity": 1.0,
        "parallax_depth": 0.0,
        "composite": "normal",
    }
    scene_bad: SceneConfig = {
        "id": "bad",
        "description": "bad timing",
        "resolution": res,
        "timing": timing_bad,
        "tone_mapping": tone,
        "camera": {"type": "circular", "amplitude": 0.0, "phase": 0.0},
        "layers": [layer_cfg],
    }
    # Runtime layers
    mod = build_module(layer_cfg)
    rt_layers = [Layer(id="bg", module=mod, opacity=1.0, parallax_depth=0.0)]
    with pytest.raises(ValueError):
        _ = render_frame_at_resolution(
            layers=rt_layers, frame_index=0, base_scene=scene_bad, resolution=res
        )

    # Valid timing but index out of range
    timing_ok: RenderTimingConfig = {"duration_seconds": 1.0, "fps": 1, "supersample_factor": 1}
    scene_idx: SceneConfig = {
        "id": "idx",
        "description": "oob",
        "resolution": res,
        "timing": timing_ok,
        "tone_mapping": tone,
        "camera": {"type": "circular", "amplitude": 0.0, "phase": 0.0},
        "layers": [layer_cfg],
    }
    with pytest.raises(ValueError):
        _ = render_frame_at_resolution(
            layers=rt_layers, frame_index=1, base_scene=scene_idx, resolution=res
        )


def test_render_scene_to_frames_invalid_fps_raises(tmp_path: str | bytes) -> None:
    res: Resolution = {"width": 8, "height": 6}
    timing_bad: RenderTimingConfig = {"duration_seconds": 1.0, "fps": 0, "supersample_factor": 1}
    tone: ToneMappingConfigExposureGamma = {"type": "exposure_gamma", "exposure": 1.0, "gamma": 2.2}
    layer_cfg: BlackBackgroundLayerConfig = {
        "module": "black_background",
        "id": "bg",
        "opacity": 1.0,
        "parallax_depth": 0.0,
        "composite": "normal",
    }
    scene_bad: SceneConfig = {
        "id": "bad2",
        "description": "bad fps",
        "resolution": res,
        "timing": timing_bad,
        "tone_mapping": tone,
        "camera": {"type": "circular", "amplitude": 0.0, "phase": 0.0},
        "layers": [layer_cfg],
    }
    mod = build_module(layer_cfg)
    rt_layers = [Layer(id="bg", module=mod, opacity=1.0, parallax_depth=0.0)]
    job: RenderJobConfig = {"output_dir": str(tmp_path), "scene": scene_bad}
    with pytest.raises(ValueError):
        _ = render_scene_to_frames(rt_layers, job)
