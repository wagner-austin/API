from __future__ import annotations

import logging
import os

from fastapi import APIRouter
from procart.images_io import write_frame_png
from procart.registry import build_module
from procart.render import render_frame_at_resolution, render_scene_to_frames
from procart.types import RenderJobConfig, Resolution, SceneConfig
from typing_extensions import TypedDict

_logger = logging.getLogger(__name__)


class _PreviewRequest(TypedDict):
    scene: SceneConfig
    frame_index: int
    width: int
    height: int


class _FramesRequest(TypedDict):
    scene: SceneConfig
    output_dir: str


class _VideoRequest(TypedDict):
    scene: SceneConfig
    output_dir: str


def build_router() -> APIRouter:
    """Build rendering router: preview image, frames, and video encode.

    Returns:
        APIRouter: Router exposing rendering operations.
    """
    r = APIRouter(prefix="/render")

    # In a future change, scenes will come from a registry. For now, expect the
    # client to POST a full SceneConfig.

    def preview(req: _PreviewRequest) -> dict[str, str]:
        scene = req["scene"]
        frame_index = int(req["frame_index"])
        width = int(req["width"])
        height = int(req["height"])
        _logger.info(
            "Preview request: scene=%s frame=%d size=%dx%d",
            scene["id"],
            frame_index,
            width,
            height,
        )
        res: Resolution = {"width": width, "height": height}
        layers = [build_module(lc) for lc in scene["layers"]]
        # Convert to runtime Layer list
        from procart.scene import Layer as _Layer

        runtime_layers = []
        for lc, mod in zip(scene["layers"], layers, strict=True):
            runtime_layers.append(
                _Layer(
                    id=lc["id"],
                    module=mod,
                    opacity=float(lc["opacity"]),
                    parallax_depth=float(lc["parallax_depth"]),
                )
            )
        rgba = render_frame_at_resolution(
            layers=runtime_layers, frame_index=frame_index, base_scene=scene, resolution=res
        )
        out_dir = os.path.join("/tmp", scene["id"], "preview")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"preview_{int(frame_index):06d}.png")
        write_frame_png(out_path, rgba)
        return {"path": out_path}

    def frames(req: _FramesRequest) -> dict[str, str]:
        scene = req["scene"]
        output_dir = str(req["output_dir"])
        _logger.info("Frames request: scene=%s output=%s", scene["id"], output_dir)
        from procart.scene import Layer as _Layer

        runtime_layers = []
        layers_cfg = list(scene["layers"])  # help mypy see concrete dict shape
        for lc in layers_cfg:
            mod = build_module(lc)
            runtime_layers.append(
                _Layer(
                    id=lc["id"],
                    module=mod,
                    opacity=float(lc["opacity"]),
                    parallax_depth=float(lc["parallax_depth"]),
                )
            )

        job: RenderJobConfig = {"output_dir": output_dir, "scene": scene}
        frames_dir = render_scene_to_frames(runtime_layers, job)
        return {"frames_dir": frames_dir}

    def video(req: _VideoRequest) -> dict[str, str]:
        # Build frames_dir and video_path, then delegate to hook
        from procart_api import _test_hooks as _hooks

        scene = req["scene"]
        output_dir = str(req["output_dir"])
        scene_id = scene["id"]
        _logger.info("Video request: scene=%s output=%s", scene_id, output_dir)
        frames_dir = os.path.join(output_dir, scene_id, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        video_path = os.path.join(output_dir, scene_id, f"{scene_id}.mp4")
        fps = int(scene["timing"]["fps"])  # Fps
        runner = _hooks.FFMPEG_RUNNER
        if runner is None:
            raise ValueError("FFMPEG_RUNNER is not set")
        runner.encode_frames_to_video(frames_dir, fps, video_path)
        return {"video_path": video_path}

    r.add_api_route("/preview", preview, methods=["POST"])
    r.add_api_route("/frames", frames, methods=["POST"])
    r.add_api_route("/video", video, methods=["POST"])
    return r


__all__ = ["build_router"]
