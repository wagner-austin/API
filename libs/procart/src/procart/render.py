from __future__ import annotations

import logging

from .color import apply_tone_map
from .images_io import ensure_dir, resize_image, write_frame_png
from .math_backend import BACKEND, FloatArray
from .registry_camera import build_camera_from_config
from .scene import Layer
from .types import FrameIndex, RenderJobConfig, Resolution, SceneConfig

_logger = logging.getLogger(__name__)


def _compose_layers(
    layers: list[Layer],
    t: float,
    resolution: Resolution,
    cam_x: float,
    cam_y: float,
) -> FloatArray:
    w = int(resolution["width"])  # Width
    h = int(resolution["height"])  # Height
    acc = BACKEND.stack_rgba(
        BACKEND.zeros(h, w),
        BACKEND.zeros(h, w),
        BACKEND.zeros(h, w),
        BACKEND.zeros(h, w),
    )

    def alpha_over(back: FloatArray, front: FloatArray) -> FloatArray:
        r_b, g_b, b_b, a_b = BACKEND.split_rgba(back)
        r_f, g_f, b_f, a_f = BACKEND.split_rgba(front)
        one = BACKEND.ones(int(r_b.shape[0]), int(r_b.shape[1]))
        inv_a_f = one - a_f
        r = r_f + r_b * inv_a_f
        g = g_f + g_b * inv_a_f
        b = b_f + b_b * inv_a_f
        a = a_f + a_b * inv_a_f
        return BACKEND.stack_rgba(r, g, b, a)

    for layer in layers:
        lx = cam_x * layer.parallax_depth
        ly = cam_y * layer.parallax_depth
        rgba = layer.module.render_frame(t, resolution, 0, lx, ly)
        r, g, b, a = BACKEND.split_rgba(rgba)
        a_scaled = a * float(layer.opacity)
        rgba_scaled = BACKEND.stack_rgba(r, g, b, a_scaled)
        acc = alpha_over(acc, rgba_scaled)
    return acc


def render_frame_at_resolution(
    *, layers: list[Layer], frame_index: FrameIndex, base_scene: SceneConfig, resolution: Resolution
) -> FloatArray:
    fps = int(base_scene["timing"]["fps"])  # Fps
    duration = float(base_scene["timing"]["duration_seconds"])  # Seconds
    if fps <= 0 or duration <= 0.0:
        raise ValueError("fps and duration must be positive")
    total_frames = fps * int(duration)
    if not (0 <= int(frame_index) < total_frames):
        raise ValueError("frame_index out of range")
    t = float(frame_index) / float(total_frames)
    # Camera path from typed config: amplitude/phase applied by builder
    cam_cfg = base_scene["camera"]
    cam_path = build_camera_from_config(cam_cfg)
    cam_x, cam_y = cam_path(t)
    return _compose_layers(layers, t, resolution, cam_x, cam_y)


def render_scene_to_frames(scene_layers: list[Layer], job: RenderJobConfig) -> str:
    """Render the scene frames to disk with supersampling and tone mapping.

    Args:
        scene_layers: Ordered list of layers to render.
        job: Render job config containing output_dir and scene config.

    Returns:
        str: Absolute directory path where frames have been written.

    Raises:
        ValueError: If invalid timing or resolution values are provided.
    """
    scene = job["scene"]
    base_res = scene["resolution"]
    timing = scene["timing"]
    supersample = int(timing["supersample_factor"])  # Factor
    if supersample <= 0:
        raise ValueError("supersample_factor must be positive")
    hi_res: Resolution = {
        "width": int(base_res["width"]) * supersample,
        "height": int(base_res["height"]) * supersample,
    }
    fps = int(timing["fps"])  # Fps
    duration = float(timing["duration_seconds"])  # Seconds
    if fps <= 0 or duration <= 0.0:
        raise ValueError("fps and duration must be positive")
    total_frames = fps * int(duration)

    # Prepare output directory
    frames_dir = job["output_dir"].rstrip("/\\") + f"/{scene['id']}/frames"
    ensure_dir(frames_dir)

    _logger.info(
        "Rendering %d frames at %dx%d (supersample=%d)",
        total_frames,
        hi_res["width"],
        hi_res["height"],
        supersample,
    )

    # Render frames
    for i in range(total_frames):
        rgba_hdr = render_frame_at_resolution(
            layers=scene_layers, frame_index=i, base_scene=scene, resolution=hi_res
        )
        # Tone map HDR -> LDR RGB
        tone_cfg = scene["tone_mapping"]
        rgb_ldr = apply_tone_map(rgba_hdr, tone_cfg)
        # Downsample to base resolution if supersampling > 1
        rgb_final = resize_image(rgb_ldr, base_res) if supersample > 1 else rgb_ldr
        out_path = f"{frames_dir}/frame_{i:06d}.png"
        write_frame_png(out_path, rgb_final)

        if (i + 1) % 10 == 0 or i == total_frames - 1:
            _logger.debug("Rendered frame %d/%d", i + 1, total_frames)

    _logger.info("Finished rendering %d frames to %s", total_frames, frames_dir)
    return frames_dir


__all__ = ["render_frame_at_resolution", "render_scene_to_frames"]
