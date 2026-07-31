from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import httpx
from procart.types import (
    BlackBackgroundLayerConfig,
    CameraConfigCircular,
    LayerConfig,
    NeonGlowConfig,
    NeonOrbsLayerConfig,
    RenderTimingConfig,
    Resolution,
    SceneConfig,
    ToneMappingConfigExposureGamma,
)
from typing_extensions import TypedDict


class _FramesRequest(TypedDict):
    scene: SceneConfig
    output_dir: str


class _VideoRequest(TypedDict):
    scene: SceneConfig
    output_dir: str


class _FramesResponse(TypedDict):
    frames_dir: str


class _VideoResponse(TypedDict):
    video_path: str


class _Args(TypedDict):
    base_url: str
    out: Path
    width: int
    height: int
    fps: int
    duration: float


def _build_demo_scene(
    *, scene_id: str, width: int, height: int, fps: int, duration: float
) -> SceneConfig:
    """Build a demo scene configuration for neon orbs rendering.

    Args:
        scene_id: Unique identifier for the scene.
        width: Resolution width in pixels.
        height: Resolution height in pixels.
        fps: Frames per second for rendering.
        duration: Duration of the scene in seconds.

    Returns:
        Complete scene configuration for rendering.
    """
    res: Resolution = {"width": int(width), "height": int(height)}
    timing: RenderTimingConfig = {
        "duration_seconds": float(duration),
        "fps": int(fps),
        "supersample_factor": 1,
    }
    tone: ToneMappingConfigExposureGamma = {
        "type": "exposure_gamma",
        "exposure": 1.0,
        "gamma": 2.2,
    }
    cam: CameraConfigCircular = {"type": "circular", "amplitude": 0.1, "phase": 0.0}

    bg: BlackBackgroundLayerConfig = {
        "module": "black_background",
        "id": "bg",
        "opacity": 1.0,
        "parallax_depth": 1.0,
        "composite": "normal",
    }

    glow: NeonGlowConfig = {
        "core_intensity": 1.4,
        "halo_intensity": 0.2,
        "core_radius": 0.02,
        "halo_radius": 0.2,
    }
    orbs: NeonOrbsLayerConfig = {
        "module": "neon_orbs",
        "id": "orbs",
        "opacity": 1.0,
        "parallax_depth": 1.0,
        "composite": "normal",
        "count": 3,
        "glow": glow,
    }

    layers: list[LayerConfig] = [bg, orbs]
    scene: SceneConfig = {
        "id": scene_id,
        "description": "procart demo scene",
        "resolution": res,
        "timing": timing,
        "tone_mapping": tone,
        "camera": cam,
        "layers": layers,
    }
    return scene


def _run_demo(
    base_url: str,
    out_dir: Path,
    *,
    width: int,
    height: int,
    fps: int,
    duration: float,
) -> Path:
    """Execute demo rendering by calling procart-api service.

    Args:
        base_url: Base URL of the running procart-api service.
        out_dir: Output directory for frames and video.
        width: Render width in pixels.
        height: Render height in pixels.
        fps: Frames per second.
        duration: Duration in seconds.

    Returns:
        Path to the rendered video file.

    Raises:
        httpx.HTTPStatusError: If API requests fail.
    """
    scene_id: Final[str] = "demo"
    scene = _build_demo_scene(
        scene_id=scene_id, width=width, height=height, fps=fps, duration=duration
    )

    out_dir_abs = out_dir.resolve()
    frames_req: _FramesRequest = {"scene": scene, "output_dir": str(out_dir_abs)}
    video_req: _VideoRequest = {"scene": scene, "output_dir": str(out_dir_abs)}

    # Use a client with a reasonable timeout; service renders synchronously for frames
    timeout = httpx.Timeout(300.0)
    with httpx.Client(base_url=base_url, timeout=timeout) as client:
        # Health check
        r_h = client.get("/healthz")
        r_h.raise_for_status()

        # Render frames
        r_f = client.post("/render/frames", json=frames_req)
        r_f.raise_for_status()
        # Response body is ignored to avoid json Any; path is deterministic
        # by convention: <out>/<scene_id>/frames
        _ = r_f.content

        # Encode video (service will call ffmpeg via RealFfmpegRunner)
        r_v = client.post("/render/video", json=video_req)
        r_v.raise_for_status()
        # Compute expected output path instead of parsing JSON
        _ = r_v.content

    # Return the produced video path
    return (out_dir_abs / scene_id / f"{scene_id}.mp4").resolve()


def main() -> int:
    """Parse arguments and run demo rendering.

    Returns:
        Exit code (0 for success).
    """
    parser = argparse.ArgumentParser(description="Render demo frames and video via procart-api")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL of a running procart-api service (default: %(default)s)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("demo_output"),
        help="Output directory to write frames/video (default: %(default)s)",
    )
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=2.0)
    namespace = parser.parse_args()

    base_url_raw: str = namespace.base_url
    out_raw: Path = namespace.out
    width_raw: int = namespace.width
    height_raw: int = namespace.height
    fps_raw: int = namespace.fps
    duration_raw: float = namespace.duration

    base_url_str: str = str(base_url_raw)
    out_path: Path = Path(out_raw)
    width_int: int = int(width_raw)
    height_int: int = int(height_raw)
    fps_int: int = int(fps_raw)
    duration_float: float = float(duration_raw)

    args: _Args = {
        "base_url": base_url_str,
        "out": out_path,
        "width": width_int,
        "height": height_int,
        "fps": fps_int,
        "duration": duration_float,
    }

    video_path = _run_demo(
        args["base_url"],
        args["out"],
        width=args["width"],
        height=args["height"],
        fps=args["fps"],
        duration=args["duration"],
    )
    print(str(video_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
