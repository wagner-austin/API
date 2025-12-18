from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

import httpx
from typing_extensions import TypedDict

from procart.types import (
    BlackBackgroundLayerConfig,
    CameraConfigCircular,
    NeonOrbsLayerConfig,
    NeonGlowConfig,
    RenderTimingConfig,
    Resolution,
    SceneConfig,
    ToneMappingConfigExposureGamma,
)


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


def _build_demo_scene(
    *, scene_id: str, width: int, height: int, fps: int, duration: float
) -> SceneConfig:
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

    layers = [bg, orbs]
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


def _run_demo(base_url: str, out_dir: Path, *, width: int, height: int, fps: int, duration: float) -> Path:
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
        r_h = client.get("/health")
        r_h.raise_for_status()

        # Render frames
        r_f = client.post("/render/frames", json=frames_req)
        r_f.raise_for_status()
        # Response body is ignored to avoid json Any; path is deterministic
        # by convention: <out>/<scene_id>/frames
        _ = r_f.content
        frames_dir = out_dir_abs / scene_id / "frames"

        # Encode video (service will call ffmpeg via RealFfmpegRunner)
        r_v = client.post("/render/video", json=video_req)
        r_v.raise_for_status()
        # Compute expected output path instead of parsing JSON
        _ = r_v.content
        video_path = (out_dir_abs / scene_id / f"{scene_id}.mp4").resolve()

    # Return the produced video path
    return video_path


def main() -> int:
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
    args = parser.parse_args()

    video_path = _run_demo(
        args.base_url,
        args.out,
        width=args.width,
        height=args.height,
        fps=args.fps,
        duration=args.duration,
    )
    print(str(video_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
