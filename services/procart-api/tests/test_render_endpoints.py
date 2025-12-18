from __future__ import annotations

import httpx
import pytest
from procart.types import (
    BlackBackgroundLayerConfig,
    RenderTimingConfig,
    Resolution,
    SceneConfig,
    ToneMappingConfigExposureGamma,
)

from procart_api.app import create_app


def _scene_minimal() -> SceneConfig:
    res: Resolution = {"width": 16, "height": 12}
    timing: RenderTimingConfig = {"duration_seconds": 1.0, "fps": 2, "supersample_factor": 1}
    tone: ToneMappingConfigExposureGamma = {"type": "exposure_gamma", "exposure": 1.0, "gamma": 2.2}
    layer: BlackBackgroundLayerConfig = {
        "module": "black_background",
        "id": "bb",
        "opacity": 1.0,
        "parallax_depth": 0.0,
        "composite": "normal",
    }
    return {
        "id": "s1",
        "description": "test scene",
        "resolution": res,
        "timing": timing,
        "tone_mapping": tone,
        "camera": {"type": "circular", "amplitude": 0.0, "phase": 0.0},
        "layers": [layer],
    }


@pytest.mark.asyncio
async def test_preview_and_frames_endpoints_roundtrip(tmp_path: str | bytes) -> None:
    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        scene = _scene_minimal()
        req_prev: dict[str, object] = {"scene": scene, "frame_index": 0, "width": 8, "height": 6}
        r_prev = await ac.post("/render/preview", json=req_prev)
        assert r_prev.status_code == 200
        data_prev: dict[str, str] = r_prev.json()
        assert data_prev["path"].endswith(".png")

        out_dir = str(tmp_path)
        req_frames: dict[str, object] = {"scene": scene, "output_dir": out_dir}
        r_frames = await ac.post("/render/frames", json=req_frames)
        assert r_frames.status_code == 200
        data_frames: dict[str, str] = r_frames.json()
        fd = data_frames["frames_dir"]
        assert fd.endswith("/s1/frames") or fd.endswith("\\s1\\frames")


@pytest.mark.asyncio
async def test_render_video_endpoint_invokes_hook(tmp_path: str | bytes) -> None:
    # Install a fake runner via hooks that writes a small file
    from procart_api import _test_hooks as _hooks

    class _FakeRunner:
        def encode_frames_to_video(self, frames_dir: str, fps: int, output_path: str) -> None:
            # Create parent dir and write a tiny file marking invocation
            import os

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                f.write(b"ok")

    _hooks.FFMPEG_RUNNER = _FakeRunner()

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        scene = _scene_minimal()
        out_dir = str(tmp_path)
        req_video: dict[str, object] = {"scene": scene, "output_dir": out_dir}
        r_video = await ac.post("/render/video", json=req_video)
        assert r_video.status_code == 200
        data_vid: dict[str, str] = r_video.json()
        vp = data_vid["video_path"]
        assert vp.endswith("s1.mp4")
        # File should exist with >0 size
        import os

        assert os.path.exists(vp) and os.path.getsize(vp) > 0


@pytest.mark.asyncio
async def test_render_video_endpoint_missing_hook_raises(tmp_path: str | bytes) -> None:
    from procart_api import _test_hooks as _hooks

    # Ensure hook is None to exercise error branch
    _hooks.FFMPEG_RUNNER = None

    app = create_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
        scene = _scene_minimal()
        out_dir = str(tmp_path)
        req_video: dict[str, object] = {"scene": scene, "output_dir": out_dir}
        # Some environments surface exceptions directly rather than a response.
        try:
            r_video = await ac.post("/render/video", json=req_video)
        except ValueError as exc:
            # Accept direct exception with the expected message.
            assert "FFMPEG_RUNNER is not set" in str(exc)
        else:
            # Or accept a 4xx/5xx response returned by installed handlers.
            assert 400 <= r_video.status_code < 600
