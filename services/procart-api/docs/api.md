# procart-api — HTTP API

Strict, typed endpoints for previewing frames, rendering frames to disk, and encoding videos via ffmpeg hook.

## Conventions

- All request bodies are plain JSON mapped to internal TypedDicts (no Pydantic models in core logic).
- Validation errors surface as structured errors via platform_core exception handlers.
- No best-effort fallbacks; invalid inputs propagate as errors.

## Endpoints

### GET /healthz

Liveness probe. Returns `{"status": "ok"}`.

### GET /readyz

Readiness probe. Returns `{"status": "ready", "reason": null}`. Rendering runs
in-process, so there is no external dependency whose reachability could make
this differ from liveness.

### GET /registries/modules
### GET /registries/camera-paths
### GET /registries/tone-mappers
### GET /registries/post-effects
### GET /registries/composite-ops

Return registry names for clients to build valid SceneConfig requests.

### POST /render/preview

Body:

```
{
  "scene": SceneConfig,
  "frame_index": 0,
  "width": 256,
  "height": 256
}
```

Response:

```
{ "path": "/tmp/<scene_id>/preview/preview_000000.png" }
```

### POST /render/frames

Body:

```
{ "scene": SceneConfig, "output_dir": "C:/out" }
```

Response:

```
{ "frames_dir": "C:/out/<scene_id>/frames" }
```

### POST /render/video

Body:

```
{ "scene": SceneConfig, "output_dir": "C:/out" }
```

Response:

```
{ "video_path": "C:/out/<scene_id>/<scene_id>.mp4" }
```

Notes:

- Uses `_test_hooks.FFMPEG_RUNNER.encode_frames_to_video(frames_dir, fps, output_path)`.
- Production sets the real runner in `main.py`; tests inject a fake runner.

