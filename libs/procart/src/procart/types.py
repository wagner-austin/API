from __future__ import annotations

from typing import Literal, NotRequired

from typing_extensions import TypedDict

# ---------------------------------------------------------------------------
# Scalar and basic aliases
# ---------------------------------------------------------------------------

FloatScalar = float
IntScalar = int
StrScalar = str
BoolScalar = bool
Scalar = FloatScalar | IntScalar | BoolScalar | StrScalar
ScalarSequence = list[Scalar]

FrameIndex = int
Fps = int
Seconds = float
Width = int
Height = int


# ---------------------------------------------------------------------------
# Core TypedDicts
# ---------------------------------------------------------------------------


class Resolution(TypedDict):
    width: Width
    height: Height


class RenderTimingConfig(TypedDict):
    duration_seconds: Seconds
    fps: Fps
    supersample_factor: int


class ToneMappingConfigExposureGamma(TypedDict):
    type: Literal["exposure_gamma"]
    exposure: float
    gamma: float


class ToneMappingConfigReinhard(TypedDict):
    type: Literal["reinhard"]
    exposure: float


class ToneMappingConfigFilmic(TypedDict):
    type: Literal["filmic"]
    exposure: float


ToneMappingConfig = (
    ToneMappingConfigExposureGamma | ToneMappingConfigReinhard | ToneMappingConfigFilmic
)


class NeonGlowConfig(TypedDict):
    core_intensity: float
    halo_intensity: float
    core_radius: float
    halo_radius: float


class BaseLayerCommon(TypedDict):
    id: str
    opacity: float
    parallax_depth: float
    composite: Literal["normal", "add", "screen", "lighten", "darken"]


class BlackBackgroundLayerConfig(BaseLayerCommon):
    module: Literal["black_background"]


class NeonOrbsLayerConfig(BaseLayerCommon):
    module: Literal["neon_orbs"]
    count: int
    glow: NeonGlowConfig


class RecursiveRectsLayerConfig(BaseLayerCommon):
    module: Literal["recursive_rects"]
    max_depth: int
    min_size: int


class FractalMandelbrotLayerConfig(BaseLayerCommon):
    module: Literal["fractal_mandelbrot"]
    max_iter: int
    bailout: float
    zoom: float
    pan_x: float
    pan_y: float
    coloring: Literal["smooth", "trap"]


class FractalJuliaLayerConfig(BaseLayerCommon):
    module: Literal["fractal_julia"]
    max_iter: int
    bailout: float
    zoom: float
    pan_x: float
    pan_y: float
    c_re: float
    c_im: float
    coloring: Literal["smooth", "trap"]


class ScheduleConfigConstant(TypedDict):
    type: Literal["constant"]
    value: float


class ScheduleConfigLinear(TypedDict):
    type: Literal["linear"]
    start: float
    end: float


ScheduleConfig = ScheduleConfigConstant | ScheduleConfigLinear


class SpiralFlowLayerConfig(BaseLayerCommon):
    module: Literal["spiral_flow"]
    turns: float
    radial_gain: float
    angular_gain: float
    falloff: float
    turns_schedule: NotRequired[ScheduleConfig]


LayerConfig = (
    BlackBackgroundLayerConfig
    | NeonOrbsLayerConfig
    | RecursiveRectsLayerConfig
    | FractalMandelbrotLayerConfig
    | FractalJuliaLayerConfig
    | SpiralFlowLayerConfig
)


class CameraConfigCircular(TypedDict):
    type: Literal["circular"]
    amplitude: float
    phase: float


class CameraConfigFigure8(TypedDict):
    type: Literal["figure_eight"]
    amplitude: float
    phase: float


CameraConfig = CameraConfigCircular | CameraConfigFigure8


class SceneConfig(TypedDict):
    id: str
    description: str
    resolution: Resolution
    timing: RenderTimingConfig
    tone_mapping: ToneMappingConfig
    camera: CameraConfig
    layers: list[LayerConfig]


class RenderJobConfig(TypedDict):
    output_dir: str
    scene: SceneConfig


__all__ = [
    "BaseLayerCommon",
    "BoolScalar",
    "CameraConfig",
    "CameraConfigCircular",
    "CameraConfigFigure8",
    "FloatScalar",
    "Fps",
    "FrameIndex",
    "Height",
    "IntScalar",
    "LayerConfig",
    "NeonGlowConfig",
    "RenderJobConfig",
    "RenderTimingConfig",
    "Resolution",
    "Scalar",
    "ScalarSequence",
    "SceneConfig",
    "ScheduleConfig",
    "ScheduleConfigConstant",
    "ScheduleConfigLinear",
    "Seconds",
    "StrScalar",
    "ToneMappingConfig",
    "ToneMappingConfigExposureGamma",
    "ToneMappingConfigFilmic",
    "ToneMappingConfigReinhard",
    "Width",
]
