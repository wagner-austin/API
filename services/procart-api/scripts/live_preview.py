"""Real-time pygame preview for procart neon orbs with keyboard controls.

Controls:
  1/2: Orb count -/+
  3/4: Core radius -/+
  5/6: Halo radius -/+
  7/8: Core intensity -/+
  9/0: Halo intensity -/+
  -/=: Speed -/+
  R: Reset defaults
  ESC: Quit
"""

from __future__ import annotations

from typing import Final, Literal, Protocol

import pygame
from platform_core.logging import get_logger
from procart.color import apply_tone_map
from procart.math_backend import BACKEND, FloatArray
from procart.modules.background import BlackBackground
from procart.modules.neon_orbs import NeonOrbs
from procart.types import (
    NeonGlowConfig,
    NeonOrbsLayerConfig,
    Resolution,
    ToneMappingConfigExposureGamma,
)
from typing_extensions import TypedDict

from scripts import _test_hooks


class _NumpyModule(Protocol):
    """The numpy surface this script consumes, typed.

    Imported through a protocol rather than `import numpy as np` because every
    raw numpy call returns `ndarray[Any, dtype[Any]]`, which this repo's mypy
    settings reject on sight.
    """

    def stack(self, arrays: list[FloatArray], axis: int) -> FloatArray: ...

    def transpose(self, a: FloatArray, axes: tuple[int, int, int]) -> FloatArray: ...


class _SurfarrayModule(Protocol):
    """pygame.surfarray, typed in terms of FloatArray.

    pygame's own signature names a concrete `ndarray[Any, dtype[Any]]`, so
    calling it directly reintroduces the Any this module just removed.
    """

    def make_surface(self, array: FloatArray) -> pygame.Surface: ...


_logger = get_logger(__name__)

_np: _NumpyModule = __import__("numpy")
_surfarray: _SurfarrayModule = __import__("pygame.surfarray", fromlist=["surfarray"])


class RenderParams(TypedDict):
    """Parameters for rendering neon orbs."""

    orb_count: int
    core_radius: float
    halo_radius: float
    core_intensity: float
    halo_intensity: float
    speed: float


ParamKey = Literal[
    "orb_count", "core_radius", "halo_radius", "core_intensity", "halo_intensity", "speed"
]


class _KeyConfig(TypedDict):
    """Configuration for a single key mapping."""

    param: ParamKey
    delta: float
    min_val: float
    max_val: float


def render_frame(
    t: float,
    resolution: Resolution,
    orb_count: int,
    core_radius: float,
    halo_radius: float,
    core_intensity: float,
    halo_intensity: float,
) -> FloatArray:
    """Render a single frame with current parameters.

    Args:
        t: Time value for animation.
        resolution: Frame resolution (width and height).
        orb_count: Number of neon orbs to render.
        core_radius: Radius of the bright core of each orb.
        halo_radius: Radius of the glow halo around each orb.
        core_intensity: Brightness intensity of the core.
        halo_intensity: Brightness intensity of the halo.

    Returns:
        RGB image as uint8 FloatArray suitable for pygame display.
    """
    # Black background
    bg_mod = BlackBackground()

    # Neon orbs with current params
    glow: NeonGlowConfig = {
        "core_intensity": core_intensity,
        "halo_intensity": halo_intensity,
        "core_radius": core_radius,
        "halo_radius": halo_radius,
    }
    orbs_cfg: NeonOrbsLayerConfig = {
        "module": "neon_orbs",
        "id": "orbs",
        "opacity": 1.0,
        "parallax_depth": 1.0,
        "composite": "normal",
        "count": orb_count,
        "glow": glow,
    }
    orbs_mod = NeonOrbs(orbs_cfg)

    # Render layers
    bg_rgba = bg_mod.render_frame(t, resolution, 0, 0.0, 0.0)
    orbs_rgba = orbs_mod.render_frame(t, resolution, 0, 0.0, 0.0)

    # Alpha composite (orbs over background)
    bg_r = BACKEND.channel(bg_rgba, 0)
    bg_g = BACKEND.channel(bg_rgba, 1)
    bg_b = BACKEND.channel(bg_rgba, 2)
    o_r = BACKEND.channel(orbs_rgba, 0)
    o_g = BACKEND.channel(orbs_rgba, 1)
    o_b = BACKEND.channel(orbs_rgba, 2)
    o_a = BACKEND.channel(orbs_rgba, 3)

    inv_a = 1.0 - o_a
    r = o_r + bg_r * inv_a
    g = o_g + bg_g * inv_a
    b = o_b + bg_b * inv_a

    # Manually stack RGB channels using numpy
    rgb_hdr_np = _np.stack([r, g, b], axis=-1)

    # Tone map - numpy ndarray is compatible with FloatArray Protocol
    tone_cfg: ToneMappingConfigExposureGamma = {
        "type": "exposure_gamma",
        "exposure": 1.0,
        "gamma": 2.2,
    }

    rgb_ldr = apply_tone_map(rgb_hdr_np, tone_cfg)

    # Convert to uint8 for pygame
    rgb_clipped = BACKEND.clip(rgb_ldr, 0.0, 1.0)
    rgb_scaled = rgb_clipped * 255.0
    result: FloatArray = rgb_scaled.astype("uint8", copy=False)
    return result


def _apply_delta_to_param(
    params: RenderParams,
    param: ParamKey,
    delta: float,
    min_val: float,
    max_val: float,
) -> None:
    """Apply delta to a parameter with bounds checking.

    Args:
        params: Parameter dictionary to update (mutated in place).
        param: Key of the parameter to update.
        delta: Amount to add to the current value.
        min_val: Minimum allowed value.
        max_val: Maximum allowed value.
    """
    if param == "orb_count":
        current = params[param]
        params[param] = max(int(min_val), min(int(max_val), current + int(delta)))
    else:
        current_val = params[param]
        params[param] = max(min_val, min(max_val, current_val + delta))


def _build_key_map() -> dict[int, _KeyConfig]:
    """Build the keyboard mapping configuration.

    Returns:
        Dictionary mapping pygame key codes to their parameter configurations.
    """
    return {
        pygame.K_1: {"param": "orb_count", "delta": -1.0, "min_val": 1.0, "max_val": 20.0},
        pygame.K_2: {"param": "orb_count", "delta": 1.0, "min_val": 1.0, "max_val": 20.0},
        pygame.K_3: {"param": "core_radius", "delta": -0.005, "min_val": 0.01, "max_val": 0.2},
        pygame.K_4: {"param": "core_radius", "delta": 0.005, "min_val": 0.01, "max_val": 0.2},
        pygame.K_5: {"param": "halo_radius", "delta": -0.01, "min_val": 0.02, "max_val": 0.5},
        pygame.K_6: {"param": "halo_radius", "delta": 0.01, "min_val": 0.02, "max_val": 0.5},
        pygame.K_7: {"param": "core_intensity", "delta": -0.2, "min_val": 0.1, "max_val": 10.0},
        pygame.K_8: {"param": "core_intensity", "delta": 0.2, "min_val": 0.1, "max_val": 10.0},
        pygame.K_9: {"param": "halo_intensity", "delta": -0.1, "min_val": 0.05, "max_val": 3.0},
        pygame.K_0: {"param": "halo_intensity", "delta": 0.1, "min_val": 0.05, "max_val": 3.0},
        pygame.K_MINUS: {"param": "speed", "delta": -0.1, "min_val": 0.1, "max_val": 3.0},
        pygame.K_EQUALS: {"param": "speed", "delta": 0.1, "min_val": 0.1, "max_val": 3.0},
    }


def _handle_key_event(
    key_code: int,
    params: RenderParams,
    defaults: RenderParams,
    key_map: dict[int, _KeyConfig],
) -> RenderParams:
    """Handle keyboard events and update parameters.

    Args:
        key_code: Pygame key code from the event.
        params: Current parameter values.
        defaults: Default parameter values for reset.
        key_map: Mapping of key codes to parameter configurations.

    Returns:
        Updated parameter dictionary.
    """
    k_r: Final[int] = pygame.K_r

    if key_code == k_r:
        return defaults.copy()

    if key_code in key_map:
        cfg = key_map[key_code]
        _apply_delta_to_param(params, cfg["param"], cfg["delta"], cfg["min_val"], cfg["max_val"])

    return params


def _draw_overlay(
    screen: pygame.Surface,
    font: pygame.font.Font,
    clock: pygame.time.Clock,
    params: RenderParams,
) -> None:
    """Draw overlay text on screen.

    Args:
        screen: Pygame surface to draw on.
        font: Pygame font for rendering text.
        clock: Pygame clock for FPS calculation.
        params: Current render parameters to display.
    """
    fps = clock.get_fps()
    lines = [
        f"FPS: {fps:.1f}",
        f"Orbs(1/2): {params['orb_count']}",
        f"Core R(3/4): {params['core_radius']:.3f}",
        f"Halo R(5/6): {params['halo_radius']:.2f}",
        f"Core I(7/8): {params['core_intensity']:.1f}",
        f"Halo I(9/0): {params['halo_intensity']:.2f}",
        f"Speed(-/=): {params['speed']:.1f}",
        "R=reset ESC=quit",
    ]
    y = 10
    for line in lines:
        text = font.render(line, True, (255, 255, 255), (0, 0, 0))
        screen.blit(text, (10, y))
        y += 20


def _process_events(
    params: RenderParams,
    defaults: RenderParams,
    key_map: dict[int, _KeyConfig],
) -> tuple[RenderParams, bool]:
    """Process all pygame events and update state.

    Args:
        params: Current render parameters.
        defaults: Default parameters for reset.
        key_map: Mapping of key codes to parameter configurations.

    Returns:
        Tuple of (updated params, continue_running flag).
    """
    k_escape: Final[int] = pygame.K_ESCAPE
    quit_event: Final[int] = pygame.QUIT
    keydown_event: Final[int] = pygame.KEYDOWN

    for event in _test_hooks.event_source():
        if event.type == quit_event:
            return params, False
        if event.type == keydown_event:
            event_key_raw: int = event.key
            event_key: int = int(event_key_raw)
            key_code: int = event_key
            if key_code == k_escape:
                return params, False
            params = _handle_key_event(key_code, params, defaults, key_map)

    return params, True


def main() -> int:
    """Run the interactive pygame preview loop.

    Returns:
        Exit code (0 for success).
    """
    render_w, render_h = 512, 512
    screen = _test_hooks.display.create(
        (render_w, render_h), "Procart Live - ESC quit, see console for controls"
    )

    pygame.font.init()
    font = pygame.font.Font(None, 24)

    defaults: RenderParams = {
        "orb_count": 3,
        "core_radius": 0.03,
        "halo_radius": 0.15,
        "core_intensity": 2.0,
        "halo_intensity": 0.4,
        "speed": 0.5,
    }
    params: RenderParams = defaults.copy()

    key_map = _build_key_map()
    clock = pygame.time.Clock()
    t = 0.0
    resolution: Resolution = {"width": render_w, "height": render_h}

    _logger.info("%s", __doc__)

    running = True
    while running:
        time_delta = clock.tick(30) / 1000.0

        params, running = _process_events(params, defaults, key_map)

        t += time_delta * params["speed"]
        t = t % 1.0

        rgb = render_frame(
            t,
            resolution,
            params["orb_count"],
            params["core_radius"],
            params["halo_radius"],
            params["core_intensity"],
            params["halo_intensity"],
        )

        # pygame wants (w, h, 3) uint8; the renderer produces (h, w, 3).
        rgb_transposed = _np.transpose(rgb.astype("uint8", copy=False), (1, 0, 2))
        surface = _surfarray.make_surface(rgb_transposed)
        screen.blit(surface, (0, 0))
        _draw_overlay(screen, font, clock, params)
        _test_hooks.display.present()

    _test_hooks.display.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
