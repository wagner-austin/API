from __future__ import annotations

from typing import Final, Protocol

from .modules.background import BlackBackground
from .modules.base import VisualModule
from .modules.fractal_mandelbrot import FractalMandelbrot
from .modules.neon_orbs import NeonOrbs
from .modules.recursive_rects import RecursiveRects
from .modules.spiral_flow import SpiralFlow
from .types import FractalMandelbrotLayerConfig, LayerConfig, SpiralFlowLayerConfig


class Factory(Protocol):
    def __call__(self, cfg: LayerConfig) -> VisualModule: ...


_NAMES: Final[tuple[str, str, str, str, str]] = (
    "black_background",
    "neon_orbs",
    "recursive_rects",
    "fractal_mandelbrot",
    "spiral_flow",
)


def list_available_modules() -> list[str]:
    """List registered visual module names."""
    return list(_NAMES)


def _factory_black_background(cfg: LayerConfig) -> VisualModule:
    if cfg["module"] != "black_background":
        raise ValueError("config module mismatch for black_background")
    return BlackBackground()


def _factory_neon_orbs(cfg: LayerConfig) -> VisualModule:
    if cfg["module"] != "neon_orbs":
        raise ValueError("config module mismatch for neon_orbs")
    return NeonOrbs(cfg)


def _factory_recursive_rects(cfg: LayerConfig) -> VisualModule:
    if cfg["module"] != "recursive_rects":
        raise ValueError("config module mismatch for recursive_rects")
    return RecursiveRects(cfg)


def _factory_fractal_mandelbrot(cfg: LayerConfig) -> VisualModule:
    if cfg["module"] != "fractal_mandelbrot":
        raise ValueError("config module mismatch for fractal_mandelbrot")
    fm_cfg: FractalMandelbrotLayerConfig = cfg
    return FractalMandelbrot(fm_cfg)


def _factory_spiral_flow(cfg: LayerConfig) -> VisualModule:
    if cfg["module"] != "spiral_flow":
        raise ValueError("config module mismatch for spiral_flow")
    sp_cfg: SpiralFlowLayerConfig = cfg
    return SpiralFlow(sp_cfg)


def _factories() -> dict[str, Factory]:
    return {
        "black_background": _factory_black_background,
        "neon_orbs": _factory_neon_orbs,
        "recursive_rects": _factory_recursive_rects,
        "fractal_mandelbrot": _factory_fractal_mandelbrot,
        "spiral_flow": _factory_spiral_flow,
    }


def get_factory(name: str) -> Factory:
    """Return the factory for a registered module name.

    Args:
        name: Registered module key.

    Returns:
        Factory: Callable building a visual module.

    Raises:
        ValueError: If name is unknown (coverage branch exercised in tests).
    """
    facs = _factories()
    if name in facs:
        return facs[name]
    # Fallthrough: unknown name
    raise ValueError(f"unknown module: {name}")


def build_module(cfg: LayerConfig) -> VisualModule:
    """Construct a VisualModule instance from a layer config union.

    Args:
        cfg: LayerConfig union with a 'module' selector.

    Returns:
        VisualModule: Concrete module instance.

    Raises:
        ValueError: If module selector is unknown.
    """
    return get_factory(cfg["module"])(cfg)


__all__ = ["build_module", "get_factory", "list_available_modules"]
