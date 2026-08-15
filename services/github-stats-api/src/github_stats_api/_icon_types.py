"""TypedDicts describing an SVG skill icon.

Icon paths come from Simple Icons (https://simpleicons.org/). Simple icons use
viewBox 0 0 24 24; multi-path icons specify their own viewBox.
"""

from __future__ import annotations

from typing_extensions import TypedDict


class IconPath(TypedDict, total=True):
    """Single SVG path element with fill color.

    Attributes:
        d: SVG path data string.
        fill: Fill color in hex format (e.g. '#3FAA39').
    """

    d: str
    fill: str


class MultiPathIcon(TypedDict, total=True):
    """Icon composed of multiple SVG paths with custom viewBox.

    Attributes:
        viewbox_width: Original viewBox width for scaling calculations.
        viewbox_height: Original viewBox height for scaling calculations.
        paths: Tuple of path elements to render.
        transform: Group-level SVG transform applied before individual paths.
    """

    viewbox_width: int
    viewbox_height: int
    paths: tuple[IconPath, ...]
    transform: str


__all__ = ["IconPath", "MultiPathIcon"]
