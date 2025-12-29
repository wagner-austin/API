from __future__ import annotations

from typing import Literal

from typing_extensions import TypedDict


class GradientStop(TypedDict, total=True):
    """A color stop in a gradient.

    Attributes:
        offset: Percentage offset (0-100).
        color: Hex color value.
    """

    offset: int
    color: str


class Gradient(TypedDict, total=True):
    """Linear gradient definition for SVG backgrounds.

    Attributes:
        angle: Gradient angle in degrees (0=right, 90=down, etc.).
        stops: Tuple of color stops defining the gradient.
    """

    angle: int
    stops: tuple[GradientStop, ...]


class Theme(TypedDict, total=True):
    """Color theme for SVG cards.

    Attributes:
        bg_color: Background color (hex). Used as fallback if no gradient.
        title_color: Title text color (hex).
        text_color: Body text color (hex).
        border_color: Border color (hex).
        icon_color: Icon color (hex).
        gradient: Optional gradient definition for background.
        glow_color: Optional glow color for text/icons (hex).
        sparkle_color: Optional sparkle/star decoration color (hex).
        sparkle_count: Number of sparkle decorations (0 for none).
    """

    bg_color: str
    title_color: str
    text_color: str
    border_color: str
    icon_color: str
    gradient: Gradient | None
    glow_color: str | None
    sparkle_color: str | None
    sparkle_count: int


# Type alias for theme names
ThemeNameLiteral = Literal[
    "default",
    "dark",
    "dracula",
    "github_dark",
    "transparent",
    "cyberpunk",
    "synthwave",
    "neon",
    "aurora",
    "radical",
]


_THEMES: dict[str, Theme] = {
    "default": {
        "bg_color": "#fffefe",
        "title_color": "#2f80ed",
        "text_color": "#434d58",
        "border_color": "#e4e2e2",
        "icon_color": "#4c71f2",
        "gradient": None,
        "glow_color": None,
        "sparkle_color": None,
        "sparkle_count": 0,
    },
    "dark": {
        "bg_color": "#151515",
        "title_color": "#fff",
        "text_color": "#9f9f9f",
        "border_color": "#333",
        "icon_color": "#79ff97",
        "gradient": None,
        "glow_color": None,
        "sparkle_color": None,
        "sparkle_count": 0,
    },
    "dracula": {
        "bg_color": "#282a36",
        "title_color": "#ff79c6",
        "text_color": "#f8f8f2",
        "border_color": "#44475a",
        "icon_color": "#bd93f9",
        "gradient": None,
        "glow_color": None,
        "sparkle_color": None,
        "sparkle_count": 0,
    },
    "github_dark": {
        "bg_color": "#0d1117",
        "title_color": "#58a6ff",
        "text_color": "#c9d1d9",
        "border_color": "#30363d",
        "icon_color": "#1f6feb",
        "gradient": None,
        "glow_color": None,
        "sparkle_color": None,
        "sparkle_count": 0,
    },
    "transparent": {
        "bg_color": "#00000000",
        "title_color": "#58a6ff",
        "text_color": "#c9d1d9",
        "border_color": "#00000000",
        "icon_color": "#1f6feb",
        "gradient": None,
        "glow_color": None,
        "sparkle_color": None,
        "sparkle_count": 0,
    },
    # --- New premium themes with visual effects ---
    "cyberpunk": {
        "bg_color": "#0a0a0f",
        "title_color": "#00fff9",
        "text_color": "#e0e0e0",
        "border_color": "#ff00ff",
        "icon_color": "#ff00ff",
        "gradient": {
            "angle": 135,
            "stops": (
                {"offset": 0, "color": "#0a0a0f"},
                {"offset": 50, "color": "#1a0a2e"},
                {"offset": 100, "color": "#0a0a0f"},
            ),
        },
        "glow_color": "#00fff9",
        "sparkle_color": "#ff00ff",
        "sparkle_count": 8,
    },
    "synthwave": {
        "bg_color": "#1a1a2e",
        "title_color": "#f72585",
        "text_color": "#e0e0e0",
        "border_color": "#7209b7",
        "icon_color": "#4cc9f0",
        "gradient": {
            "angle": 180,
            "stops": (
                {"offset": 0, "color": "#1a1a2e"},
                {"offset": 50, "color": "#2d1b4e"},
                {"offset": 100, "color": "#1a1a2e"},
            ),
        },
        "glow_color": "#f72585",
        "sparkle_color": "#4cc9f0",
        "sparkle_count": 6,
    },
    "neon": {
        "bg_color": "#0d0d0d",
        "title_color": "#39ff14",
        "text_color": "#ffffff",
        "border_color": "#39ff14",
        "icon_color": "#ff073a",
        "gradient": {
            "angle": 45,
            "stops": (
                {"offset": 0, "color": "#0d0d0d"},
                {"offset": 100, "color": "#1a1a1a"},
            ),
        },
        "glow_color": "#39ff14",
        "sparkle_color": "#ff073a",
        "sparkle_count": 10,
    },
    "aurora": {
        "bg_color": "#0f0c29",
        "title_color": "#a8ff78",
        "text_color": "#e0e0e0",
        "border_color": "#78ffd6",
        "icon_color": "#78ffd6",
        "gradient": {
            "angle": 135,
            "stops": (
                {"offset": 0, "color": "#0f0c29"},
                {"offset": 33, "color": "#302b63"},
                {"offset": 66, "color": "#24243e"},
                {"offset": 100, "color": "#0f0c29"},
            ),
        },
        "glow_color": "#78ffd6",
        "sparkle_color": "#a8ff78",
        "sparkle_count": 12,
    },
    "radical": {
        "bg_color": "#141321",
        "title_color": "#fe428e",
        "text_color": "#f8f8f2",
        "border_color": "#a9fef7",
        "icon_color": "#f8d847",
        "gradient": {
            "angle": 160,
            "stops": (
                {"offset": 0, "color": "#141321"},
                {"offset": 50, "color": "#1e1b32"},
                {"offset": 100, "color": "#141321"},
            ),
        },
        "glow_color": "#fe428e",
        "sparkle_color": "#a9fef7",
        "sparkle_count": 8,
    },
}


def get_theme(name: str) -> Theme:
    """Get theme by name.

    Args:
        name: Theme name.

    Returns:
        Theme TypedDict.
    """
    return _THEMES.get(name, _THEMES["default"])


def get_theme_names() -> tuple[str, ...]:
    """Get all available theme names.

    Returns:
        Tuple of theme name strings.
    """
    return tuple(_THEMES.keys())


__all__ = [
    "Gradient",
    "GradientStop",
    "Theme",
    "ThemeNameLiteral",
    "get_theme",
    "get_theme_names",
]
