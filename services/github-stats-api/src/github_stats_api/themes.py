from __future__ import annotations

from typing_extensions import TypedDict


class Theme(TypedDict, total=True):
    """Color theme for SVG cards.

    Attributes:
        bg_color: Background color (hex).
        title_color: Title text color (hex).
        text_color: Body text color (hex).
        border_color: Border color (hex).
        icon_color: Icon color (hex).
    """

    bg_color: str
    title_color: str
    text_color: str
    border_color: str
    icon_color: str


_THEMES: dict[str, Theme] = {
    "default": {
        "bg_color": "#fffefe",
        "title_color": "#2f80ed",
        "text_color": "#434d58",
        "border_color": "#e4e2e2",
        "icon_color": "#4c71f2",
    },
    "dark": {
        "bg_color": "#151515",
        "title_color": "#fff",
        "text_color": "#9f9f9f",
        "border_color": "#333",
        "icon_color": "#79ff97",
    },
    "dracula": {
        "bg_color": "#282a36",
        "title_color": "#ff79c6",
        "text_color": "#f8f8f2",
        "border_color": "#44475a",
        "icon_color": "#bd93f9",
    },
    "github_dark": {
        "bg_color": "#0d1117",
        "title_color": "#58a6ff",
        "text_color": "#c9d1d9",
        "border_color": "#30363d",
        "icon_color": "#1f6feb",
    },
    "transparent": {
        "bg_color": "#00000000",
        "title_color": "#58a6ff",
        "text_color": "#c9d1d9",
        "border_color": "#00000000",
        "icon_color": "#1f6feb",
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


__all__ = ["Theme", "get_theme"]
