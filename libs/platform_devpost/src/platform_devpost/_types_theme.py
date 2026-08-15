"""platform_devpost Theme and DisplayedLocation payloads."""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    require_int,
    require_str,
)

# -----------------------------------------------------------------------------
# Theme
# -----------------------------------------------------------------------------


class Theme:
    """A hackathon theme/category.

    Attributes:
        id: Theme identifier.
        name: Theme display name.
    """

    __slots__ = ("id", "name")

    def __init__(self, *, id: int, name: str) -> None:
        """Initialize theme.

        Args:
            id: Theme identifier.
            name: Theme display name.
        """
        self.id = id
        self.name = name


def encode_theme(theme: Theme) -> JSONObject:
    """Encode Theme to JSON-serializable dict.

    Args:
        theme: Theme to encode.

    Returns:
        JSON-serializable dict.
    """
    result: JSONObject = {
        "id": theme.id,
        "name": theme.name,
    }
    return result


def decode_theme(data: JSONObject) -> Theme:
    """Decode Theme from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated Theme.

    Raises:
        JSONTypeError: If validation fails.
    """
    return Theme(
        id=require_int(data, "id"),
        name=require_str(data, "name"),
    )


# -----------------------------------------------------------------------------
# DisplayedLocation
# -----------------------------------------------------------------------------


class DisplayedLocation:
    """Location information for a hackathon.

    Attributes:
        icon: Icon name for display.
        location: Location text description.
    """

    __slots__ = ("icon", "location")

    def __init__(self, *, icon: str, location: str) -> None:
        """Initialize displayed location.

        Args:
            icon: Icon name for display.
            location: Location text description.
        """
        self.icon = icon
        self.location = location


def encode_displayed_location(loc: DisplayedLocation) -> JSONObject:
    """Encode DisplayedLocation to JSON-serializable dict.

    Args:
        loc: DisplayedLocation to encode.

    Returns:
        JSON-serializable dict.
    """
    result: JSONObject = {
        "icon": loc.icon,
        "location": loc.location,
    }
    return result


def decode_displayed_location(data: JSONObject) -> DisplayedLocation:
    """Decode DisplayedLocation from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated DisplayedLocation.

    Raises:
        JSONTypeError: If validation fails.
    """
    return DisplayedLocation(
        icon=require_str(data, "icon"),
        location=require_str(data, "location"),
    )
