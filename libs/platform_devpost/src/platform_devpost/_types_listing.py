"""platform_devpost Hackathon listing payloads."""

from __future__ import annotations

from platform_core.json_utils import (
    JSONObject,
    require_int,
    require_list,
)

from platform_devpost._types_hackathon import (
    Hackathon,
    decode_hackathon,
    encode_hackathon,
)
from platform_devpost._types_validation import _require_dict_value

# -----------------------------------------------------------------------------
# API Response Types
# -----------------------------------------------------------------------------


class HackathonListMeta:
    """Metadata for hackathon list response.

    Attributes:
        total_count: Total number of hackathons matching query.
        per_page: Number of hackathons per page.
    """

    __slots__ = ("per_page", "total_count")

    def __init__(self, *, total_count: int, per_page: int) -> None:
        """Initialize meta.

        Args:
            total_count: Total number of hackathons.
            per_page: Number per page.
        """
        self.total_count = total_count
        self.per_page = per_page


def encode_list_meta(meta: HackathonListMeta) -> JSONObject:
    """Encode HackathonListMeta to JSON-serializable dict.

    Args:
        meta: Meta to encode.

    Returns:
        JSON-serializable dict.
    """
    result: JSONObject = {
        "total_count": meta.total_count,
        "per_page": meta.per_page,
    }
    return result


def decode_list_meta(data: JSONObject) -> HackathonListMeta:
    """Decode HackathonListMeta from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated HackathonListMeta.

    Raises:
        JSONTypeError: If validation fails.
    """
    return HackathonListMeta(
        total_count=require_int(data, "total_count"),
        per_page=require_int(data, "per_page"),
    )


class HackathonListResponse:
    """Response from hackathon list API.

    Attributes:
        hackathons: Tuple of hackathons.
        meta: Pagination metadata.
    """

    __slots__ = ("hackathons", "meta")

    def __init__(
        self,
        *,
        hackathons: tuple[Hackathon, ...],
        meta: HackathonListMeta,
    ) -> None:
        """Initialize response.

        Args:
            hackathons: Tuple of hackathons.
            meta: Pagination metadata.
        """
        self.hackathons = hackathons
        self.meta = meta


def encode_list_response(resp: HackathonListResponse) -> JSONObject:
    """Encode HackathonListResponse to JSON-serializable dict.

    Args:
        resp: Response to encode.

    Returns:
        JSON-serializable dict.
    """
    result: JSONObject = {
        "hackathons": [encode_hackathon(h) for h in resp.hackathons],
        "meta": encode_list_meta(resp.meta),
    }
    return result


def decode_list_response(data: JSONObject) -> HackathonListResponse:
    """Decode HackathonListResponse from dict with validation.

    Args:
        data: JSON object to decode.

    Returns:
        Validated HackathonListResponse.

    Raises:
        JSONTypeError: If validation fails.
    """
    hackathons_raw = require_list(data, "hackathons")
    meta_raw = data.get("meta")

    return HackathonListResponse(
        hackathons=tuple(
            decode_hackathon(_require_dict_value(h, f"hackathons[{i}]"))
            for i, h in enumerate(hackathons_raw)
        ),
        meta=decode_list_meta(_require_dict_value(meta_raw, "meta")),
    )
