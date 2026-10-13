"""Freeze daily decision windows and input artifacts before prospective acquisition."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from hashlib import sha256
from pathlib import Path
from zoneinfo import ZoneInfo

from platform_core.json_utils import JSONObject, dump_json_str

from .campaign_records import Campaign, FrozenModel, decode_campaign, encode_campaign
from .forecast_records import CITY_COORDINATES
from .paper_records import PaperPlan, encode_paper_plan


def write_new_json(path: Path, document: JSONObject) -> None:
    """Persist JSON exclusively, refusing any overwrite.

    Args:
        path: New destination with an existing parent directory.
        document: JSON payload.

    Raises:
        OSError: Creation or writing fails, including an existing destination.
    """
    with path.open("x", encoding="utf-8", newline="\n") as destination:
        destination.write(dump_json_str(document, indent=2))


def plan_campaign(
    name: str,
    first: date,
    last: date,
    created_at: str,
    protocol: Path,
    models: tuple[Path, Path],
    implementation: str,
) -> Campaign:
    """Freeze both cycles in three cities at 16:00 Eastern on each prior day.

    Args:
        name: Study identity.
        first: First event date, inclusive.
        last: Last event date, inclusive, at most 366 days after first.
        created_at: Actual manifest preparation timestamp.
        protocol: Frozen research protocol.
        models: Fitted 00 and 12 UTC model reports in that order.
        implementation: Exact collector implementation digest.

    Returns:
        Validated campaign with 15-minute acquisition windows and no fitted changes.

    Raises:
        ValueError: CAMPAIGN-006 for an invalid date range or manifest validation failure.
        OSError: A frozen source cannot be read.
    """
    days = (last - first).days
    if not 0 <= days <= 366:
        raise ValueError("[CAMPAIGN-006] require an ordered date range of at most 367 days")
    frozen_models = tuple(
        FrozenModel(
            hour=hour, path=str(path.resolve()), digest=sha256(path.read_bytes()).hexdigest()
        )
        for hour, path in zip((0, 12), models, strict=True)
    )
    protocol_digest = sha256(protocol.read_bytes()).hexdigest()
    plans: list[PaperPlan] = []
    for offset in range(days + 1):
        day = first + timedelta(days=offset)
        start = datetime.combine(
            day - timedelta(days=1), time(16), ZoneInfo("America/New_York")
        ).astimezone(UTC)
        for series in CITY_COORDINATES:
            for model in frozen_models:
                plans.append(
                    PaperPlan(
                        name=f"{day.isoformat()}-{series}-{model['hour']:02d}",
                        series=series,
                        event_day=day.isoformat(),
                        forecast_hour=model["hour"],
                        window_start=start.isoformat(),
                        window_end=(start + timedelta(minutes=15)).isoformat(),
                        model_digest=model["digest"],
                        protocol_digest=protocol_digest,
                    )
                )
    return decode_campaign(
        encode_campaign(
            Campaign(
                name=name,
                created_at=created_at,
                implementation_digest=implementation,
                protocol_path=str(protocol.resolve()),
                protocol_digest=protocol_digest,
                models=frozen_models,
                plans=tuple(plans),
            )
        )
    )


def persist_campaign(campaign: Campaign, root: Path) -> str:
    """Create all frozen plans and write the manifest last as the completion marker.

    Args:
        campaign: Validated pre-acquisition manifest.
        root: New campaign directory; partial or existing directories are refused.

    Returns:
        Manifest digest to pin in the scheduler action.

    Raises:
        OSError: Directory creation, exclusive writes or evidence reads fail.
    """
    root.mkdir(parents=True, exist_ok=False)
    plans = root / "plans"
    plans.mkdir()
    for plan in campaign["plans"]:
        write_new_json(plans / (plan["name"] + ".json"), encode_paper_plan(plan))
    path = root / "campaign.json"
    write_new_json(path, encode_campaign(campaign))
    return sha256(path.read_bytes()).hexdigest()


__all__ = ["persist_campaign", "plan_campaign", "write_new_json"]
