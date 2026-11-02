"""Real model artifacts and campaigns of one or more runs for scheduler integration tests."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

from platform_core.json_utils import load_json_str, narrow_json_to_dict

from covenant_ml.markets.campaign_planning import persist_campaign
from covenant_ml.markets.campaign_records import Campaign, FrozenModel
from covenant_ml.markets.paper_records import decode_paper_plan, encode_paper_plan
from covenant_ml.markets.provenance import implementation_digest

from ._paper_fixtures import cli_inputs, paper_plan


def single_campaign(root: Path, days: int = 1) -> tuple[Campaign, Path, str]:
    """Persist a campaign whose plans share one capture window, one event day apiece.

    The first plan is the frozen ``paper-test`` plan; each further plan is that plan
    on the next event day, named for its offset.
    """
    plan = paper_plan()
    cli_inputs(root, plan)
    frozen = decode_paper_plan(
        narrow_json_to_dict(load_json_str((root / "plan.json").read_text("utf-8")))
    )
    first = date.fromisoformat(frozen["event_day"])
    later = tuple(
        decode_paper_plan(
            {
                **encode_paper_plan(frozen),
                "name": f"{frozen['name']}-{offset}",
                "event_day": (first + timedelta(days=offset)).isoformat(),
            }
        )
        for offset in range(1, days)
    )
    campaign = Campaign(
        name="test-campaign",
        created_at=(
            datetime.fromisoformat(frozen["window_start"]) - timedelta(minutes=1)
        ).isoformat(),
        implementation_digest=implementation_digest(),
        protocol_path=str(root / "protocol.json"),
        protocol_digest=frozen["protocol_digest"],
        models=tuple(
            FrozenModel(hour=hour, path=str(root / "model.json"), digest=frozen["model_digest"])
            for hour in (0, 12)
        ),
        plans=(frozen, *later),
    )
    destination = root / "campaign"
    digest = persist_campaign(campaign, destination)
    return campaign, destination, digest


__all__ = ["single_campaign"]
