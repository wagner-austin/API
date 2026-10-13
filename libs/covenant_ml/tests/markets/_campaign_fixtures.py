"""Real model artifacts and one-run campaigns for scheduler integration tests."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from platform_core.json_utils import load_json_str, narrow_json_to_dict

from covenant_ml.markets.campaign_planning import persist_campaign
from covenant_ml.markets.campaign_records import Campaign, FrozenModel
from covenant_ml.markets.paper_records import decode_paper_plan
from covenant_ml.markets.provenance import implementation_digest

from ._paper_fixtures import cli_inputs, paper_plan


def single_campaign(root: Path) -> tuple[Campaign, Path, str]:
    plan = paper_plan()
    cli_inputs(root, plan)
    frozen = decode_paper_plan(
        narrow_json_to_dict(load_json_str((root / "plan.json").read_text("utf-8")))
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
        plans=(frozen,),
    )
    destination = root / "campaign"
    digest = persist_campaign(campaign, destination)
    return campaign, destination, digest


__all__ = ["single_campaign"]
