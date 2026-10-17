"""Shared frozen input and acquisition-window validation for execution and analysis."""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from hashlib import sha256
from pathlib import Path

from platform_core.json_utils import dump_json_str, load_json_str, narrow_json_to_dict

from . import _test_hooks
from .campaign_records import Campaign, decode_campaign
from .evidence_location import locate_evidence
from .paper_records import PaperPlan, encode_paper_plan
from .validation import parse_timestamp


def load_campaign_inputs(root: Path, digest: str) -> Campaign:
    """Verify manifest, model reports, protocol and exact persisted plans.

    Args:
        root: Persisted campaign directory.
        digest: Original manifest digest pinned outside that directory.

    Returns:
        Validated immutable campaign; this reader need not be the frozen collector.

    Raises:
        ValueError: CAMPAIGN-007 when a frozen input changed.
        OSError: A required source is absent.
    """
    payload = (root / "campaign.json").read_bytes()
    if sha256(payload).hexdigest() != digest:
        raise ValueError("[CAMPAIGN-007] campaign manifest differs from scheduler identity")
    campaign = decode_campaign(narrow_json_to_dict(load_json_str(payload.decode("utf-8"))))
    relocations = _test_hooks.evidence_relocations()
    artifacts = [(campaign["protocol_path"], campaign["protocol_digest"])]
    artifacts.extend((model["path"], model["digest"]) for model in campaign["models"])
    for path, expected in artifacts:
        if sha256(locate_evidence(path, relocations).read_bytes()).hexdigest() != expected:
            raise ValueError("[CAMPAIGN-007] frozen protocol or model artifact changed")
    for plan in campaign["plans"]:
        expected_text = dump_json_str(encode_paper_plan(plan), indent=2)
        if (root / "plans" / (plan["name"] + ".json")).read_bytes() != expected_text.encode():
            raise ValueError("[CAMPAIGN-007] persisted acquisition plan changed")
    return campaign


def acquisition_window(plan: PaperPlan, operation: str) -> tuple[int, int]:
    """Resolve capture timing or the fixed two-days-later settlement window.

    Args:
        plan: Frozen acquisition plan.
        operation: Validated capture or settlement operation.

    Returns:
        Inclusive start and end in epoch milliseconds.

    Raises:
        ValueError: CAMPAIGN-015 for an unknown acquisition operation.
    """
    if operation not in ("capture", "settlement"):
        raise ValueError("[CAMPAIGN-015] unknown acquisition operation")
    if operation == "capture":
        return parse_timestamp(plan["window_start"]), parse_timestamp(plan["window_end"])
    day = date.fromisoformat(plan["event_day"]) + timedelta(days=2)
    start = datetime.combine(day, time(15), UTC)
    return parse_timestamp(start.isoformat()), parse_timestamp(
        (start + timedelta(hours=1)).isoformat()
    )


__all__ = ["acquisition_window", "load_campaign_inputs"]
