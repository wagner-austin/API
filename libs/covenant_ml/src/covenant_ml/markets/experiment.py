"""Frozen experiment contract, provenance and dependence-aware result emission."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from datetime import date
from hashlib import sha256
from pathlib import Path

from platform_core.comparability import RunFingerprint
from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    dump_json_str,
    narrow_json_to_dict,
    require_dict,
    require_list,
)
from platform_core.run_record import Observation
from typing_extensions import ReadOnly, TypedDict

from .codecs import decode_rule, encode_rule
from .replay import replay_markets
from .replay_codecs import encode_result
from .reporting import write_market_report
from .statistics import encode_summary, summarize_replay
from .types import DOLLAR, SCHEMA_VERSION, MakerRule, MarketHistory
from .validation import require_bounded_int, require_day, require_nonempty


class ExperimentPlan(TypedDict):
    """Configurable economic parameters of a single fixed-window replication."""

    name: ReadOnly[str]
    series: ReadOnly[tuple[str, ...]]
    timezone: ReadOnly[str]
    start: ReadOnly[str]
    end: ReadOnly[str]
    rule: ReadOnly[MakerRule]
    friction: ReadOnly[int]
    protocol_digest: ReadOnly[str]


def encode_plan(plan: ExperimentPlan) -> JSONObject:
    """Encode a complete experiment configuration.

    Args:
        plan: Validated plan.

    Returns:
        JSON object.
    """
    series: list[JSONValue] = list(plan["series"])
    return {
        "name": plan["name"],
        "series": series,
        "timezone": plan["timezone"],
        "start": plan["start"],
        "end": plan["end"],
        "rule": encode_rule(plan["rule"]),
        "friction": plan["friction"],
        "protocol_digest": plan["protocol_digest"],
    }


def decode_plan(document: JSONObject) -> ExperimentPlan:
    """Require all parameters, unique series and a canonical source-protocol hash.

    Args:
        document: JSON object.

    Returns:
        Immutable experiment plan.

    Raises:
        ValueError: MARKET-025 for invalid dates, series or protocol digest.
        TypeError: Missing or incorrectly typed fields.
    """
    series = tuple(
        require_nonempty({"series": row}, "series") for row in require_list(document, "series")
    )
    start, end = require_day(document, "start"), require_day(document, "end")
    digest = require_nonempty(document, "protocol_digest")
    if (
        not series
        or len(set(series)) != len(series)
        or start > end
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
    ):
        raise ValueError("[MARKET-025] invalid series, date window or protocol SHA-256")
    return ExperimentPlan(
        name=require_nonempty(document, "name"),
        series=series,
        timezone=require_nonempty(document, "timezone"),
        start=start,
        end=end,
        rule=decode_rule(require_dict(document, "rule")),
        friction=require_bounded_int(document, "friction", 0, DOLLAR),
        protocol_digest=digest,
    )


def evaluate_panel(histories: Sequence[MarketHistory], plan: ExperimentPlan) -> JSONObject:
    """Evaluate the frozen rule plus declared touch/one-cent-through sensitivities.

    Args:
        histories: Complete unique-contract panel for every declared series.
        plan: Frozen plan; primary threshold must be two cents.

    Returns:
        Encoded results, all per-contract outcomes, pooled and descriptive summaries.

    Raises:
        ValueError: MARKET-026 for a mismatched panel or changed primary threshold.
    """
    if {h["series"] for h in histories} != set(plan["series"]) or plan["rule"]["through"] != 200:
        raise ValueError(
            "[MARKET-026] panel must cover declared series and primary two-cent threshold"
        )
    start, end = date.fromisoformat(plan["start"]), date.fromisoformat(plan["end"])
    arms: list[JSONValue] = []
    for through in (200, 100, 0):
        rule = encode_rule(plan["rule"])
        rule["through"] = through
        results = replay_markets(histories, decode_rule(rule))
        summary = summarize_replay(results, start, end, plan["friction"])
        per_order = summary["per_order"]
        candidate = per_order is not None and per_order["ci_low"] > 0
        by_city: JSONObject = {}
        for series in plan["series"]:
            by_city[series] = encode_summary(
                summarize_replay(
                    [r for r in results if r["series"] == series], start, end, plan["friction"]
                )
            )
        monthly: JSONObject = {}
        for month in sorted({r["event_day"][:7] for r in results}):
            rows = [r for r in results if r["event_day"].startswith(month)]
            monthly[month] = {
                "pnl": sum(r["pnl"] for r in rows),
                "contracts": sum(r["contracts"] for r in rows),
            }
        encoded_results: list[JSONValue] = [encode_result(r) for r in results]
        counts: JSONObject = dict(Counter(r["status"].value for r in results))
        arms.append(
            {
                "through": through,
                "primary": through == 200,
                "candidate_gate": candidate,
                "summary": encode_summary(summary),
                "by_city": by_city,
                "monthly": monthly,
                "exclusions": counts,
                "orders": encoded_results,
            }
        )
    return {
        "plan": encode_plan(plan),
        "schema_version": SCHEMA_VERSION,
        "results": arms,
        "interpretation": "Historical replication with hypothetical fills; not executable profit.",
    }


def write_report(
    report: JSONObject,
    plan: ExperimentPlan,
    fingerprint: RunFingerprint,
    output: Path,
    archive_paths: Sequence[Path],
) -> Path:
    """Bind the complete report and source archive to an actual shared RunRecord.

    Args:
        report: Result of evaluate_panel.
        plan: Frozen plan used for the report.
        fingerprint: Captured arithmetic environment.
        output: New report path; overwriting is refused.
        archive_paths: Exact raw-response files used by collection.

    Returns:
        Run-record sidecar path.

    Raises:
        FileExistsError: Report or sidecar already exists.
        OSError: A source cannot be read or output cannot be persisted.
    """
    observations: list[Observation] = []
    for item in require_list(report, "results"):
        arm = narrow_json_to_dict(item)
        threshold = require_bounded_int(arm, "through", 0, DOLLAR)
        summary = require_dict(arm, "summary")
        observations.append(
            Observation(
                name=f"through_{threshold}_pnl",
                value=require_bounded_int(summary, "pnl", -(2**63), 2**63 - 1) / DOLLAR,
            )
        )
    return write_market_report(
        report,
        fingerprint,
        output,
        archive_paths,
        plan["name"],
        sha256(dump_json_str(encode_plan(plan)).encode()).hexdigest(),
        tuple(observations),
    )


__all__ = ["ExperimentPlan", "decode_plan", "encode_plan", "evaluate_panel", "write_report"]
