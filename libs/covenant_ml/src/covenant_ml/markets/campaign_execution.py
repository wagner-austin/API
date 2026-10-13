"""Execute each frozen acquisition once with durable failure and missed-window evidence."""

from __future__ import annotations

import sqlite3
import sys
from collections.abc import Callable
from contextlib import closing
from datetime import UTC, date, datetime, time, timedelta
from hashlib import sha256
from pathlib import Path

from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
)

from .campaign_planning import write_new_json
from .campaign_process import CommandRunner
from .campaign_records import (
    Attempt,
    AttemptState,
    Campaign,
    decode_attempt,
    decode_campaign,
    encode_attempt,
)
from .paper_records import PaperPlan, encode_paper_plan
from .paper_scoring import load_verified_predictions
from .provenance import implementation_digest
from .reporting import read_verified_market_report
from .validation import parse_timestamp, read_document


def load_campaign(root: Path, digest: str) -> Campaign:
    """Verify the pinned manifest, implementation, model reports, protocol and plans.

    Args:
        root: Persisted campaign directory.
        digest: Original manifest digest pinned outside that directory.

    Returns:
        Validated campaign.

    Raises:
        ValueError: CAMPAIGN-007 when any frozen input changed.
        OSError: A required source is absent.
    """
    payload = (root / "campaign.json").read_bytes()
    if sha256(payload).hexdigest() != digest:
        raise ValueError("[CAMPAIGN-007] campaign manifest differs from scheduler identity")
    campaign = decode_campaign(narrow_json_to_dict(load_json_str(payload.decode("utf-8"))))
    if campaign["implementation_digest"] != implementation_digest():
        raise ValueError("[CAMPAIGN-007] collector implementation changed after campaign freeze")
    artifacts = [(campaign["protocol_path"], campaign["protocol_digest"])]
    artifacts.extend((model["path"], model["digest"]) for model in campaign["models"])
    for path, expected in artifacts:
        if sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError("[CAMPAIGN-007] frozen protocol or model artifact changed")
    for plan in campaign["plans"]:
        expected_text = dump_json_str(encode_paper_plan(plan), indent=2)
        if (root / "plans" / (plan["name"] + ".json")).read_bytes() != expected_text.encode():
            raise ValueError("[CAMPAIGN-007] persisted acquisition plan changed")
    return campaign


def _window(plan: PaperPlan, operation: str) -> tuple[int, int]:
    """Return the capture window or fixed two-days-later settlement window.

    Args:
        plan: Frozen acquisition plan.
        operation: Capture or settlement operation selected by the runner.

    Returns:
        Inclusive start and end in epoch milliseconds.
    """
    if operation == "capture":
        return parse_timestamp(plan["window_start"]), parse_timestamp(plan["window_end"])
    day = date.fromisoformat(plan["event_day"]) + timedelta(days=2)
    start = datetime.combine(day, time(15), UTC)
    return parse_timestamp(start.isoformat()), parse_timestamp(
        (start + timedelta(hours=1)).isoformat()
    )


def _command(
    root: Path,
    campaign: Campaign,
    plan: PaperPlan,
    operation: str,
    market_url: str,
    forecast_url: str,
) -> tuple[str, ...]:
    """Build an argument vector for the existing capture or settlement CLI.

    Args:
        root: Campaign evidence directory.
        campaign: Validated input manifest.
        plan: Frozen acquisition plan.
        operation: Capture or settlement operation.
        market_url: Explicit public market origin.
        forecast_url: Explicit forecast origin.

    Returns:
        Immutable arguments including the current Python interpreter.
    """
    name = plan["name"]
    base = (sys.executable, "-m")
    output = root / "reports" / f"{name}-{operation}.json"
    archive = root / "archives" / f"{name}-{operation}"
    tail = (
        "--archive",
        str(archive),
        "--out",
        str(output),
        "--mode",
        "capture",
        "--market-url",
        market_url,
    )
    if operation == "settlement":
        return (
            *base,
            "scripts.paper_settlements",
            "--predictions",
            str(root / "reports" / f"{name}-capture.json"),
            *tail,
        )
    models = {model["hour"]: model["path"] for model in campaign["models"]}
    return (
        *base,
        "scripts.paper_research",
        "--plan",
        str(root / "plans" / (name + ".json")),
        "--protocol",
        campaign["protocol_path"],
        "--models",
        models[plan["forecast_hour"]],
        "--forecast-url",
        forecast_url,
        *tail,
    )


def _record(
    path: Path,
    plan: PaperPlan,
    operation: str,
    state: AttemptState,
    observed: str,
    code: int | None = None,
    digest: str | None = None,
) -> None:
    """Validate and exclusively persist a terminal attempt result.

    Args:
        path: New result destination.
        plan: Frozen acquisition identity.
        operation: Capture or settlement operation.
        state: Terminal attempt disposition.
        observed: Actual UTC observation timestamp.
        code: Process return code, absent for an unexecuted attempt.
        digest: Completed report hash, absent without a verified report.

    Raises:
        ValueError: Attempt fields are inconsistent.
        OSError: Exclusive persistence fails.
    """
    attempt = decode_attempt(
        encode_attempt(
            Attempt(
                name=plan["name"],
                operation=operation,
                state=state,
                observed_at=observed,
                returncode=code,
                report_digest=digest,
            )
        )
    )
    write_new_json(path, encode_attempt(attempt))


def tick_campaign(
    root: Path,
    digest: str,
    clock: Callable[[], str],
    execute: CommandRunner,
    market_url: str,
    forecast_url: str,
) -> JSONObject:
    """Run each due acquisition once, propagating failures after recording their status.

    Args:
        root: Absolute frozen campaign directory.
        digest: Manifest digest pinned in the scheduler action.
        clock: Actual UTC receipt clock or controlled test clock.
        execute: Real process runner, directly injected.
        market_url: Public market API origin.
        forecast_url: Explicit-run forecast API origin.

    Returns:
        Counts of recorded outcomes and future jobs for this complete tick.

    Raises:
        RuntimeError: CAMPAIGN-009 for interrupted attempts or CAMPAIGN-010 for nonzero exits.
        ValueError: Frozen input or completed evidence changed.
        OSError: Required evidence operations or process launch failed.
        TimeoutExpired: A command exceeded its limit; its started marker remains.
        OperationalError: Another tick holds the campaign's exclusive SQLite transaction.
    """
    campaign = load_campaign(root, digest)
    for directory in ("attempts", "reports", "archives"):
        (root / directory).mkdir(exist_ok=True)
    with closing(sqlite3.connect(root / "runner.sqlite", timeout=0)) as mutex, mutex:
        mutex.execute("BEGIN IMMEDIATE")
        totals = {state.value: 0 for state in AttemptState}
        totals["future"] = 0
        for plan in campaign["plans"]:
            for operation in ("capture", "settlement"):
                disposition = _run_attempt(
                    root,
                    digest,
                    campaign,
                    plan,
                    operation,
                    clock,
                    execute,
                    market_url,
                    forecast_url,
                )
                totals[disposition] += 1
        return dict(totals)


def _existing_result(path: Path, report: Path, plan: PaperPlan, operation: str) -> str:
    """Verify previous terminal evidence without repeating an acquisition.

    Args:
        path: Existing result record.
        report: Expected completed report path.
        plan: Frozen acquisition identity.
        operation: Expected operation.

    Returns:
        Verified terminal disposition name.

    Raises:
        ValueError: CAMPAIGN-008 for changed identity or evidence.
        OSError: Required evidence is unavailable.
    """
    result = decode_attempt(read_document(path))
    if result["name"] != plan["name"] or result["operation"] != operation:
        raise ValueError("[CAMPAIGN-008] attempt identity changed")
    if result["state"] is AttemptState.COMPLETE:
        _, _, digest = read_verified_market_report(report, "CAMPAIGN-008")
        if digest != result["report_digest"]:
            raise ValueError("[CAMPAIGN-008] completed attempt report changed")
    return result["state"].value


def _run_attempt(
    root: Path,
    digest: str,
    campaign: Campaign,
    plan: PaperPlan,
    operation: str,
    clock: Callable[[], str],
    execute: CommandRunner,
    market_url: str,
    forecast_url: str,
) -> str:
    """Classify or execute one acquisition, preserving every failed start.

    Args:
        root: Campaign evidence directory.
        digest: Pinned manifest identity.
        campaign: Validated input manifest.
        plan: Frozen acquisition specification.
        operation: Capture or settlement operation.
        clock: UTC observation clock.
        execute: Injected process runner.
        market_url: Public market origin.
        forecast_url: Explicit-run forecast origin.

    Returns:
        Terminal disposition or future for a window that has not opened.

    Raises:
        RuntimeError: CAMPAIGN-009 for interruption or CAMPAIGN-010 for process failure.
        ValueError: Prior evidence or subprocess output is invalid.
        OSError: Evidence or process operations fail.
    """
    name = f"{plan['name']}-{operation}"
    result_path = root / "attempts" / (name + ".json")
    report_path = root / "reports" / (name + ".json")
    if result_path.exists():
        return _existing_result(result_path, report_path, plan, operation)
    observed = clock()
    now = parse_timestamp(observed)
    start, end = _window(plan, operation)
    marker = root / "attempts" / (name + ".started.json")
    if marker.exists():
        _record(result_path, plan, operation, AttemptState.INTERRUPTED, observed)
        raise RuntimeError("[CAMPAIGN-009] interrupted acquisition preserved; no repeat")
    if now < start:
        return "future"
    if now > end:
        _record(result_path, plan, operation, AttemptState.MISSED, observed)
        return "missed"
    if operation == "settlement":
        original = root / "attempts" / (plan["name"] + "-capture.json")
        captured = decode_attempt(read_document(original))
        if captured["state"] is not AttemptState.COMPLETE:
            _record(result_path, plan, operation, AttemptState.DEPENDENCY_MISSING, observed)
            return "dependency_missing"
    write_new_json(marker, {"campaign_digest": digest, "started_at": observed})
    command = _command(root, campaign, plan, operation, market_url, forecast_url)
    code = execute(command, Path(__file__).parents[3], root / "attempts" / (name + ".log"))
    if code != 0:
        _record(result_path, plan, operation, AttemptState.FAILED, clock(), code)
        raise RuntimeError(f"[CAMPAIGN-010] acquisition exited {code}; see {name}.log")
    if operation == "capture":
        load_verified_predictions(report_path)
    _, _, report_digest = read_verified_market_report(report_path, "CAMPAIGN-008")
    _record(result_path, plan, operation, AttemptState.COMPLETE, clock(), 0, report_digest)
    return "complete"


__all__ = ["load_campaign", "tick_campaign"]
