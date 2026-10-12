"""Factories and collection orchestration; no order placement interface exists."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from enum import StrEnum
from pathlib import Path

from platform_core.http_client import HttpxClient
from platform_core.json_utils import JSONObject

from . import _test_hooks
from .archive import ResponseArchive
from .collection import MarketCollector
from .experiment import ExperimentPlan
from .kalshi import MarketSpec
from .types import MarketHistory


class AcquisitionMode(StrEnum):
    """Explicit fresh acquisition, checkpoint continuation, or network-free replay."""

    CAPTURE = "capture"
    RESUME = "resume"
    REPLAY = "replay"


def build_collector(
    root: Path, base_url: str, mode: AcquisitionMode
) -> tuple[MarketCollector, HttpxClient]:
    """Inject the real HTTP client, clock and pacing hooks into a focused collector.

    Args:
        root: Raw evidence directory.
        base_url: Public API origin and prefix.
        mode: Explicit acquisition operation.

    Returns:
        Collector and owned client, which the caller must close.
    """
    client = _test_hooks.client_factory(60.0, None)
    archive = ResponseArchive(root, base_url, client, _test_hooks.clock, _test_hooks.pause)
    readers: dict[AcquisitionMode, Callable[[str, dict[str, str | int]], JSONObject]] = {
        AcquisitionMode.CAPTURE: archive.capture,
        AcquisitionMode.RESUME: archive.resume_capture,
        AcquisitionMode.REPLAY: archive.replay,
    }
    return MarketCollector(readers[mode]), client


def collect_panel(
    plan: ExperimentPlan, collector: MarketCollector, workers: int, progress: Callable[[str], None]
) -> tuple[MarketHistory, ...]:
    """Collect a complete deterministic panel, retaining resumable request checkpoints.

    Args:
        plan: Validated frozen experiment.
        collector: Configured capture or replay service.
        workers: Positive number of concurrent independent contract requests.
        progress: Status sink; sees counts but no trading performance.

    Returns:
        Histories in canonical inventory order, across every declared series.

    Raises:
        ValueError: MARKET-028 for invalid concurrency, or invalid source evidence.
        HTTPStatusError: Any failed response aborts the run without publishing results.
    """
    if not 1 <= workers <= 4:
        raise ValueError("[MARKET-028] workers must be between one and four")
    cutoff = collector.read("/historical/cutoff", {})
    histories: list[MarketHistory] = []

    def collect(market: MarketSpec) -> MarketHistory:
        return collector.history(market, cutoff, plan["rule"])

    for series in plan["series"]:
        collector.verify_fee(series)
        inventory = collector.inventory(
            series,
            plan["timezone"],
            date.fromisoformat(plan["start"]),
            date.fromisoformat(plan["end"]),
        )
        progress(f"{series}: {len(inventory)} contracts")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for index, history in enumerate(pool.map(collect, inventory), 1):
                histories.append(history)
                if index % 100 == 0 or index == len(inventory):
                    progress(f"{series}: {index}/{len(inventory)} histories verified")
    return tuple(histories)


__all__ = ["AcquisitionMode", "build_collector", "collect_panel"]
