"""Frozen monthly CRSP design, separate from acquisition and run evidence."""

from __future__ import annotations

from math import isfinite

from platform_core.json_utils import (
    JSONObject,
    require_float,
    require_int,
    require_list,
    require_str,
)
from typing_extensions import ReadOnly, TypedDict

from .campaign_records import require_digest
from .validation import require_bounded_int, require_day, require_nonempty


class CrspProtocol(TypedDict):
    """Exact subscribed sources, chronology and declared portfolio assumptions."""

    version: ReadOnly[int]
    library: ReadOnly[str]
    membership: ReadOnly[str]
    returns: ReadOnly[str]
    calendar: ReadOnly[str]
    delists: ReadOnly[str]
    membership_columns: ReadOnly[tuple[str, ...]]
    returns_columns: ReadOnly[tuple[str, ...]]
    calendar_columns: ReadOnly[tuple[str, ...]]
    delists_columns: ReadOnly[tuple[str, ...]]
    source_sha256: ReadOnly[str]
    history_start: ReadOnly[str]
    train_start: ReadOnly[str]
    validation_start: ReadOnly[str]
    holdout_start: ReadOnly[str]
    holdout_end: ReadOnly[str]
    coverage_end: ReadOnly[str]
    share_codes: ReadOnly[tuple[int, ...]]
    exchange_codes: ReadOnly[tuple[int, ...]]
    calendar_group: ReadOnly[str]
    execution_flag: ReadOnly[str]
    terminal_rule: ReadOnly[str]
    min_price: ReadOnly[float]
    min_history: ReadOnly[int]
    liquidity_window: ReadOnly[int]
    max_universe: ReadOnly[int]
    selection_fraction: ReadOnly[float]
    cost_bps: ReadOnly[tuple[int, ...]]
    bootstrap_block_months: ReadOnly[int]
    bootstrap_draws: ReadOnly[int]
    bootstrap_seed: ReadOnly[int]


def encode_crsp_protocol(record: CrspProtocol) -> JSONObject:
    """Encode every frozen field, with ordered arrays retained.

    Args:
        record: Validated study design.

    Returns:
        JSON suitable for the existing protocol digest and admission policy.
    """
    return {
        "version": record["version"],
        "library": record["library"],
        "membership": record["membership"],
        "returns": record["returns"],
        "calendar": record["calendar"],
        "delists": record["delists"],
        "membership_columns": list(record["membership_columns"]),
        "returns_columns": list(record["returns_columns"]),
        "calendar_columns": list(record["calendar_columns"]),
        "delists_columns": list(record["delists_columns"]),
        "source_sha256": record["source_sha256"],
        "history_start": record["history_start"],
        "train_start": record["train_start"],
        "validation_start": record["validation_start"],
        "holdout_start": record["holdout_start"],
        "holdout_end": record["holdout_end"],
        "coverage_end": record["coverage_end"],
        "share_codes": list(record["share_codes"]),
        "exchange_codes": list(record["exchange_codes"]),
        "calendar_group": record["calendar_group"],
        "execution_flag": record["execution_flag"],
        "terminal_rule": record["terminal_rule"],
        "min_price": record["min_price"],
        "min_history": record["min_history"],
        "liquidity_window": record["liquidity_window"],
        "max_universe": record["max_universe"],
        "selection_fraction": record["selection_fraction"],
        "cost_bps": list(record["cost_bps"]),
        "bootstrap_block_months": record["bootstrap_block_months"],
        "bootstrap_draws": record["bootstrap_draws"],
        "bootstrap_seed": record["bootstrap_seed"],
    }


def _require_fixed(document: JSONObject, key: str, expected: str) -> str:
    """Validate a source or convention supported by this version.

    Args:
        document: Untrusted protocol.
        key: Exact field name.
        expected: Supported source or convention.

    Returns:
        Validated value.

    Raises:
        ValueError: CRSP-001 when a different source or convention is requested.
    """
    value = require_nonempty(document, key)
    if value != expected:
        raise ValueError(f"[CRSP-001] {key} must be {expected}")
    return value


def _require_codes(document: JSONObject, key: str, expected: tuple[int, ...]) -> tuple[int, ...]:
    """Validate the exact ordered numeric universe or cost specification.

    Args:
        document: Untrusted protocol.
        key: Required array.
        expected: Supported codes, in canonical order.

    Returns:
        Validated immutable codes.

    Raises:
        ValueError: CRSP-001 for changed, repeated or reordered codes.
    """
    values = tuple(require_int({"code": item}, "code") for item in require_list(document, key))
    if values != expected:
        raise ValueError(f"[CRSP-001] {key} must be {expected}")
    return values


def _require_columns(document: JSONObject, key: str, expected: tuple[str, ...]) -> tuple[str, ...]:
    """Bind an extraction's exact ordered columns to the implemented field semantics.

    Args:
        document: Untrusted frozen design.
        key: Required column list field.
        expected: Columns consumed by this version of the evaluator.

    Returns:
        Exact ordered names, suitable for the extraction projection.

    Raises:
        ValueError: CRSP-001 if any column is absent, added, repeated or reordered.
    """
    columns = tuple(
        require_str({"column": value}, "column") for value in require_list(document, key)
    )
    if columns != expected:
        raise ValueError(f"[CRSP-001] {key} differs from the supported exact column set")
    return columns


def decode_crsp_protocol(document: JSONObject) -> CrspProtocol:
    """Refuse ambiguous sources, chronology, costs and unsupported conventions.

    Args:
        document: Untrusted frozen protocol JSON.

    Returns:
        Typed design. Its exact bytes/digest are bound by existing admission.

    Raises:
        ValueError: CRSP-001 for malformed design; shared validators propagate.
        TypeError: Missing or wrongly typed fields.
    """
    if set(document) != set(CrspProtocol.__annotations__):
        raise ValueError("[CRSP-001] require exact protocol fields")
    history, train, validation, holdout, end, coverage = (
        require_day(document, key)
        for key in (
            "history_start",
            "train_start",
            "validation_start",
            "holdout_start",
            "holdout_end",
            "coverage_end",
        )
    )
    price = require_float(document, "min_price")
    fraction = require_float(document, "selection_fraction")
    if not history < train < validation < holdout <= end < coverage:
        raise ValueError("[CRSP-001] history, development, holdout and coverage must be ordered")
    if not isfinite(price) or price <= 0 or not 0 < fraction <= 1:
        raise ValueError("[CRSP-001] price and selection fraction must be finite and positive")
    minimum = require_bounded_int(document, "min_history", 252, 10000)
    liquidity = require_bounded_int(document, "liquidity_window", 1, minimum)
    return CrspProtocol(
        version=require_bounded_int(document, "version", 1, 1),
        library=_require_fixed(document, "library", "crsp_a_stock"),
        membership=_require_fixed(document, "membership", "dsenames"),
        returns=_require_fixed(document, "returns", "stkdlysecuritydata"),
        calendar=_require_fixed(document, "calendar", "metaexchangecalendar"),
        delists=_require_fixed(document, "delists", "stkdelists"),
        membership_columns=_require_columns(
            document, "membership_columns", ("permno", "namedt", "nameendt", "shrcd", "exchcd")
        ),
        returns_columns=_require_columns(
            document,
            "returns_columns",
            (
                "permno",
                "dlycaldt",
                "dlyprevdt",
                "dlyret",
                "dlyretmissflg",
                "dlyretdurflg",
                "dlydelflg",
                "dlyprc",
                "dlyprcflg",
                "dlyvol",
            ),
        ),
        calendar_columns=_require_columns(
            document, "calendar_columns", ("caldt", "exchangegroup", "tradingflg")
        ),
        delists_columns=_require_columns(
            document,
            "delists_columns",
            (
                "permno",
                "delistingdt",
                "deldlydt",
                "delamtdt",
                "delnextdt",
                "delret",
                "delretmisstype",
            ),
        ),
        source_sha256=require_digest(document, "source_sha256"),
        history_start=history,
        train_start=train,
        validation_start=validation,
        holdout_start=holdout,
        holdout_end=end,
        coverage_end=coverage,
        share_codes=_require_codes(document, "share_codes", (10, 11)),
        exchange_codes=_require_codes(document, "exchange_codes", (1, 2, 3)),
        calendar_group=_require_fixed(document, "calendar_group", "CIZ"),
        execution_flag=_require_fixed(document, "execution_flag", "TR"),
        terminal_rule=_require_fixed(document, "terminal_rule", "dated-value-next-rebalance"),
        min_price=price,
        min_history=minimum,
        liquidity_window=liquidity,
        max_universe=require_bounded_int(document, "max_universe", 1, 1000),
        selection_fraction=fraction,
        cost_bps=_require_codes(document, "cost_bps", (10, 5, 25)),
        bootstrap_block_months=require_bounded_int(document, "bootstrap_block_months", 1, 36),
        bootstrap_draws=require_bounded_int(document, "bootstrap_draws", 2, 100000),
        bootstrap_seed=require_bounded_int(document, "bootstrap_seed", 0, 2**31 - 1),
    )
