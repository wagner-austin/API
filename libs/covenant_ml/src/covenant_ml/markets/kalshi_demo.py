"""Authenticated, read-only demo preflight; never grants permission to submit orders."""

from __future__ import annotations

import re
from collections.abc import Callable

from platform_core.http_client import HttpxClient
from platform_core.json_utils import (
    JSONObject,
    require_list,
    require_str,
)
from typing_extensions import ReadOnly, TypedDict

from .kalshi_auth import KalshiSigner
from .kalshi_transport import DEMO_ORIGIN, DemoTransport
from .validation import parse_fixed, parse_timestamp, require_bounded_int, require_nonempty

_MAX_INT = 2**63 - 1


class DemoPreflight(TypedDict):
    """Primary-account observations, with exact microdollars and no trade authority."""

    credential_fingerprint: ReadOnly[str]
    balance_microdollars: ReadOnly[int]
    portfolio_value_cents: ReadOnly[int]
    balance_updated_seconds: ReadOnly[int]
    started_ms: ReadOnly[int]
    completed_ms: ReadOnly[int]


def encode_preflight(row: DemoPreflight) -> JSONObject:
    """Serialize validated preflight observations.

    Args:
        row: Immutable DemoPreflight evidence.

    Returns:
        JSON object retaining exact integer units.
    """
    return {
        "credential_fingerprint": row["credential_fingerprint"],
        "balance_microdollars": row["balance_microdollars"],
        "portfolio_value_cents": row["portfolio_value_cents"],
        "balance_updated_seconds": row["balance_updated_seconds"],
        "started_ms": row["started_ms"],
        "completed_ms": row["completed_ms"],
    }


def decode_preflight(document: JSONObject) -> DemoPreflight:
    """Validate observation identity, money and a bounded collection interval.

    Args:
        document: JSON object from the account boundary or saved evidence.

    Returns:
        DemoPreflight with a zero portfolio and at most thirty seconds of reads.

    Raises:
        ValueError: KALSHI-DEMO-001 for an invalid fingerprint or collection interval.
        ValueError: MARKET-001 for invalid balances or nonzero portfolio value.
        TypeError: A required field is missing or incorrectly typed.
    """
    fingerprint = require_nonempty(document, "credential_fingerprint")
    started = require_bounded_int(document, "started_ms", 0, _MAX_INT)
    completed = require_bounded_int(document, "completed_ms", 0, _MAX_INT)
    updated = require_bounded_int(document, "balance_updated_seconds", 0, completed // 1000)
    if re.fullmatch("[0-9a-f]{64}", fingerprint) is None or not 0 <= completed - started <= 30000:
        raise ValueError("[KALSHI-DEMO-001] invalid fingerprint or preflight collection interval")
    return DemoPreflight(
        credential_fingerprint=fingerprint,
        balance_microdollars=require_bounded_int(document, "balance_microdollars", 0, _MAX_INT),
        portfolio_value_cents=require_bounded_int(document, "portfolio_value_cents", 0, 0),
        balance_updated_seconds=updated,
        started_ms=started,
        completed_ms=completed,
    )


class DemoAccountClient:
    """Read demo primary-account evidence using a supplied HTTP client and clock."""

    def __init__(
        self,
        client: HttpxClient,
        signer: KalshiSigner,
        clock: Callable[[], str],
        *,
        origin: str = DEMO_ORIGIN,
    ) -> None:
        """Bind dependencies to demo or a real loopback integration-test server.

        Args:
            client: Caller-owned HTTP client, with redirects disabled.
            signer: Loaded API key signing service.
            clock: Aware ISO wall-clock reader.
            origin: Demo origin or HTTP 127.0.0.1 origin for socket integration tests.

        Raises:
            ValueError: KALSHI-DEMO-002 for every other origin, including production.
        """
        self._transport = DemoTransport(client, signer, clock, origin=origin)
        self._signer = signer
        self._clock = clock

    def preflight(self) -> DemoPreflight:
        """Inspect an idle primary demo account before any execution integration.

        This is sequential evidence, not an atomic account snapshot. Other account
        activity may race these reads. Success never authorizes order submission.

        Returns:
            Validated evidence with exact balances and credential fingerprint.

        Raises:
            ValueError: KALSHI-DEMO-004 for returned orders, positions or a cursor.
            ValueError: KALSHI-DEMO-005 for balance changes during collection.
            ValueError: KALSHI-DEMO-006 for inconsistent cent and dollar balances.
            TypeError: Missing or malformed required response fields.
            HTTPError: An authenticated request fails at the transport boundary.
        """
        started = parse_timestamp(self._clock())
        first = self._balance()
        orders = self._transport.get("/portfolio/orders", {"subaccount": 0, "limit": 1})
        self._require_empty(orders, ("orders",))
        positions = self._transport.get(
            "/portfolio/positions", {"subaccount": 0, "limit": 1, "settlement_status": "unsettled"}
        )
        self._require_empty(positions, ("market_positions", "event_positions"))
        last = self._balance()
        completed = parse_timestamp(self._clock())
        if first != last:
            raise ValueError("[KALSHI-DEMO-005] account balance changed during preflight")
        balance, portfolio, updated = last
        return decode_preflight(
            {
                "credential_fingerprint": self._signer.fingerprint,
                "balance_microdollars": balance,
                "portfolio_value_cents": portfolio,
                "balance_updated_seconds": updated,
                "started_ms": started,
                "completed_ms": completed,
            }
        )

    def _balance(self) -> tuple[int, int, int]:
        """Validate primary-account aggregate balance across all exchange shards.

        Returns:
            Exact microdollars, portfolio cents and last-update epoch seconds.

        Raises:
            ValueError: KALSHI-DEMO-006 if cents and dollar strings disagree.
            ValueError: Invalid native monetary values.
        """
        document = self._transport.get("/portfolio/balance", {"subaccount": 0})
        cents = require_bounded_int(document, "balance", 0, _MAX_INT)
        microdollars = parse_fixed(require_str(document, "balance_dollars"), 6)
        if microdollars // 10000 != cents:
            raise ValueError("[KALSHI-DEMO-006] native cent and dollar balances disagree")
        return (
            microdollars,
            require_bounded_int(document, "portfolio_value", 0, _MAX_INT),
            require_bounded_int(document, "updated_ts", 0, _MAX_INT),
        )

    @staticmethod
    def _require_empty(document: JSONObject, fields: tuple[str, ...]) -> None:
        """Require explicit empty evidence, refusing incomplete or populated pages.

        Args:
            document: Native account page.
            fields: Required account record arrays.

        Raises:
            ValueError: KALSHI-DEMO-004 for records or a continuation cursor.
            TypeError: Required arrays or the explicit cursor are missing.
        """
        if require_str(document, "cursor") != "":
            raise ValueError("[KALSHI-DEMO-004] account evidence has a continuation cursor")
        for field in fields:
            if require_list(document, field):
                raise ValueError(f"[KALSHI-DEMO-004] account evidence contains {field}")
