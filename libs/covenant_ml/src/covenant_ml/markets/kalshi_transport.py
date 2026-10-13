"""Signed demo HTTP boundary with explicit status requirements and no retries."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping

from platform_core.http_client import HttpxClient, HttpxResponse
from platform_core.json_utils import JSONObject, load_json_str, narrow_json_to_dict

from .execution_records import OrderIntent, order_payload
from .kalshi_auth import KalshiSigner
from .validation import parse_timestamp

DEMO_ORIGIN = "https://external-api.demo.kalshi.co"
_PREFIX = "/trade-api/v2"


class DemoTransport:
    """Own request authentication and restrict every request to demo or loopback."""

    def __init__(
        self,
        client: HttpxClient,
        signer: KalshiSigner,
        clock: Callable[[], str],
        *,
        origin: str = DEMO_ORIGIN,
    ) -> None:
        """Bind the real HTTP client, signer and clock.

        Args:
            client: Caller-owned client with redirects disabled.
            signer: Validated signing key.
            clock: Aware ISO wall-clock reader.
            origin: Demo origin or a real loopback integration-test server.

        Raises:
            ValueError: KALSHI-DEMO-002 for every other origin.
        """
        if (
            origin != DEMO_ORIGIN
            and re.fullmatch(r"http://127\.0\.0\.1:[0-9]{1,5}", origin) is None
        ):
            raise ValueError("[KALSHI-DEMO-002] account client only accepts demo or loopback")
        self._client, self._signer, self._clock = client, signer, clock
        self._origin = origin

    @staticmethod
    def _document(response: HttpxResponse, path: str, expected: int) -> JSONObject:
        """Require the endpoint status and parse its response without logging secrets.

        Args:
            response: Actual HTTP response.
            path: Endpoint path for the error message.
            expected: Required HTTP status.

        Returns:
            JSON object to be immediately validated by the endpoint caller.

        Raises:
            ValueError: KALSHI-DEMO-003 for a status mismatch or invalid JSON.
        """
        if response.status_code != expected:
            raise ValueError(
                f"[KALSHI-DEMO-003] {path} returned HTTP {response.status_code}; request failed"
            )
        return narrow_json_to_dict(load_json_str(response.text))

    def get(self, path: str, params: Mapping[str, str | int]) -> JSONObject:
        """Send one signed GET with query parameters excluded from its signature.

        Args:
            path: Endpoint relative to the trade API prefix.
            params: Explicit endpoint filters.

        Returns:
            Native JSON object.

        Raises:
            ValueError: Invalid signature input, HTTP status or JSON.
            HTTPError: Transport failure, propagated without retries.
        """
        full_path = _PREFIX + path
        headers = self._signer.headers("GET", full_path, parse_timestamp(self._clock()))
        return self._document(
            self._client.get(self._origin + full_path, headers=headers, params=params), path, 200
        )

    def submit(self, intent: OrderIntent) -> JSONObject:
        """Send one IOC demo order after the orchestration layer has reserved risk.

        Args:
            intent: Validated and durably reserved one-order intent.

        Returns:
            Native acknowledgement; it is not cumulative execution evidence.

        Raises:
            ValueError: Invalid signature input, non-201 status or invalid JSON.
            HTTPError: Ambiguous transport failure; never resend automatically.
        """
        path = "/portfolio/events/orders"
        observed = parse_timestamp(self._clock())
        if not intent["created_ms"] <= observed < intent["expires_ms"]:
            raise ValueError("[DEMO-TRIAL-008] intent expired before HTTP submission")
        headers = self._signer.headers("POST", _PREFIX + path, observed)
        payload = order_payload(intent)
        payload["subaccount"] = 0
        return self._document(
            self._client.post(self._origin + _PREFIX + path, headers=headers, json=payload),
            path,
            201,
        )
