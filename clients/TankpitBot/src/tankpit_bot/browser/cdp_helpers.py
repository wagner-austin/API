"""CDP evaluation and protocol key discovery helpers."""

from __future__ import annotations

import base64
import re

from platform_core.json_utils import JSONObject, require_dict, require_str
from platform_core.logging import get_logger

from tankpit_bot._test_hooks import CDPSessionProtocol
from tankpit_bot.protocol.framing import decode_frame

log = get_logger(__name__)

_STATIC_KEY_PATTERN = re.compile(r'"([^"]{1000})"')


def decode_captured_body(payload: str) -> bytes:
    """Decode one captured raw WebSocket payload to its body bytes.

    Args:
        payload: Base64-encoded framed message payload.

    Returns:
        Decoded frame body bytes.
    """
    framed = base64.b64decode(payload)
    body, remaining = decode_frame(framed)
    if remaining:
        raise ValueError(f"unexpected trailing bytes in framed message: {remaining.hex()}")
    return body


def evaluate_string(
    cdp: CDPSessionProtocol,
    expression: str,
    *,
    await_promise: bool = False,
) -> str:
    """Evaluate JavaScript and return the string result.

    Args:
        cdp: Active CDP session.
        expression: JavaScript expression to evaluate.
        await_promise: Whether Runtime.evaluate should await a returned Promise.

    Returns:
        String value returned by the expression.
    """
    params: JSONObject = {
        "expression": expression,
        "returnByValue": True,
    }
    if await_promise:
        params["awaitPromise"] = True
    result = cdp.send("Runtime.evaluate", params)
    result_obj = require_dict(result, "result")
    return require_str(result_obj, "value")


def get_magic_key(cdp: CDPSessionProtocol) -> str:
    """Return the current session magic key from the page runtime.

    Args:
        cdp: Active CDP session.

    Returns:
        The current ``tankpit.magic`` value, or an empty string when absent.
    """
    return evaluate_string(
        cdp,
        """
        (() => {
            if (typeof tankpit !== 'undefined' && typeof tankpit.magic === 'string') {
                return tankpit.magic;
            }
            return '';
        })()
        """,
    )


def get_tpclient_url(cdp: CDPSessionProtocol) -> str:
    """Return the loaded tpclient script URL.

    Args:
        cdp: Active CDP session.

    Returns:
        Loaded tpclient script URL, or an empty string when not found.
    """
    return evaluate_string(
        cdp,
        """
        (() => {
            const script = Array.from(document.querySelectorAll('script[src]')).find(
                (item) => item.src.includes('tpclient')
            );
            return script ? script.src : '';
        })()
        """,
    )


def load_tpclient_static_key(cdp: CDPSessionProtocol, tpclient_url: str) -> str:
    """Fetch the loaded tpclient source and extract the current static key.

    Args:
        cdp: Active CDP session.
        tpclient_url: Loaded tpclient script URL.

    Returns:
        Current 1000-character static key string.

    Raises:
        ValueError: If the loaded script does not contain the expected key.
    """
    js_content = evaluate_string(
        cdp,
        f"fetch({tpclient_url!r}).then((response) => response.text())",
        await_promise=True,
    )
    match = _STATIC_KEY_PATTERN.search(js_content)
    if match is None:
        raise ValueError("tpclient static key was not found in loaded script")
    return match.group(1)


__all__ = [
    "decode_captured_body",
    "evaluate_string",
    "get_magic_key",
    "get_tpclient_url",
    "load_tpclient_static_key",
]
