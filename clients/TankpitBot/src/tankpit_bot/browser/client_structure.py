"""One-shot structural survey of the live ``window.__tankpitActiveGame`` tree.

The per-tick snapshot captures primitive fields and one level of
collections, but live run 20260610-011x showed the client's entity
lists are NOT direct children of ``activeGame.h`` (its object-valued
children are render internals: canvas layers, keymaps, pixel buffers).
Rather than guessing one nesting level at a time, this module walks the
whole client object to a bounded depth ONCE per session and emits its
shape -- per node: type, key names, array length, and the first item's
keys for collections. The artifact then shows exactly where (and
whether) the client keeps semantic tank/container lists.

The survey is emitted at the first healthy tick and never again for the
process lifetime; tests reset the gate explicitly.
"""

from __future__ import annotations

from platform_core.json_utils import JSONObject, dump_json_str, require_dict

from tankpit_bot._test_hooks import CDPSessionProtocol
from tankpit_bot.runtime_logging import emit_diagnostic

_SURVEY_EXPRESSION = """
(() => {
    const MAX_DEPTH = 4;
    const MAX_KEYS = 60;
    function shape(value, depth) {
        if (value === null) return {kind: 'null'};
        const t = typeof value;
        if (t !== 'object') return {kind: t};
        if (Array.isArray(value)) {
            const out = {kind: 'array', length: value.length};
            if (value.length > 0 && depth < MAX_DEPTH) {
                out.item = shape(value[0], depth + 1);
            }
            return out;
        }
        const keys = Object.keys(value);
        const out = {kind: 'object', key_count: keys.length, keys: keys.slice(0, MAX_KEYS)};
        if (depth < MAX_DEPTH) {
            const children = {};
            for (const key of keys.slice(0, MAX_KEYS)) {
                const child = value[key];
                if (child !== null && typeof child === 'object') {
                    children[key] = shape(child, depth + 1);
                }
            }
            out.children = children;
        }
        return out;
    }
    const activeGame =
        window.__tankpitActiveGame && typeof window.__tankpitActiveGame === 'object'
            ? window.__tankpitActiveGame
            : null;
    if (activeGame === null) return null;
    return shape(activeGame, 0);
})()
"""


class ClientStructureSurveyor:
    """Emits the client structure survey once per session.

    "Once" is one SESSION's once, so the gate is instance state: two
    sessions in one process each owe their own survey, and a shared
    gate would silently withhold the second one
    ([[session-state-deglobalisation]] step 5). A fresh instance has
    not yet surveyed, which is why no reset function exists.
    """

    def __init__(self) -> None:
        self._emitted = False

    def maybe_emit(self, cdp: CDPSessionProtocol) -> bool:
        """Capture and emit the client structure survey once per session.

        Args:
            cdp: Active CDP session attached to the live tankpit page.

        Returns:
            True when the survey was captured and emitted; False when it
            was already emitted this session or the client object is not
            yet present on the page.

        Raises:
            JSONTypeError: When the evaluated survey payload is not a
                JSON object; malformed captures are surfaced instead of
                dropped.
        """
        if self._emitted:
            return False
        result = cdp.send(
            "Runtime.evaluate",
            {"expression": _SURVEY_EXPRESSION, "returnByValue": True},
        )
        result_obj = require_dict(result, "result")
        raw_value = result_obj.get("value")
        if raw_value is None:
            return False
        survey: JSONObject = require_dict({"survey": raw_value}, "survey")
        self._emitted = True
        emit_diagnostic(
            diagnostic_kind="client_structure_survey",
            survey_json=dump_json_str(survey),
        )
        return True


__all__ = [
    "ClientStructureSurveyor",
]
