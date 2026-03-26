"""World state synchronization for the tick loop.

Installs a JavaScript WebSocket hook in the browser to capture raw binary
messages, then drains them each tick and feeds them through the protocol
decoder to keep the world state fresh.
"""

from __future__ import annotations

from platform_core.json_utils import load_json_str, narrow_json_to_dict
from platform_core.logging import get_logger

from tankpit_bot._test_hooks import BotProtocol
from tankpit_bot.sniffer.decoders import process_received_message

log = get_logger(__name__)

# JavaScript that hooks EventTarget.prototype.addEventListener to intercept
# all WebSocket message handlers.  Captured binary frames are stored as
# base64 strings in window.__rawMsgs for drain_js_messages() to collect.
_INSTALL_HOOK_JS = r"""(() => {
let info = {
  allWS: (window.__allWS || []).length,
  recv: window.__wsRecvCount || 0,
  wsWrapped: typeof window.__rawMsgs !== 'undefined'
};
if (!window.__runtimeHooked) {
  window.__runtimeHooked = true;
  if (!window.__rawMsgs) window.__rawMsgs = [];
  if (!window.__wsRecvCount) window.__wsRecvCount = 0;
  const origAEL = EventTarget.prototype.addEventListener;
  EventTarget.prototype.addEventListener = function(type, fn, opts) {
    if (this instanceof WebSocket && type === 'message') {
      if (!window.__allWS) window.__allWS = [];
      if (window.__allWS.indexOf(this) === -1) window.__allWS.push(this);
      window.__capturedWS = this;
      const origFn = fn;
      const ws = this;
      fn = function(event) {
        window.__wsRecvCount++;
        if (ws.readyState === 1) window.__capturedWS = ws;
        try {
          if (event.data instanceof Blob) {
            const reader = new FileReader();
            reader.onload = function() {
              const bytes = new Uint8Array(reader.result);
              let b = '';
              for (let i = 0; i < bytes.length; i += 8192) {
                b += String.fromCharCode.apply(
                  null, bytes.subarray(i, i + 8192));
              }
              window.__rawMsgs.push(btoa(b));
              if (window.__rawMsgs.length > 500)
                window.__rawMsgs = window.__rawMsgs.slice(-200);
            };
            reader.readAsArrayBuffer(event.data);
          }
        } catch(e) {}
        return origFn.call(this, event);
      };
      info.hooked_msg_listener = true;
    }
    return origAEL.call(this, type, fn, opts);
  };
  info.runtime_hook = 'installed';
} else {
  info.runtime_hook = 'already_installed';
}
return JSON.stringify(info);
})()"""

# JavaScript that atomically drains the captured message queue.
_DRAIN_JS = r"""
(() => {
    let msgs = window.__rawMsgs || [];
    let count = window.__wsRecvCount || 0;
    window.__rawMsgs = [];
    return JSON.stringify({count: count, msgs: msgs});
})()
"""


def install_ws_hook(bot: BotProtocol) -> None:
    """Install the JavaScript WebSocket message capture hook.

    Patches EventTarget.prototype.addEventListener in the browser so that
    all WebSocket 'message' events are captured as base64 into
    window.__rawMsgs.  Safe to call multiple times — the hook checks
    whether it has already been installed.

    Args:
        bot: Bot instance with a CDP session.
    """
    cdp = bot._cdp
    if cdp is None:
        return

    result = cdp.send("Runtime.evaluate", {"expression": _INSTALL_HOOK_JS, "returnByValue": True})
    inner = result.get("result")
    hook_val = inner.get("value", "") if isinstance(inner, dict) else ""
    log.info("WS hook: %s", hook_val)


def drain_js_messages(bot: BotProtocol) -> int:
    """Drain raw WebSocket messages from JavaScript and decode them.

    Reads the base64-encoded messages accumulated by the JS hook in
    window.__rawMsgs, feeds each one through process_received_message()
    to update world state, then returns the number of messages drained.

    Args:
        bot: Bot instance with a CDP session.

    Returns:
        Number of messages drained and decoded.
    """
    cdp = bot._cdp
    if cdp is None:
        return 0

    result = cdp.send("Runtime.evaluate", {"expression": _DRAIN_JS, "returnByValue": True})
    inner = result.get("result")
    if not isinstance(inner, dict):
        return 0
    val = inner.get("value")
    if not isinstance(val, str) or not val:
        return 0

    data = narrow_json_to_dict(load_json_str(val))
    raw_msgs = data.get("msgs", [])
    if not isinstance(raw_msgs, list):
        return 0

    count = 0
    for b64 in raw_msgs:
        if isinstance(b64, str):
            process_received_message(b64)
            count += 1

    if count > 0:
        total_recv = data.get("count", 0)
        log.info("JS->Python: %d raw msgs (total recv: %s)", count, total_recv)

    return count


__all__ = [
    "drain_js_messages",
    "install_ws_hook",
]
