"""JavaScript hook source injected into every tankpit page on load.

The script does three things at page-load time, before any tankpit
client code runs:

1. Captures the live game-client object onto
   ``window.__tankpitActiveGame`` using a ``defineProperty`` trap on
   ``Object.prototype``. The first object that assigns all six minified
   properties (``map``, ``h``, ``i``, ``va``, ``Ha``, ``s``) is
   recognized as the game client.
2. Hooks ``EventTarget.prototype.addEventListener`` so that every
   ``WebSocket`` instance the page creates has its incoming-message
   handler wrapped. Incoming binary frames are base64-encoded and
   pushed onto ``window.__rawMsgs`` (capped at the most recent 500
   entries, trimmed to 200 on overflow).
3. Hooks ``WebSocket.prototype.send`` so the bot's
   :func:`tankpit_bot.browser.session.send_websocket_bytes` injections
   can be distinguished from the page client's own sends. The current
   send is labelled via ``window.__codexCurrentSendLabel``; a stack
   trace, label, and origin (``"bot_injected"`` vs ``"page_client"``)
   are pushed onto ``window.__sentFrameMetaQueue`` (capped at 500,
   trimmed to 200 on overflow).

The script is exported as :data:`BROWSER_HOOK_SOURCE`. Consumers inject
it via Playwright's ``page.add_init_script`` so it runs before the
tankpit bundle.
"""

from __future__ import annotations

BROWSER_HOOK_SOURCE = """
            (function() {
                window.__capturedWS = null;
                window.__allWS = [];
                window.__rawMsgs = [];
                window.__wsRecvCount = 0;
                window.__codexCurrentSendLabel = null;
                window.__sentFrameMetaQueue = [];
                window.__lastPageClientSendPerfMs = null;
                window.__lastBotInjectedSendPerfMs = null;
                window.__tankpitActiveGame = null;

                function maybeCaptureGameClient(candidate) {
                    if (!candidate || typeof candidate !== 'object') {
                        return;
                    }
                    const mapObject =
                        candidate.map && typeof candidate.map === 'object'
                            ? candidate.map
                            : null;
                    const worldObject =
                        candidate.h && typeof candidate.h === 'object'
                            ? candidate.h
                            : null;
                    const selfTank =
                        candidate.i && typeof candidate.i === 'object'
                            ? candidate.i
                            : null;
                    const transport =
                        candidate.va && typeof candidate.va === 'object'
                            ? candidate.va
                            : null;
                    const actionQueue =
                        worldObject &&
                        worldObject.j &&
                        typeof worldObject.j === 'object' &&
                        Array.isArray(worldObject.j.actions)
                            ? worldObject.j.actions
                            : null;
                    if (
                        mapObject !== null &&
                        worldObject !== null &&
                        selfTank !== null &&
                        transport !== null &&
                        actionQueue !== null &&
                        typeof candidate.s === 'number' &&
                        typeof candidate.Ha === 'boolean'
                    ) {
                        window.__tankpitActiveGame = candidate;
                    }
                }

                function installClientProbe(propertyName) {
                    const storageName = '__codexProbeValue_' + propertyName;
                    Object.defineProperty(Object.prototype, propertyName, {
                        configurable: true,
                        enumerable: false,
                        get: function() {
                            if (Object.prototype.hasOwnProperty.call(this, storageName)) {
                                return this[storageName];
                            }
                            return undefined;
                        },
                        set: function(value) {
                            Object.defineProperty(this, storageName, {
                                value: value,
                                writable: true,
                                configurable: true,
                                enumerable: false
                            });
                            Object.defineProperty(this, propertyName, {
                                value: value,
                                writable: true,
                                configurable: true,
                                enumerable: true
                            });
                            maybeCaptureGameClient(this);
                        }
                    });
                }

                installClientProbe('map');
                installClientProbe('h');
                installClientProbe('i');
                installClientProbe('va');
                installClientProbe('Ha');
                installClientProbe('s');

                // Hook EventTarget.prototype.addEventListener globally.
                // This catches ALL addEventListener calls, including those
                // made by the game on WebSocket instances.
                const origAEL = EventTarget.prototype.addEventListener;
                EventTarget.prototype.addEventListener = function(type, fn, opts) {
                    if (this instanceof WebSocket && type === 'message') {
                        if (window.__allWS.indexOf(this) === -1) {
                            window.__allWS.push(this);
                        }
                        const ws = this;
                        const origFn = fn;
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
                                            b += String.fromCharCode.apply(null,
                                                bytes.subarray(i, i + 8192));
                                        }
                                        window.__rawMsgs.push(btoa(b));
                                        if (window.__rawMsgs.length > 500) {
                                            window.__rawMsgs = window.__rawMsgs.slice(-200);
                                        }
                                    };
                                    reader.readAsArrayBuffer(event.data);
                                }
                            } catch(e) {}
                            return origFn.call(this, event);
                        };
                    }
                    return origAEL.call(this, type, fn, opts);
                };

                // Hook send for command injection
                const origSend = WebSocket.prototype.send;
                WebSocket.prototype.send = function(data) {
                    if (!window.__capturedWS || window.__capturedWS.readyState !== 1) {
                        if (this.readyState === 1) window.__capturedWS = this;
                    }
                    if (window.__allWS.indexOf(this) === -1) {
                        window.__allWS.push(this);
                    }
                    const currentLabel =
                        typeof window.__codexCurrentSendLabel === 'string'
                            ? window.__codexCurrentSendLabel
                            : null;
                    const perfNow = performance.now();
                    const err = new Error();
                    const stack = typeof err.stack === 'string' ? err.stack : '';
                    if (currentLabel) {
                        window.__lastBotInjectedSendPerfMs = perfNow;
                    } else {
                        window.__lastPageClientSendPerfMs = perfNow;
                    }
                    window.__sentFrameMetaQueue.push({
                        origin: currentLabel ? 'bot_injected' : 'page_client',
                        label: currentLabel || '',
                        stack: stack
                    });
                    if (window.__sentFrameMetaQueue.length > 500) {
                        window.__sentFrameMetaQueue = window.__sentFrameMetaQueue.slice(-200);
                    }
                    return origSend.call(this, data);
                };
            })();
            """


__all__ = [
    "BROWSER_HOOK_SOURCE",
]
