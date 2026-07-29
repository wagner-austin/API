"""Self-contained phone watch page served by the bot service.

One HTML string, zero external assets, served at ``GET /watch``. The
page is the fiesta-free replacement for the vibeshine tankpit stream
(2026-07-28): live game video via the ``/video`` MJPEG relay, session
state via the ``/status`` SSE stream, and the same start/stop/mode
controls the SPA bot panel offers — all against the service's own
routes.

Every URL in the page is RELATIVE (no leading slash) so the page works
identically on both of its origins:

* direct: ``http://<host>:27100/watch`` → ``video`` resolves to
  ``/video``
* proxied: ``https://tankpit.austinwagner.org/api/tankbot/watch`` →
  ``video`` resolves to ``/api/tankbot/video`` (nginx strips the
  prefix back off)
"""

from __future__ import annotations

WATCH_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no">
<title>TankpitBot</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #101418; color: #d7e0ea;
    font-family: ui-monospace, Menlo, Consolas, monospace;
    min-height: 100vh; display: flex; flex-direction: column;
    align-items: center; gap: 12px; padding: 12px;
  }
  h1 { font-size: 1.1rem; letter-spacing: 0.15em; color: #8fd18f; }
  #view {
    width: 100%; max-width: 720px; aspect-ratio: 1 / 1;
    background: #000; border: 1px solid #2a3540; border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    overflow: hidden;
  }
  #video { width: 100%; height: 100%; object-fit: contain; display: none; }
  #placeholder { color: #5a6a7a; font-size: 0.9rem; }
  #stats {
    display: flex; gap: 16px; flex-wrap: wrap; justify-content: center;
    font-size: 0.85rem; color: #9fb2c4;
  }
  #stats b { color: #d7e0ea; }
  #buttons { display: flex; gap: 8px; flex-wrap: wrap; justify-content: center; }
  button {
    font: inherit; padding: 10px 18px; border-radius: 6px;
    border: 1px solid #2a3540; background: #1a232c; color: #d7e0ea;
    cursor: pointer;
  }
  button:active { background: #2a3540; }
  #btn-start { border-color: #3c8c50; color: #8fd18f; }
  #btn-stop  { border-color: #8c3c50; color: #d18f9f; }
  #banner { font-size: 0.8rem; color: #d1b48f; min-height: 1em; }
</style>
</head>
<body>
<h1>TANKPITBOT</h1>
<div id="view">
  <img id="video" alt="live game view">
  <div id="placeholder">no session</div>
</div>
<div id="stats">
  <span>mode <b id="s-mode">-</b></span>
  <span>kills <b id="s-kills">0</b></span>
  <span>hits <b id="s-hits">0</b></span>
  <span>misses <b id="s-misses">0</b></span>
  <span>radars <b id="s-radars">0</b></span>
  <span>teleports <b id="s-teleports">0</b></span>
</div>
<div id="buttons">
  <button id="btn-start">START</button>
  <button id="btn-stop">STOP</button>
  <button data-mode="HUNT">HUNT</button>
  <button data-mode="COLLECT">GATHER</button>
  <button data-mode="AUTO">AUTO</button>
  <button data-mode="UNSET">IDLE</button>
</div>
<div id="banner"></div>
<script>
"use strict";
const banner = document.getElementById("banner");
const video = document.getElementById("video");
const placeholder = document.getElementById("placeholder");

function post(path, body) {
  const options = body === undefined
    ? { method: "POST" }
    : { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body) };
  fetch(path, options).then((r) => {
    banner.textContent = r.ok ? "" : path + " -> " + r.status;
  }).catch((e) => { banner.textContent = String(e); });
}

document.getElementById("btn-start").addEventListener("click", () => post("start"));
document.getElementById("btn-stop").addEventListener("click", () => post("stop"));
for (const b of document.querySelectorAll("[data-mode]")) {
  b.addEventListener("click", () => post("mode", { manual_mode: b.dataset.mode }));
}

let streaming = false;
function setStreaming(on) {
  if (on === streaming) { return; }
  streaming = on;
  if (on) {
    video.src = "video?t=" + Date.now();
    video.style.display = "block";
    placeholder.style.display = "none";
  } else {
    video.removeAttribute("src");
    video.style.display = "none";
    placeholder.style.display = "block";
  }
}

function connectStatus() {
  const source = new EventSource("status");
  source.onmessage = (event) => {
    const s = JSON.parse(event.data);
    document.getElementById("s-mode").textContent =
      s.running ? (s.active_mode || "-") : "idle";
    document.getElementById("s-kills").textContent = s.stats.kills;
    document.getElementById("s-hits").textContent = s.stats.hits;
    document.getElementById("s-misses").textContent = s.stats.misses;
    document.getElementById("s-radars").textContent = s.stats.radars_used;
    document.getElementById("s-teleports").textContent = s.stats.teleports;
    setStreaming(s.running);
  };
  source.onerror = () => {
    source.close();
    setStreaming(false);
    banner.textContent = "status stream lost - reconnecting";
    setTimeout(connectStatus, 3000);
  };
}
connectStatus();
</script>
</body>
</html>
"""

__all__ = [
    "WATCH_PAGE_HTML",
]
