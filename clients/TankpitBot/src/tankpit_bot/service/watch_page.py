"""Self-contained phone watch page served by the bot service.

One HTML string, no assets from outside the service, served at
``GET /watch``. The page is the fiesta-free replacement for the
vibeshine tankpit stream (2026-07-28): live game video from the
session's HLS files under ``/video/``, session state via the
``/status`` SSE stream, and the same start/stop/mode controls the SPA
bot panel offers — all against the service's own routes. The one
script it loads, ``watch/hls.js``, is the vendored hls.js build the
wheel itself carries ([[packaged-data-assets]]).

Playback takes BOTH standard HLS paths, and needs both to cover real
devices rather than as a courtesy: iOS Safari plays HLS natively and
(before 17.1) has no Media Source Extensions for hls.js to use, while
Chrome — desktop and Android — has MSE and no native HLS at all.
Feature detection picks the one path each device actually has.

Every URL in the page is RELATIVE (no leading slash) so the page works
identically on both of its origins:

* direct: ``http://<host>:27100/watch`` → ``video/index.m3u8``
  resolves to ``/video/index.m3u8``
* proxied: ``https://tankpit.austinwagner.org/api/tankbot/watch`` →
  it resolves to ``/api/tankbot/video/index.m3u8`` (nginx strips the
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
  #video { width: 100%; height: 100%; object-fit: contain; display: none;
           background: #000; }
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
  <video id="video" autoplay muted playsinline></video>
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
<script src="watch/hls.js"></script>
<script>
"use strict";
const banner = document.getElementById("banner");
const video = document.getElementById("video");
const placeholder = document.getElementById("placeholder");
const HLS_URL = "video/index.m3u8";

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
let hls = null;
function setStreaming(on) {
  if (on === streaming) { return; }
  streaming = on;
  if (on) {
    // `autoplay muted playsinline` together are what let this start
    // without a gesture: an unmuted autoplay is blocked outright, and
    // on iOS a video without `playsinline` is hoisted into the
    // fullscreen player.
    if (video.canPlayType("application/vnd.apple.mpegurl")) {
      // Native HLS (iOS/macOS Safari). The cache buster is not
      // optional: a browser remembers a finished or failed media
      // URL, so re-assigning the same src is a no-op and the
      // element would sit black forever.
      video.src = HLS_URL + "?t=" + Date.now();
    } else {
      // MSE path (Chrome desktop/Android) via the vendored hls.js.
      // A fatal error (encoder still warming, service restarted)
      // tears the instance down; the next status tick re-arms.
      hls = new Hls();
      hls.on(Hls.Events.ERROR, (event, data) => {
        if (data.fatal) {
          banner.textContent = "video: " + data.type + " - retrying";
          setStreaming(false);
        }
      });
      hls.loadSource(HLS_URL);
      hls.attachMedia(video);
    }
    video.style.display = "block";
    placeholder.style.display = "none";
  } else {
    if (hls !== null) { hls.destroy(); hls = null; }
    // Pause before clearing. Removing src alone leaves the element in
    // a playing state pointed at nothing, which keeps the decoder
    // alive and, on some builds, the request open.
    video.pause();
    video.removeAttribute("src");
    video.load();
    video.style.display = "none";
    placeholder.style.display = "block";
  }
}

// A live stream has no end, so `ended` means the source dropped --
// the service restarted, the encoder stopped, or the playlist went
// away. Re-arming on the next status tick is what makes that
// recoverable without a manual page reload, which is the failure the
// old MJPEG viewer had.
video.addEventListener("ended", () => { streaming = false; });
video.addEventListener("error", () => { streaming = false; });

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
