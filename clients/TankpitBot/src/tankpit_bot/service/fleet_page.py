"""The fleet manager's control page — one HTML file, no dependencies.

Served at ``GET /`` on the fleet port (default 27300). Everything the
page does rides the same JSON endpoints the operating AI uses
(``/bots``, ``/bots/{instance}/stats``, stop/restart/remove) — the
page is a convenience skin over the API, never a second control path.
Completely separate from the fiesta SPA stack: no nginx, no SSE, no
streaming, no external assets.
"""

from __future__ import annotations

FLEET_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>tankpit fleet</title>
<style>
  body { background:#111; color:#ddd; font-family:Consolas,monospace;
         margin:1.5rem; }
  h1 { font-size:1.2rem; color:#8c8; }
  table { border-collapse:collapse; width:100%; margin-top:1rem; }
  th, td { border:1px solid #333; padding:.4rem .6rem; text-align:left;
           font-size:.9rem; }
  th { background:#1a1a1a; color:#aaa; }
  tr.dead td { color:#777; }
  .alive { color:#6d6; } .exited { color:#d66; }
  .clean { color:#6d6; } .crash { color:#d66; }
  button { background:#222; color:#ddd; border:1px solid #555;
           padding:.25rem .7rem; margin-right:.3rem; cursor:pointer;
           font-family:inherit; }
  button:hover { background:#333; }
  button:disabled { color:#555; border-color:#333; cursor:default; }
  form { margin-top:1.2rem; display:flex; gap:.5rem; flex-wrap:wrap;
         align-items:center; }
  input { background:#1a1a1a; color:#ddd; border:1px solid #444;
          padding:.3rem .5rem; font-family:inherit; width:7rem; }
  #error { color:#d66; margin-top:.8rem; min-height:1.2rem; }
  .muted { color:#666; }
</style>
</head>
<body>
<h1>tankpit fleet</h1>
<table>
  <thead><tr>
    <th>instance</th><th>account</th><th>pid</th><th>state</th>
    <th>bounds</th><th>kills</th><th>deaths</th><th>rank</th>
    <th>duration</th><th>exit</th><th>actions</th>
  </tr></thead>
  <tbody id="rows"><tr><td colspan="11" class="muted">loading…</td></tr></tbody>
</table>
<form id="spawn">
  <input id="instance" placeholder="instance" required>
  <input id="account" placeholder="account (opt)">
  <input id="kills" placeholder="kills (0=∞)" type="number" min="0" value="0">
  <input id="seconds" placeholder="seconds (0=∞)" type="number" min="0" value="0">
  <button type="submit">spawn</button>
</form>
<div id="error"></div>
<script>
"use strict";
const statsCache = {};

function fmtDuration(s) {
  if (s === null || s === undefined) return "";
  const m = Math.floor(s / 60);
  return m + "m" + String(s % 60).padStart(2, "0") + "s";
}

async function act(method, path) {
  const response = await fetch(path, {method});
  if (!response.ok) {
    document.getElementById("error").textContent =
      method + " " + path + " -> " + response.status + " " + await response.text();
    return false;
  }
  document.getElementById("error").textContent = "";
  return true;
}

async function refreshStats(instance) {
  try {
    const response = await fetch("/bots/" + instance + "/stats");
    if (response.ok) statsCache[instance] = await response.json();
  } catch (e) { /* stats are best-effort; the row still renders */ }
}

function row(bot) {
  const s = statsCache[bot.instance] || {};
  const tr = document.createElement("tr");
  if (!bot.alive) tr.className = "dead";
  const state = bot.alive
    ? '<span class="alive">alive</span>'
    : '<span class="exited">exit ' + bot.returncode + "</span>";
  const bounds = (bot.kills || "∞") + "k / " + (bot.seconds || "∞") + "s";
  const exit_ = s.available
    ? (s.clean_exit ? '<span class="clean">' + (s.exit_reason || "clean") + "</span>"
                    : (bot.alive ? "" : '<span class="crash">no scorecard</span>'))
    : "";
  const rank = s.available && s.rank_number >= 0 ? s.rank_number : "";
  tr.innerHTML =
    "<td>" + bot.instance + "</td><td>" + (bot.account || "default") + "</td>" +
    "<td>" + bot.pid + "</td><td>" + state + "</td><td>" + bounds + "</td>" +
    "<td>" + (s.available ? s.kills : "") + "</td>" +
    "<td>" + (s.available ? s.deaths : "") + "</td>" +
    "<td>" + rank + "</td>" +
    "<td>" + (s.available ? fmtDuration(s.duration_s) : "") + "</td>" +
    "<td>" + exit_ + "</td>";
  const actions = document.createElement("td");
  const stop = document.createElement("button");
  stop.textContent = "stop";
  stop.disabled = !bot.alive;
  stop.onclick = () => act("POST", "/bots/" + bot.instance + "/stop");
  const restart = document.createElement("button");
  restart.textContent = "restart";
  restart.disabled = bot.alive;
  restart.onclick = () => act("POST", "/bots/" + bot.instance + "/restart");
  const remove = document.createElement("button");
  remove.textContent = "remove";
  remove.disabled = bot.alive;
  remove.onclick = () => act("DELETE", "/bots/" + bot.instance);
  actions.append(stop, restart, remove);
  tr.appendChild(actions);
  return tr;
}

async function refresh() {
  const response = await fetch("/bots");
  const body = await response.json();
  const tbody = document.getElementById("rows");
  tbody.replaceChildren();
  if (!body.bots.length) {
    tbody.innerHTML = '<tr><td colspan="11" class="muted">no bots — spawn one below</td></tr>';
    return;
  }
  for (const bot of body.bots) tbody.appendChild(row(bot));
  for (const bot of body.bots) refreshStats(bot.instance);
}

document.getElementById("spawn").addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = {
    instance: document.getElementById("instance").value.trim(),
    account: document.getElementById("account").value.trim(),
    kills: Number(document.getElementById("kills").value) || 0,
    seconds: Number(document.getElementById("seconds").value) || 0,
  };
  const response = await fetch("/bots", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    document.getElementById("error").textContent =
      "spawn -> " + response.status + " " + await response.text();
    return;
  }
  document.getElementById("error").textContent = "";
  document.getElementById("instance").value = "";
  refresh();
});

refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>
"""

__all__ = ["FLEET_PAGE_HTML"]
