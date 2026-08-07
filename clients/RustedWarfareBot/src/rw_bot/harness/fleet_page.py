"""The match fleet's control page — one HTML file, no dependencies.

Served at ``GET /`` on the fleet port (default 27500). Everything the
page does rides the same JSON endpoints the operating AI uses — the
page is a convenience skin over the API, never a second control path.
No external assets, no framework, nothing outside loopback.
"""

from __future__ import annotations

FLEET_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>rusted warfare fleet</title>
<style>
  body { background:#141210; color:#ddd; font-family:Consolas,monospace;
         margin:1.5rem; }
  h1 { font-size:1.2rem; color:#d95; }
  table { border-collapse:collapse; width:100%; margin-top:1rem; }
  th, td { border:1px solid #3a332c; padding:.4rem .6rem; text-align:left;
           font-size:.9rem; }
  th { background:#1c1915; color:#aaa; }
  tr.dead td { color:#777; }
  .alive { color:#6d6; } .exited { color:#d66; }
  .verdict { color:#dc8; }
  button { background:#242019; color:#ddd; border:1px solid #5a5245;
           padding:.25rem .7rem; margin-right:.3rem; cursor:pointer;
           font-family:inherit; }
  button:hover { background:#332d24; }
  button:disabled { color:#555; border-color:#333; cursor:default; }
  form { margin-top:1.2rem; display:flex; gap:.5rem; flex-wrap:wrap;
         align-items:center; }
  input { background:#1c1915; color:#ddd; border:1px solid #494236;
          padding:.3rem .5rem; font-family:inherit; width:7rem; }
  input.wide { width:16rem; }
  #error { color:#d66; margin-top:.8rem; min-height:1.2rem; }
  .muted { color:#666; }
  pre { background:#1c1915; border:1px solid #3a332c; padding:.5rem;
        font-size:.8rem; overflow-x:auto; }
</style>
</head>
<body>
<h1>rusted warfare fleet</h1>
<table>
  <thead><tr>
    <th>instance</th><th>seed</th><th>map</th><th>opp</th><th>diff</th>
    <th>ff</th><th>pid</th><th>state</th><th>verdict</th><th>actions</th>
  </tr></thead>
  <tbody id="rows"><tr><td colspan="10" class="muted">loading…</td></tr></tbody>
</table>
<form id="spawn">
  <input id="instance" placeholder="instance" required>
  <input id="seed" placeholder="seed" type="number" min="0" value="0">
  <input id="map" class="wide" placeholder="map (blank = default)">
  <input id="opponents" placeholder="opponents" type="number" min="0" value="1">
  <input id="difficulty" placeholder="difficulty" type="number" min="0" value="0">
  <input id="fastforward" placeholder="fast-forward x" type="number" min="0" value="0">
  <input id="tree" class="wide" placeholder="frozen tree (blank = worktree)">
  <button type="submit">spawn match</button>
</form>
<div id="error"></div>
<div id="report"></div>
<script>
"use strict";
const statsCache = {};

async function act(method, path) {
  const response = await fetch(path, {method});
  if (!response.ok) {
    document.getElementById("error").textContent =
      method + " " + path + " -> " + response.status + " " + await response.text();
    return;
  }
  document.getElementById("error").textContent = "";
}

async function refreshStats(instance) {
  try {
    const response = await fetch("/bots/" + instance + "/stats");
    if (response.ok) statsCache[instance] = await response.json();
  } catch (e) { /* stats are best-effort; the row still renders */ }
}

function showReport(instance) {
  const s = statsCache[instance];
  const target = document.getElementById("report");
  if (!s || !s.report.length) {
    target.textContent = "";
    return;
  }
  const pre = document.createElement("pre");
  pre.textContent = "== " + instance + " ==\\n" + s.report.join("\\n");
  target.replaceChildren(pre);
}

function row(bot) {
  const s = statsCache[bot.instance] || {};
  const tr = document.createElement("tr");
  if (!bot.alive) tr.className = "dead";
  const state = bot.alive
    ? '<span class="alive">running</span>'
    : '<span class="exited">exit ' + bot.returncode + "</span>";
  const verdict = s.finished
    ? '<span class="verdict">' + s.verdict.replace(/^verdict\\s*/, "") + "</span>"
    : (s.available ? "in progress" : "");
  tr.innerHTML =
    "<td>" + bot.instance + "</td><td>" + bot.seed + "</td>" +
    "<td>" + (bot.map || "default") + "</td><td>" + bot.opponents + "</td>" +
    "<td>" + bot.difficulty + "</td><td>" + (bot.fastforward || "1") + "x</td>" +
    "<td>" + bot.pid + "</td><td>" + state + "</td><td>" + verdict + "</td>";
  const actions = document.createElement("td");
  const stop = document.createElement("button");
  stop.textContent = "kill";
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
  const report = document.createElement("button");
  report.textContent = "report";
  report.onclick = () => showReport(bot.instance);
  actions.append(stop, restart, remove, report);
  tr.appendChild(actions);
  return tr;
}

async function refresh() {
  const response = await fetch("/bots");
  const body = await response.json();
  const tbody = document.getElementById("rows");
  tbody.replaceChildren();
  if (!body.bots.length) {
    tbody.innerHTML =
      '<tr><td colspan="10" class="muted">no matches — spawn one below</td></tr>';
    return;
  }
  for (const bot of body.bots) tbody.appendChild(row(bot));
  for (const bot of body.bots) refreshStats(bot.instance);
}

document.getElementById("spawn").addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = {
    instance: document.getElementById("instance").value.trim(),
    seed: Number(document.getElementById("seed").value) || 0,
    map: document.getElementById("map").value.trim(),
    opponents: Number(document.getElementById("opponents").value) || 0,
    difficulty: Number(document.getElementById("difficulty").value) || 0,
    fastforward: Number(document.getElementById("fastforward").value) || 0,
    tree: document.getElementById("tree").value.trim(),
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
