"""The fleet manager's control page — one HTML file, no dependencies.

Served at ``GET /`` on the fleet port (default 27300). Everything the
page does rides the same JSON endpoints the operating AI uses
(``/bots``, ``/accounts``, ``/bots/{instance}/stats``,
stop/restart/remove) — the page is a convenience skin over the API,
never a second control path. Completely separate from the fiesta SPA
stack: no nginx, no SSE, no streaming, no external assets.

Accounts are CONFIG: the launch form offers a dropdown of the
usernames in ``accounts.json`` (plus the default), never free text —
matching the manager, which refuses selectors outside the file.
"""

from __future__ import annotations

FLEET_PAGE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>tankpit fleet</title>
<style>
  :root {
    --bg:#0f1115; --card:#161a21; --line:#262c37; --ink:#d6dae2;
    --dim:#8a93a3; --good:#5ecb71; --bad:#e0656a; --accent:#67b0e8;
    --gold:#d9b45b;
  }
  * { box-sizing:border-box; }
  body { background:var(--bg); color:var(--ink); margin:0;
         font-family:"Segoe UI",system-ui,sans-serif; }
  header { padding:1.1rem 1.6rem; border-bottom:1px solid var(--line);
           display:flex; align-items:baseline; gap:1rem; }
  header h1 { margin:0; font-size:1.15rem; letter-spacing:.03em; }
  header .sub { color:var(--dim); font-size:.85rem; }
  main { padding:1.4rem 1.6rem; display:grid; gap:1.4rem;
         max-width:1100px; }
  .card { background:var(--card); border:1px solid var(--line);
          border-radius:10px; padding:1.1rem 1.3rem; }
  .card h2 { margin:0 0 .8rem; font-size:.95rem; color:var(--accent);
             letter-spacing:.04em; text-transform:uppercase; }
  table { border-collapse:collapse; width:100%; font-size:.9rem; }
  th, td { padding:.5rem .7rem; text-align:left;
           border-bottom:1px solid var(--line); }
  th { color:var(--dim); font-weight:600; font-size:.78rem;
       text-transform:uppercase; letter-spacing:.05em; }
  tr:last-child td { border-bottom:none; }
  tr.dead td { color:var(--dim); }
  .pill { display:inline-block; padding:.1rem .55rem; border-radius:99px;
          font-size:.78rem; font-weight:600; }
  .pill.alive { background:rgba(94,203,113,.14); color:var(--good); }
  .pill.done  { background:rgba(138,147,163,.16); color:var(--dim); }
  .pill.crash { background:rgba(224,101,106,.14); color:var(--bad); }
  .rank { color:var(--gold); font-weight:600; }
  button { background:#1d232d; color:var(--ink); border:1px solid #39424f;
           padding:.32rem .8rem; border-radius:6px; margin-right:.35rem;
           cursor:pointer; font-size:.82rem; }
  button:hover { background:#242c38; }
  button:disabled { color:#555e6b; border-color:#2a313c; cursor:default;
                    background:transparent; }
  button.primary { background:var(--accent); color:#0d141b;
                   border-color:var(--accent); font-weight:600;
                   padding:.45rem 1.2rem; }
  button.primary:hover { filter:brightness(1.1); }
  form { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr));
         gap:.9rem 1.1rem; align-items:end; }
  label { display:block; font-size:.78rem; color:var(--dim);
          margin-bottom:.3rem; text-transform:uppercase;
          letter-spacing:.05em; }
  .hint { font-size:.75rem; color:var(--dim); margin-top:.3rem; }
  input, select { width:100%; background:#0f1319; color:var(--ink);
                  border:1px solid #39424f; border-radius:6px;
                  padding:.45rem .6rem; font-size:.88rem; }
  #error { color:var(--bad); min-height:1.1rem; font-size:.85rem;
           margin-top:.6rem; }
  .empty { color:var(--dim); padding:1rem 0; }
  footer { color:var(--dim); font-size:.75rem; padding:0 1.6rem 1.4rem;
           max-width:1100px; }
  footer code { color:var(--ink); }
</style>
</head>
<body>
<header>
  <h1>tankpit fleet</h1>
  <span class="sub">spawn and manage bots — refreshes every 3&nbsp;s</span>
</header>
<main>
<section class="card">
  <h2>Bots</h2>
  <table>
    <thead><tr>
      <th>name</th><th>account</th><th>status</th><th>limits</th>
      <th>kills</th><th>deaths</th><th>rank</th><th>time</th><th></th>
    </tr></thead>
    <tbody id="rows"><tr><td colspan="9" class="empty">loading…</td></tr></tbody>
  </table>
</section>
<section class="card">
  <h2>Launch a bot</h2>
  <form id="spawn">
    <div>
      <label for="instance">Name</label>
      <input id="instance" placeholder="e.g. alpha" required
             pattern="[a-z0-9][a-z0-9_-]{0,31}">
      <div class="hint">lowercase letters/numbers — names its logs under
        runs/bot/&lt;name&gt;/</div>
    </div>
    <div>
      <label for="account">Account</label>
      <select id="account"><option value="">default</option></select>
      <div class="hint">from accounts.json — accounts are config, not
        free text</div>
    </div>
    <div>
      <label for="kills">Stop after kills</label>
      <input id="kills" type="number" min="0" value="20">
      <div class="hint">0 = keep playing until stopped</div>
    </div>
    <div>
      <label for="seconds">Stop after seconds</label>
      <input id="seconds" type="number" min="0" value="0">
      <div class="hint">0 = no time limit</div>
    </div>
    <div>
      <button type="submit" class="primary">Launch</button>
    </div>
  </form>
  <div id="error"></div>
</section>
</main>
<footer>
  Stop is graceful — the bot finishes its fight, tops off, and quits with
  a full scorecard. Logs: <code>runs/bot/&lt;name&gt;/latest.log</code>,
  digest: <code>make digest</code>.
</footer>
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
      "error " + response.status + ": " + await response.text();
    return;
  }
  document.getElementById("error").textContent = "";
  refresh();
}

async function refreshStats(instance) {
  try {
    const response = await fetch("/bots/" + instance + "/stats");
    if (response.ok) statsCache[instance] = await response.json();
  } catch (e) { /* stats are best-effort; the row still renders */ }
}

function statusPill(bot, s) {
  if (bot.alive) return '<span class="pill alive">running</span>';
  if (s.available && s.clean_exit) return '<span class="pill done">finished</span>';
  return '<span class="pill crash">exit ' + bot.returncode + "</span>";
}

function row(bot) {
  const s = statsCache[bot.instance] || {};
  const tr = document.createElement("tr");
  if (!bot.alive) tr.className = "dead";
  const limits =
    (bot.kills ? bot.kills + " kills" : "") +
    (bot.kills && bot.seconds ? ", " : "") +
    (bot.seconds ? bot.seconds + " s" : "") || "none";
  const rank = s.available && s.rank_number >= 0
    ? '<span class="rank">' + s.rank_number + "</span>" : "";
  tr.innerHTML =
    "<td>" + bot.instance + "</td>" +
    "<td>" + (bot.account || "default") + "</td>" +
    "<td>" + statusPill(bot, s) + "</td>" +
    "<td>" + limits + "</td>" +
    "<td>" + (s.available ? s.kills : "") + "</td>" +
    "<td>" + (s.available ? s.deaths : "") + "</td>" +
    "<td>" + rank + "</td>" +
    "<td>" + (s.available ? fmtDuration(s.duration_s) : "") + "</td>";
  const actions = document.createElement("td");
  const stop = document.createElement("button");
  stop.textContent = "stop";
  stop.title = "graceful: finish the fight, top off, quit with a scorecard";
  stop.disabled = !bot.alive;
  stop.onclick = () => act("POST", "/bots/" + bot.instance + "/stop");
  const restart = document.createElement("button");
  restart.textContent = "restart";
  restart.title = "relaunch with the same account and limits";
  restart.disabled = bot.alive;
  restart.onclick = () => act("POST", "/bots/" + bot.instance + "/restart");
  const remove = document.createElement("button");
  remove.textContent = "remove";
  remove.title = "drop this row (bot must be stopped first)";
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
    tbody.innerHTML =
      '<tr><td colspan="9" class="empty">no bots yet — launch one below</td></tr>';
    return;
  }
  for (const bot of body.bots) tbody.appendChild(row(bot));
  for (const bot of body.bots) refreshStats(bot.instance);
}

async function loadAccounts() {
  try {
    const response = await fetch("/accounts");
    if (!response.ok) return;
    const body = await response.json();
    const select = document.getElementById("account");
    for (const name of body.accounts) {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      select.appendChild(option);
    }
    if (body.accounts.length) {
      select.options[0].textContent = "default (" + body.accounts[0] + ")";
    }
  } catch (e) { /* dropdown falls back to plain "default" */ }
}

document.getElementById("spawn").addEventListener("submit", async (event) => {
  event.preventDefault();
  const payload = {
    instance: document.getElementById("instance").value.trim(),
    account: document.getElementById("account").value,
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
      "launch failed (" + response.status + "): " + await response.text();
    return;
  }
  document.getElementById("error").textContent = "";
  document.getElementById("instance").value = "";
  refresh();
});

loadAccounts();
refresh();
setInterval(refresh, 3000);
</script>
</body>
</html>
"""

__all__ = ["FLEET_PAGE_HTML"]
