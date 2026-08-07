"""The fleet manager's control page — one HTML file, no dependencies.

Served at ``GET /`` on the fleet port (default 27300). Two layers:

* **The HUD strip** — one glass card per bot, and it is literally the
  ``make run`` overlay: the CSS and body come from
  :mod:`tankpit_bot.browser.overlay_hud` (id selector re-scoped to a
  class so many cards coexist), and the payload comes from
  ``GET /bots/{i}/hud`` — the exact per-tick dict the in-page HUD
  renders. Mode band, fuel meter, stocks, do/why/tgt, K/H/M/RJ — all
  identical to what a human sees over the live game.
* **The fleet table** — lifecycle: status, limits, kills/deaths/rank
  totals from the digest, stop/restart/remove, and the launch form
  (accounts from config, never free text).

Both poll every second; the server caches digest work.
"""

from __future__ import annotations

from tankpit_bot.browser.overlay_hud import _HUD_BODY, _HUD_CSS

_CARD_CSS = (
    _HUD_CSS.replace("#tankpit-bot-hud", ".tph-card")
    .replace("position:fixed;top:8px;right:8px;z-index:2147483000;", "position:relative;")
    .replace(".tph-card .tph-flag{pointer-events:auto", ".tph-card .tph-flag{display:none")
)
# The flag button is hidden: flags belong to the human WATCHING the
# live game window, where the click lands in that run's event ledger.

_PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>tankpit fleet</title>
<style>
  body { background:#0f1115; color:#d6dae2; margin:0;
         font-family:"Segoe UI",system-ui,sans-serif; }
  header { padding:1rem 1.6rem; border-bottom:1px solid #262c37;
           display:flex; align-items:baseline; gap:1rem; }
  header h1 { margin:0; font-size:1.15rem; letter-spacing:.03em; }
  header .sub { color:#8a93a3; font-size:.85rem; }
  main { padding:1.3rem 1.6rem; display:grid; gap:1.3rem; max-width:1200px; }
  #huds { display:flex; flex-wrap:wrap; gap:1rem; }
  .hudwrap .hudname { font:600 .8rem "Segoe UI",sans-serif; color:#8a93a3;
                      margin:0 0 .3rem .2rem; text-transform:uppercase;
                      letter-spacing:.06em; }
  table { border-collapse:collapse; width:100%; font-size:.9rem; }
  th, td { border:1px solid #262c37; padding:.45rem .7rem; text-align:left; }
  th { background:#161a21; color:#8a93a3; font-size:.78rem;
       text-transform:uppercase; letter-spacing:.05em; }
  tr.dead td { color:#6d7585; }
  .alive { color:#5ecb71; font-weight:600; }
  .done { color:#8a93a3; } .crash { color:#e0656a; }
  .rank { color:#d9b45b; font-weight:600; }
  button { background:#1d232d; color:#d6dae2; border:1px solid #39424f;
           padding:.3rem .8rem; border-radius:6px; margin-right:.35rem;
           cursor:pointer; font-size:.82rem; }
  button:hover { background:#242c38; }
  button:disabled { color:#555e6b; border-color:#2a313c; cursor:default; }
  button.primary { background:#67b0e8; color:#0d141b; border-color:#67b0e8;
                   font-weight:600; padding:.45rem 1.2rem; }
  form { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
         gap:.9rem 1.1rem; align-items:end; background:#161a21;
         border:1px solid #262c37; border-radius:10px; padding:1rem 1.2rem; }
  label { display:block; font-size:.75rem; color:#8a93a3; margin-bottom:.3rem;
          text-transform:uppercase; letter-spacing:.05em; }
  .hint { font-size:.73rem; color:#8a93a3; margin-top:.3rem; }
  input, select { width:100%; background:#0f1319; color:#d6dae2;
                  border:1px solid #39424f; border-radius:6px;
                  padding:.45rem .6rem; font-size:.88rem; }
  #error { color:#e0656a; min-height:1.1rem; font-size:.85rem; }
  .empty { color:#8a93a3; }
__CARD_CSS__
</style>
</head>
<body>
<header><h1>tankpit fleet</h1>
<span class="sub" id="headline">connecting…</span></header>
<main>
<div id="huds"></div>
<table>
  <thead><tr>
    <th>name</th><th>account</th><th>status</th><th>limits</th>
    <th>kills</th><th>deaths</th><th>rank</th><th>time</th><th>actions</th>
  </tr></thead>
  <tbody id="rows"><tr><td colspan="9" class="empty">loading…</td></tr></tbody>
</table>
<form id="spawn">
  <div><label for="instance">Name</label>
    <input id="instance" placeholder="e.g. alpha" required
           pattern="[a-z0-9][a-z0-9_-]{0,31}">
    <div class="hint">logs land in runs/bot/&lt;name&gt;/</div></div>
  <div><label for="account">Account</label>
    <select id="account"><option value="">default</option></select>
    <div class="hint">from accounts.json — config, not free text</div></div>
  <div><label for="kills">Stop after kills</label>
    <input id="kills" type="number" min="0" value="20">
    <div class="hint">0 = play until stopped</div></div>
  <div><label for="seconds">Stop after seconds</label>
    <input id="seconds" type="number" min="0" value="0">
    <div class="hint">0 = no time limit</div></div>
  <div><button type="submit" class="primary">Launch</button>
    <div id="error"></div></div>
</form>
</main>
<script>
"use strict";
const HUD_BODY = __HUD_BODY__;
const registry = {};
const stats = {};
const huds = {};

function fmtDuration(s) {
  if (s === null || s === undefined || s < 0) return "";
  return Math.floor(s / 60) + "m" + String(s % 60).padStart(2, "0") + "s";
}

async function act(method, path) {
  const response = await fetch(path, {method});
  document.getElementById("error").textContent = response.ok
    ? "" : "error " + response.status + ": " + await response.text();
  if (response.ok) poll();
}

function hudCard(name) {
  let wrap = document.querySelector('[data-hud="' + name + '"]');
  if (!wrap) {
    wrap = document.createElement("div");
    wrap.className = "hudwrap";
    wrap.dataset.hud = name;
    const title = document.createElement("div");
    title.className = "hudname";
    title.textContent = name;
    const card = document.createElement("div");
    card.className = "tph-card";
    card.innerHTML = HUD_BODY;
    wrap.append(title, card);
    document.getElementById("huds").appendChild(wrap);
  }
  return wrap.querySelector(".tph-card");
}

function paintHud(name) {
  const d = huds[name];
  const card = hudCard(name);
  if (!d || d.available === false) {
    card.style.opacity = ".45";
    return;
  }
  card.style.opacity = registry[name] && registry[name].alive ? "1" : ".6";
  const q = (id) => card.querySelector("#" + id);
  const t = (id, v) => { const s = q(id); if (s) s.textContent = String(v); };
  t("tph-state", d.state_text);
  const mode = q("tph-mode");
  mode.textContent = d.mode_text;
  mode.style.color = d.mode_color;
  mode.style.background = d.mode_band;
  t("tph-pos", d.pos_text);
  t("tph-fuel", d.fuel_text);
  const fill = q("tph-fill");
  fill.style.width = d.fuel_pct + "%";
  fill.style.background = d.fuel_color;
  for (const i of [0, 1, 2, 3, 4]) {
    const slot = q("tph-s" + i);
    slot.textContent = String(d["s" + i]);
    slot.style.color = d["s" + i + "c"];
  }
  t("tph-do", d.do_text);
  const sent = q("tph-sent");
  sent.textContent = d.sent_text;
  sent.style.color = d.sent_color;
  t("tph-why", d.why_text);
  t("tph-tgt", d.tgt_text);
  t("tph-act", d.act_text);
  t("tph-k", d.kills); t("tph-h", d.hits);
  t("tph-m", d.misses); t("tph-rj", d.rejects);
}

function row(bot) {
  const s = stats[bot.instance] || {};
  const tr = document.createElement("tr");
  if (!bot.alive) tr.className = "dead";
  const status = bot.alive ? '<span class="alive">running</span>'
    : (s.available && s.clean_exit
       ? '<span class="done">' + (s.exit_reason || "finished") + "</span>"
       : '<span class="crash">exit ' + bot.returncode + "</span>");
  const limits = ((bot.kills ? bot.kills + "k" : "") +
    (bot.kills && bot.seconds ? " / " : "") +
    (bot.seconds ? bot.seconds + "s" : "")) || "none";
  const up = bot.alive ? Math.floor((Date.now() - bot.started_ms) / 1000)
                       : (s.available ? s.duration_s : -1);
  tr.innerHTML =
    "<td>" + bot.instance + "</td><td>" + (bot.account || "default") + "</td>" +
    "<td>" + status + "</td><td>" + limits + "</td>" +
    "<td>" + (s.available ? s.kills : "") + "</td>" +
    "<td>" + (s.available ? s.deaths : "") + "</td>" +
    '<td><span class="rank">' +
    (s.available && s.rank_number >= 0 ? s.rank_number : "") + "</span></td>" +
    "<td>" + fmtDuration(up) + "</td>";
  const actions = document.createElement("td");
  for (const [label, method, path, disabled] of [
    ["stop", "POST", "/stop", !bot.alive],
    ["restart", "POST", "/restart", bot.alive],
    ["remove", "DELETE", "", bot.alive],
  ]) {
    const button = document.createElement("button");
    button.textContent = label;
    button.disabled = disabled;
    button.onclick = () => act(method, "/bots/" + bot.instance + path);
    actions.appendChild(button);
  }
  tr.appendChild(actions);
  return tr;
}

async function poll() {
  try {
    const body = await (await fetch("/bots")).json();
    for (const name of Object.keys(registry)) delete registry[name];
    for (const bot of body.bots) registry[bot.instance] = bot;
  } catch (e) {
    document.getElementById("headline").textContent = "fleet unreachable";
    return;
  }
  const names = Object.keys(registry).sort();
  for (const name of names) {
    try {
      const [statsResp, hudResp] = await Promise.all([
        fetch("/bots/" + name + "/stats"), fetch("/bots/" + name + "/hud")]);
      if (statsResp.ok) stats[name] = await statsResp.json();
      if (hudResp.ok) huds[name] = await hudResp.json();
    } catch (e) { /* keep last known */ }
    paintHud(name);
  }
  for (const wrap of document.querySelectorAll(".hudwrap")) {
    if (!registry[wrap.dataset.hud]) wrap.remove();
  }
  const tbody = document.getElementById("rows");
  tbody.replaceChildren();
  if (!names.length) {
    tbody.innerHTML =
      '<tr><td colspan="9" class="empty">no bots yet — launch one below</td></tr>';
  }
  for (const name of names) tbody.appendChild(row(registry[name]));
  const running = names.filter((n) => registry[n].alive).length;
  document.getElementById("headline").textContent =
    names.length + " bot" + (names.length === 1 ? "" : "s") +
    " · " + running + " running";
}

async function loadAccounts() {
  try {
    const body = await (await fetch("/accounts")).json();
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
  } catch (e) { /* dropdown falls back to plain default */ }
}

document.getElementById("spawn").addEventListener("submit", async (event) => {
  event.preventDefault();
  const response = await fetch("/bots", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      instance: document.getElementById("instance").value.trim(),
      account: document.getElementById("account").value,
      kills: Number(document.getElementById("kills").value) || 0,
      seconds: Number(document.getElementById("seconds").value) || 0,
    }),
  });
  document.getElementById("error").textContent = response.ok
    ? "" : "launch failed (" + response.status + "): " + await response.text();
  if (response.ok) { document.getElementById("instance").value = ""; poll(); }
});

loadAccounts();
poll();
setInterval(poll, 1000);
</script>
</body>
</html>
"""

FLEET_PAGE_HTML = _PAGE_TEMPLATE.replace("__CARD_CSS__", _CARD_CSS).replace(
    "__HUD_BODY__", repr(_HUD_BODY)
)

__all__ = ["FLEET_PAGE_HTML"]
