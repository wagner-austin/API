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
  totals from the digest, stop/restart/remove, and the launch form.
  Every selector on that form is a dropdown, never free text:
  accounts come from ``GET /accounts`` (accounts.json), rooms from
  ``GET /rooms`` (:mod:`tankpit_bot.types.rooms`), roles from the
  fleet-role vocabulary. Nothing on this page asks a human to
  remember a spelling.

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
<title>Tankpit Fleet</title>
<style>
  body { margin:0; color:#e9e9f0;
         font-family:"Segoe UI",system-ui,sans-serif;
         background:#0c0f16 radial-gradient(1100px 700px at 18% -8%,
           rgba(37,52,120,.55), transparent 62%) no-repeat fixed; }
  header { padding:1rem 1.6rem; display:flex; align-items:baseline;
           gap:1rem; border-bottom:1px solid rgba(255,255,255,.09);
           background:rgba(24,34,80,.22);
           backdrop-filter:blur(6px) saturate(1.1); }
  header h1 { margin:0; font-size:1.2rem; letter-spacing:.06em;
              color:rgb(57,255,20);
              text-shadow:0 0 8px rgba(57,255,20,.45); }
  header .sub { color:#9aa3b5; font-size:.85rem; }
  .dot { display:inline-block; width:.55em; height:.55em;
         border-radius:50%; margin-right:.4em;
         background:#5ecb71; box-shadow:0 0 6px #5ecb71; }
  .dot.off { background:#e0656a; box-shadow:0 0 6px #e0656a; }
  main { padding:1.3rem 1.6rem; display:grid; gap:1.3rem; max-width:1240px; }
  #huds { display:grid; gap:1.1rem;
          grid-template-columns:repeat(auto-fill, 272px);
          align-items:start; justify-content:start; }
  .hudwrap { width:272px; }
  .hudwrap .hudname { font:600 .8rem "Segoe UI",sans-serif; color:#9aa3b5;
                      margin:0 0 .35rem .2rem; text-transform:uppercase;
                      letter-spacing:.08em; height:1.1rem;
                      overflow:hidden; white-space:nowrap; }
  .panel { background:rgba(24,34,80,.28); border-radius:10px;
           border:2px solid; padding:.2rem;
           border-color:rgba(255,255,255,.22) rgba(0,0,0,.8)
             rgba(0,0,0,.8) rgba(255,255,255,.14);
           box-shadow:0 8px 24px rgba(0,0,0,.6),
             0 0 22px rgba(25,50,230,.14);
           backdrop-filter:blur(6px) saturate(1.1); }
  table { border-collapse:collapse; width:100%; font-size:.9rem; }
  th, td { border-bottom:1px solid rgba(255,255,255,.07);
           padding:.5rem .8rem; text-align:left; white-space:nowrap; }
  tr:last-child td { border-bottom:none; }
  th { color:#9aa3b5; font-size:.76rem; text-transform:uppercase;
       letter-spacing:.07em; border-bottom:1px solid rgba(255,255,255,.14); }
  tr.dead td { color:#6d7585; }
  .alive { color:#5ecb71; font-weight:600; }
  .done { color:#8a93a3; } .crash { color:#e0656a; }
  .rank { color:#d9b45b; font-weight:600; }
  .lb { color:#8a93a3; font-size:.8em; }
  button { background:#1d232d; color:#d6dae2; border:1px solid #39424f;
           padding:.3rem .8rem; border-radius:6px; margin-right:.35rem;
           cursor:pointer; font-size:.82rem; }
  button:hover { background:#242c38; }
  button:disabled { color:#555e6b; border-color:#2a313c; cursor:default; }
  button.primary { background:#67b0e8; color:#0d141b; border-color:#67b0e8;
                   font-weight:600; height:34px; padding:0 1.4rem;
                   box-sizing:border-box; }
  form { display:flex; flex-wrap:nowrap; gap:1.1rem;
         align-items:flex-start; overflow-x:auto;
         padding:1rem 1.2rem; background:rgba(24,34,80,.28);
         border-radius:10px; border:2px solid;
         backdrop-filter:blur(6px) saturate(1.1);
         border-color:rgba(255,255,255,.22) rgba(0,0,0,.8)
           rgba(0,0,0,.8) rgba(255,255,255,.14);
         box-shadow:0 8px 24px rgba(0,0,0,.6); }
  /* Every field is WIDTH-BOUND. The form is nowrap over overflow-x,
     so one content-sized field with a long hint scrolls the whole row
     sideways — which is exactly what the colour readout did. The
     modifiers below widen; nothing sizes to its text. */
  form .field { display:flex; flex-direction:column; flex:0 0 auto; width:130px; }
  form .field.wide { width:150px; }
  form .field.num { width:130px; }
  /* The colour field carries the widest hint on the form (five slot
     counts, or the empty-state line). Content-sized, it stretched the
     nowrap row until the whole form scrolled sideways. */
  form .field.tank { width:200px; }
  label { display:block; height:1.1rem; font-size:.75rem; color:#8a93a3;
          margin-bottom:.3rem; text-transform:uppercase;
          letter-spacing:.05em; white-space:nowrap; }
  .hint { height:2.4em; overflow:hidden; font-size:.73rem;
          color:#8a93a3; margin-top:.3rem; }
  input, select { width:100%; height:34px; box-sizing:border-box;
                  background:#0f1319; color:#d6dae2;
                  border:1px solid #39424f; border-radius:6px;
                  padding:0 .6rem; font-size:.88rem; }
  #error { color:#e0656a; min-height:1.1rem; font-size:.85rem; }
  .empty { color:#8a93a3; }
__CARD_CSS__
</style>
</head>
<body>
<header><h1>Tankpit Fleet</h1>
<span class="sub" id="headline"><span class="dot off"></span>connecting…</span></header>
<main>
<div id="huds"></div>
<div class="panel">
<table>
  <thead><tr>
    <th>name</th><th>account</th><th>role</th><th>room</th><th>color</th><th>status</th><th>limits</th>
    <th>kills</th><th>deaths</th><th>hit/miss</th><th>dmg +/-</th>
    <th>tp</th><th>0-radar</th><th>inv start&rarr;now</th>
    <th>rank</th><th>time</th><th>actions</th>
  </tr></thead>
  <tbody id="rows"><tr><td colspan="17" class="empty">loading…</td></tr></tbody>
</table>
</div>
<form id="spawn">
  <div class="field wide"><label for="account">Account</label>
    <select id="account"><option value="">default</option></select>
    <div class="hint">from accounts.json</div></div>
  <div class="field"><label for="role">Role</label>
    <select id="role">
      <option value="fighter">Fighter</option>
      <option value="gatherer">Gatherer</option>
    </select>
    <div class="hint">gatherer never hunts</div></div>
  <div class="field wide"><label for="room">Room</label>
    <select id="room"><option value="">Practice</option></select>
    <div class="hint">the lobby's two rooms</div></div>
  <div class="field tank"><label for="troop">Color</label>
    <select id="troop"></select>
    <div class="hint" id="troopinfo">own rank, fuel and stock</div></div>
  <div class="field num"><label for="kills">Stop after kills</label>
    <input id="kills" type="number" min="0" value="20">
    <div class="hint">0 = play until stopped</div></div>
  <div class="field num"><label for="seconds">Stop after seconds</label>
    <input id="seconds" type="number" min="0" value="0">
    <div class="hint">0 = no time limit</div></div>
  <div class="field"><label>&nbsp;</label>
    <button type="submit" class="primary">Launch</button>
    <div class="hint"></div></div>
</form>
<div id="error"></div>
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
  const h = huds[bot.instance];
  const tr = document.createElement("tr");
  if (!bot.alive) tr.className = "dead";
  const status = bot.alive ? '<span class="alive">running</span>'
    : (s.available && s.clean_exit
       ? '<span class="done">' + (s.exit_reason || "finished") + "</span>"
       : '<span class="crash">exit ' + bot.returncode + "</span>");
  const limits = ((bot.kills ? bot.kills + " kills" : "") +
    (bot.kills && bot.seconds ? " / " : "") +
    (bot.seconds ? bot.seconds + " s" : "")) || "none";
  const up = bot.alive ? Math.floor((Date.now() - bot.started_ms) / 1000)
                       : (s.available ? s.duration_s : -1);
  tr.innerHTML =
    "<td>" + bot.instance + "</td><td>" + (bot.account || "default") + "</td>" +
    "<td>" + bot.role + "</td>" +
    "<td>" + (bot.room || "Practice") + "</td>" +
    "<td>" + (bot.troop || "default") + "</td>" +
    "<td>" + status + "</td><td>" + limits + "</td>" +
    "<td>" + (s.available ? s.kills : "") + "</td>" +
    "<td>" + (s.available ? s.deaths : "") + "</td>" +
    "<td>" + (s.available ? s.hits + "/" + s.misses : "") + "</td>" +
    // The damage ledger is emitted at TEARDOWN with fuel-confirmed
    // totals, so a LIVE bot has none — printing "0 / 0" claimed it
    // had dealt and taken nothing, which is a different statement.
    // Live rows read the HUD payload, which carries the damage book
    // as of this tick; finished rows read the digest's teardown
    // ledger. Same numbers, different channel — the digest simply
    // does not exist until the bot exits.
    "<td>" + (bot.alive
      ? (h ? h.dealt + " / " + h.taken : "—")
      : (s.available ? s.damage_dealt + " / " + s.damage_taken : "")) + "</td>" +
    "<td>" + (s.available ? s.teleports : "") + "</td>" +
    "<td>" + (s.available ? s.zero_yield_radars : "") + "</td>" +
    "<td>" + (s.available && s.inventory_first.length === 5
      ? s.inventory_first.join("·") + " → " + s.inventory_last.join("·")
      : "") + "</td>" +
    // The WORD is the rank; the number beside it is the leaderboard
    // position, which is what this column used to show on its own.
    '<td><span class="rank">' + (s.available && s.rank_name ? s.rank_name : "") +
    "</span>" + (s.available && s.leaderboard_position >= 0
      ? ' <span class="lb">#' + s.leaderboard_position + "</span>" : "") + "</td>" +
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
    document.getElementById("headline").innerHTML =
      '<span class="dot off"></span>fleet unreachable';
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
      '<tr><td colspan="17" class="empty">no bots yet — launch one below</td></tr>';
  }
  for (const name of names) tbody.appendChild(row(registry[name]));
  const running = names.filter((n) => registry[n].alive).length;
  document.getElementById("headline").innerHTML =
    '<span class="dot"></span>fleet online · ' +
    names.length + " bot" + (names.length === 1 ? "" : "s") +
    " · " + running + " running";
}

async function fillSelect(id, path, key) {
  try {
    const body = await (await fetch(path)).json();
    const select = document.getElementById(id);
    if (!body[key].length) return;
    select.replaceChildren();
    for (const name of body[key]) {
      const option = document.createElement("option");
      option.value = name;
      option.textContent = name;
      select.appendChild(option);
    }
  } catch (e) { /* dropdown falls back to plain default */ }
}

const RANKS = ["recruit", "private", "corporal", "sergeant", "lieutenant",
               "captain", "major", "colonel", "general"];
let tanks = {};

// Rank is measured, never derived: the lobby names only the colour an
// account played LAST, so the other three are blank until somebody
// enters them. Blank says "not measured", never "recruit".
function paintTroopInfo() {
  const hint = document.getElementById("troopinfo");
  const account = document.getElementById("account").value;
  const room = document.getElementById("room").value || "Practice";
  const colour = document.getElementById("troop").value;
  // The registry nests under "accounts"; its other top-level keys are
  // provenance and per-room facts, not tanks.
  const byAccount = tanks.accounts || {};
  const cell = ((byAccount[account] || {})[room] || {})[colour];
  if (cell === undefined) {
    hint.textContent = "no reading for " + colour + " on " + room;
    return;
  }
  // Line 1 is what the tank IS: rank, and the fuel it actually has —
  // the value the last session left, not the rank-derived cap, since
  // a parked tank keeps what it had. The cap only appears beside it,
  // as the ceiling that fuel is measured against.
  const rank = RANKS.indexOf(cell.rank);
  let head = cell.rank;
  if (cell.fuel !== undefined) {
    head += " · fuel " + cell.fuel + (rank >= 0 ? "/" + (1000 + 100 * rank) : "");
  } else if (rank >= 0) {
    head += " · cap " + (1000 + 100 * rank);
  }
  if (cell.kills !== undefined) { head += " · K" + cell.kills + " D" + cell.deaths; }
  // Line 2 is what it CARRIES, in the HUD's slot order.
  const slots = ["AR", "DU", "MI", "HO", "RA"];
  const inv = cell.inventory
    ? slots.map((label, i) => label + cell.inventory[i]).join(" ")
    : "no stock reading — play it once";
  hint.replaceChildren(head, document.createElement("br"), inv);
}

async function loadTanks() {
  try {
    tanks = (await (await fetch("/tanks")).json()).tanks;
  } catch (e) { tanks = {}; }
}

for (const id of ["account", "room", "troop"]) {
  document.getElementById(id).addEventListener("change", paintTroopInfo);
}

document.getElementById("spawn").addEventListener("submit", async (event) => {
  event.preventDefault();
  const response = await fetch("/bots", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      account: document.getElementById("account").value,
      role: document.getElementById("role").value,
      room: document.getElementById("room").value,
      troop: document.getElementById("troop").value,
      kills: Number(document.getElementById("kills").value) || 0,
      seconds: Number(document.getElementById("seconds").value) || 0,
    }),
  });
  document.getElementById("error").textContent = response.ok
    ? "" : "launch failed (" + response.status + "): " + await response.text();
  if (response.ok) poll();
});

// One paint, after ALL of account, room, colour and the registry have
// landed. Painting per-fill raced: whichever resolved last decided
// what the readout described, and a late room fill left it reading
// the Practice fallback while the dropdown displayed World.
Promise.all([
  fillSelect("account", "/accounts", "accounts"),
  fillSelect("room", "/rooms", "rooms"),
  fillSelect("troop", "/troops", "troops"),
  loadTanks(),
]).then(paintTroopInfo);
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
