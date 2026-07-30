"""In-page HUD: a fixed-size fiesta-styled glass card + flag button.

Replaces the auto-sized green-on-black text block (which resized every
tick as line lengths changed) with a FIXED-GEOMETRY card: the DOM and
its stylesheet are built once, and every later tick only assigns text
and colors into pre-sized slots — the card never changes width or
height while the bot runs.

The look is carried over from the fiesta streaming SPA
(``~/PROJECTS/MCPs/fiesta/src/style.css``): the stippled frosted-glass
panel (blue tint, dot grid, backdrop blur, two-tone bevel border, blue
halo) and the console-button face (translucent neon fill, sheen +
corner glare, key-rim insets, emissive glow) — with the retro theme's
neon green / purple / hot pink palette projected by
:mod:`tankpit_bot.browser.overlay`.

The card is ``pointer-events: none`` so it can never eat a game
input; only the flag button opts back in. Clicking it calls the
:data:`~tankpit_bot.browser.flag_capture.FLAG_BINDING_NAME` binding
(when armed) with the click's timestamp and sequence number.
"""

from __future__ import annotations

from platform_core.json_utils import dump_json_str

from tankpit_bot._test_hooks import CDPSessionProtocol
from tankpit_bot.browser.flag_capture import FLAG_BINDING_NAME
from tankpit_bot.browser.overlay import OverlayStateDict, render_overlay_payload

HUD_ELEMENT_ID = "tankpit-bot-hud"
"""DOM id of the HUD card (and, suffixed ``-style``, its stylesheet)."""

_HUD_CSS = """
#tankpit-bot-hud{position:fixed;top:8px;right:8px;z-index:2147483000;width:272px;
pointer-events:none;color:#e9e9f0;
font:11px/1 system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
font-variant-numeric:tabular-nums;text-align:left;border-radius:10px;
background-color:rgba(24,34,80,0.28);
background-image:radial-gradient(rgba(255,255,255,0.04) 0.5px,transparent 0.7px);
background-size:9px 9px;
backdrop-filter:blur(6px) saturate(1.1);-webkit-backdrop-filter:blur(6px) saturate(1.1);
border:2px solid;
border-color:rgba(255,255,255,0.22) rgba(0,0,0,0.8) rgba(0,0,0,0.8) rgba(255,255,255,0.14);
box-shadow:0 8px 24px rgba(0,0,0,0.6),0 0 22px rgba(25,50,230,0.18),
inset 0 2px 3px rgba(255,255,255,0.1),inset 0 -3px 6px rgba(0,0,0,0.5);
overflow:hidden}
#tankpit-bot-hud,#tankpit-bot-hud *{box-sizing:border-box}
#tankpit-bot-hud .tph-row{display:flex;align-items:center;justify-content:space-between;
height:20px;padding:0 10px;white-space:nowrap;overflow:hidden}
#tankpit-bot-hud .tph-label{color:#9696a8;font-size:9px;letter-spacing:0.06em;
text-transform:uppercase;margin-right:4px}
#tankpit-bot-hud .tph-clip{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;
min-width:0;flex:1 1 auto}
#tankpit-bot-hud .tph-head{height:24px;padding-top:2px}
#tankpit-bot-hud .tph-title{color:rgb(57,255,20);font-weight:700;font-size:10px;
letter-spacing:0.12em;text-shadow:0 0 6px rgba(57,255,20,0.5)}
#tankpit-bot-hud .tph-chip{color:#9696a8;font-size:9px;letter-spacing:0.06em;
text-transform:uppercase}
#tankpit-bot-hud .tph-band{height:22px;line-height:22px;padding:0 10px;font-weight:700;
letter-spacing:0.10em;font-size:11px;text-transform:uppercase;white-space:nowrap;
overflow:hidden;text-shadow:0 0 8px currentColor;
border-top:1px solid rgba(255,255,255,0.08);border-bottom:1px solid rgba(0,0,0,0.45)}
#tankpit-bot-hud .tph-meter{height:6px;margin:3px 10px 4px;border-radius:3px;
background:rgba(0,0,0,0.55);box-shadow:inset 0 1px 2px rgba(0,0,0,0.8);overflow:hidden}
#tankpit-bot-hud .tph-fill{height:100%;width:0%;border-radius:3px;
box-shadow:0 0 6px currentColor;transition:width 300ms linear}
#tankpit-bot-hud .tph-stocks{display:grid;grid-template-columns:repeat(5,1fr);
height:20px;align-items:center;padding:0 10px}
#tankpit-bot-hud .tph-stocks b{font-weight:600;margin-left:3px}
#tankpit-bot-hud .tph-sent{width:14px;text-align:center;flex:0 0 auto}
#tankpit-bot-hud .tph-foot{height:32px;border-top:1px solid rgba(255,255,255,0.08)}
#tankpit-bot-hud .tph-stats b{font-weight:600;margin:0 6px 0 2px}
#tankpit-bot-hud .tph-flag{pointer-events:auto;cursor:pointer;height:22px;
padding:0 12px;border:none;border-radius:11px;color:rgba(245,240,255,0.92);
font:600 10px/1 system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;
letter-spacing:0.08em;background-color:rgba(200,0,200,0.32);
background-image:radial-gradient(120% 120% at 120% 120%,rgba(255,255,255,0.07) 30%,transparent 70%),
linear-gradient(rgba(255,255,255,0.07),rgba(255,255,255,0.015) 45%,rgba(0,0,0,0.12));
box-shadow:0 3px 6px rgba(0,0,0,0.35),0 0 10px rgba(200,0,200,0.22),
inset 0 1px 0 rgba(255,255,255,0.25),inset 0 -1px 0 rgba(0,0,0,0.4);
text-shadow:0 -1px 1px rgba(0,0,0,0.55),0 0 4px rgba(255,255,255,0.35);
user-select:none;font-variant-numeric:tabular-nums}
#tankpit-bot-hud .tph-flag:active{transform:scale(0.92)}
#tankpit-bot-hud .tph-flag.tph-flash{background-color:rgba(57,255,20,0.45)}
"""

_HUD_BODY = """
<div class="tph-row tph-head"><span class="tph-title">TANKPIT BOT</span>
<span id="tph-state" class="tph-chip"></span></div>
<div id="tph-mode" class="tph-band"></div>
<div class="tph-row"><span><span class="tph-label">pos</span><span id="tph-pos"></span></span>
<span><span class="tph-label">fuel</span><span id="tph-fuel"></span></span></div>
<div class="tph-meter"><div id="tph-fill" class="tph-fill"></div></div>
<div class="tph-stocks">
<span><span class="tph-label">AR</span><b id="tph-s0"></b></span>
<span><span class="tph-label">DU</span><b id="tph-s1"></b></span>
<span><span class="tph-label">MI</span><b id="tph-s2"></b></span>
<span><span class="tph-label">HO</span><b id="tph-s3"></b></span>
<span><span class="tph-label">RA</span><b id="tph-s4"></b></span>
</div>
<div class="tph-row"><span class="tph-clip"><span class="tph-label">do</span>
<span id="tph-do"></span></span><span id="tph-sent" class="tph-sent"></span></div>
<div class="tph-row"><span class="tph-clip"><span class="tph-label">why</span>
<span id="tph-why"></span></span></div>
<div class="tph-row"><span class="tph-clip"><span class="tph-label">tgt</span>
<span id="tph-tgt"></span></span><span><span class="tph-label">act</span>
<span id="tph-act"></span></span></div>
<div class="tph-row tph-foot"><span class="tph-stats">
<span class="tph-label">K</span><b id="tph-k"></b>
<span class="tph-label">H</span><b id="tph-h"></b>
<span class="tph-label">M</span><b id="tph-m"></b>
<span class="tph-label">RJ</span><b id="tph-rj"></b>
</span><button id="tph-flag" class="tph-flag" type="button">&#9873; FLAG</button></div>
"""

_UPDATE_TEMPLATE = """
(() => {
  const d = __PAYLOAD__;
  let el = document.getElementById('tankpit-bot-hud');
  if (!el) {
    const style = document.createElement('style');
    style.id = 'tankpit-bot-hud-style';
    style.textContent = __CSS__;
    document.head.appendChild(style);
    el = document.createElement('div');
    el.id = 'tankpit-bot-hud';
    el.innerHTML = __BODY__;
    document.body.appendChild(el);
    const btn = el.querySelector('#tph-flag');
    btn.addEventListener('click', () => {
      const n = (window.__tpHudFlags = (window.__tpHudFlags || 0) + 1);
      btn.textContent = '\\u2691 ' + n;
      btn.classList.add('tph-flash');
      setTimeout(() => btn.classList.remove('tph-flash'), 350);
      if (typeof window.__FLAG_BINDING__ === 'function') {
        window.__FLAG_BINDING__(JSON.stringify({
          clicked_at_ms: Date.now(),
          flag_seq: n,
        }));
      }
    });
  }
  const t = (id, v) => {
    const s = document.getElementById(id);
    const text = String(v);
    if (s.textContent !== text) { s.textContent = text; }
  };
  t('tph-state', d.state_text);
  const mode = document.getElementById('tph-mode');
  mode.textContent = d.mode_text;
  mode.style.color = d.mode_color;
  mode.style.background = d.mode_band;
  t('tph-pos', d.pos_text);
  t('tph-fuel', d.fuel_text);
  const fill = document.getElementById('tph-fill');
  fill.style.width = d.fuel_pct + '%';
  fill.style.background = d.fuel_color;
  fill.style.color = d.fuel_color;
  for (const i of [0, 1, 2, 3, 4]) {
    const slot = document.getElementById('tph-s' + i);
    slot.textContent = String(d['s' + i]);
    slot.style.color = d['s' + i + 'c'];
  }
  t('tph-do', d.do_text);
  const sent = document.getElementById('tph-sent');
  sent.textContent = d.sent_text;
  sent.style.color = d.sent_color;
  t('tph-why', d.why_text);
  t('tph-tgt', d.tgt_text);
  t('tph-act', d.act_text);
  t('tph-k', d.kills);
  t('tph-h', d.hits);
  t('tph-m', d.misses);
  t('tph-rj', d.rejects);
  return true;
})()
"""


def build_hud_expression(overlay: OverlayStateDict) -> str:
    """Render the one-shot HUD update expression for a tick.

    Args:
        overlay: This tick's payload.

    Returns:
        A self-contained JS expression: first evaluation installs the
        stylesheet, card DOM, and flag-button handler; every
        evaluation assigns the payload into the fixed slots.
    """
    return (
        _UPDATE_TEMPLATE.replace("__PAYLOAD__", dump_json_str(render_overlay_payload(overlay)))
        .replace("__CSS__", dump_json_str(_HUD_CSS))
        .replace("__BODY__", dump_json_str(_HUD_BODY))
        .replace("__FLAG_BINDING__", FLAG_BINDING_NAME)
    )


def update_bot_overlay(cdp: CDPSessionProtocol, overlay: OverlayStateDict) -> None:
    """Create or update the in-page HUD with this tick's payload.

    Args:
        cdp: Active CDP session attached to the live tankpit page.
        overlay: Payload to render.
    """
    cdp.send(
        "Runtime.evaluate",
        {
            "expression": build_hud_expression(overlay),
            "returnByValue": True,
        },
    )


__all__ = [
    "HUD_ELEMENT_ID",
    "build_hud_expression",
    "update_bot_overlay",
]
