"""Bot updates inside the game's preset-chat panel, captured with the video.

The game's select stays in place for its own layout and references. Its
replacement uses the select's computed font, colors and border, occupies
the same rectangle, and resizes with it. No caption clock or video delay is
involved: the screen recorder sees the words beside the action itself.
"""

from __future__ import annotations

from platform_core.json_utils import dump_json_str

from tankpit_bot._test_hooks import CDPSessionProtocol
from tankpit_bot.browser.cdp_utils import _extract_runtime_value
from tankpit_bot.service.demo_caption_words import CaptionWordsDict, encode_caption_words

CAPTION_PANEL_ID = "tankpit-bot-status"

_UPDATE = """
(() => {
    const words = __WORDS__;
    const select = document.querySelector('.message-list select');
    if (!select) return 'CAPTION_PANEL_MISSING: game preset-chat selector not found';
    let panel = document.getElementById('tankpit-bot-status');
    if (!panel) {
        panel = document.createElement('div');
        panel.id = 'tankpit-bot-status';
        panel.setAttribute('role', 'status');
        panel.setAttribute('aria-live', 'polite');
        panel.style.cssText = 'position:absolute;inset:0;box-sizing:border-box;'
            + 'padding:3px 4px;overflow:hidden;pointer-events:none;white-space:normal;';
        const doing = document.createElement('div');
        doing.style.cssText = 'font-weight:bold;margin-bottom:3px;';
        const why = document.createElement('div');
        panel.append(doing, why);
        const parent = select.parentElement;
        // The live client lays this box out in normal flow. Establish its
        // own containing block so inset:0 cannot cover the combat log.
        if (getComputedStyle(parent).position === 'static') {
            parent.style.position = 'relative';
        }
        parent.append(panel);
        const matchStyle = () => {
            const style = getComputedStyle(select);
            panel.style.font = style.font;
            panel.style.lineHeight = '1.15';
            panel.style.color = style.color;
            panel.style.background = style.backgroundColor;
            panel.style.border = style.border;
        };
        matchStyle();
        new ResizeObserver(matchStyle).observe(select);
        // Opacity preserves the game's layout and the log scraper's text
        // boundary; disable all interaction with the invisible chat control.
        select.style.opacity = '0';
        select.style.pointerEvents = 'none';
        select.inert = true;
        select.tabIndex = -1;
        select.setAttribute('aria-hidden', 'true');
    }
    panel.children[0].textContent = words.doing;
    panel.children[1].textContent = words.why;
    return 'CAPTION_PANEL_UPDATED';
})()
"""


def update_caption_panel(cdp: CDPSessionProtocol, words: CaptionWordsDict) -> None:
    """Paint an update in the game's existing chat-selection rectangle.

    Args:
        cdp: The bot browser's real session.
        words: What the bot is doing and why.

    Raises:
        RuntimeError: The client layout changed or the script failed.
        ValueError: The browser returned no evaluation value.
    """
    expression = _UPDATE.replace("__WORDS__", dump_json_str(encode_caption_words(words)))
    result = cdp.send("Runtime.evaluate", {"expression": expression, "returnByValue": True})
    value = _extract_runtime_value(result)
    if value != "CAPTION_PANEL_UPDATED":
        raise RuntimeError(f"CAPTION_PANEL_UPDATE_FAILED: {value}")


__all__ = ["CAPTION_PANEL_ID", "update_caption_panel"]
