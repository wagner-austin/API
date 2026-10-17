"""Run the status-panel script in real Chromium, including layout drift."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
from platform_core.json_utils import narrow_json_to_dict

from tankpit_bot._test_hooks import CDPSessionProtocol, PageProtocol
from tankpit_bot._test_hooks.playwright_loader import _real_get_sync_playwright
from tankpit_bot.browser.caption_panel import update_caption_panel
from tankpit_bot.service.demo_caption_words import CaptionWordsDict

_PAGE = """
<style>
.game-right {position:absolute;left:400px;top:240px;width:180px}
.history {height:180px}
.message-list {width:180px;height:88px}
.message-list select {width:100%;height:100%;font:10px Arial;background:#ffffe0;color:#000}
</style>
<div class="game-right"><div class="history">You hit orange-5</div>
<div class="message-list"><select size="10"><option>Attack the red</option>
<option>Be right back</option></select></div></div>
"""


@pytest.fixture()
def client() -> Generator[tuple[PageProtocol, CDPSessionProtocol], None, None]:
    """Yield a real page with the game's chat-panel DOM contract.

    Yields:
        A page and its real CDP session; no network or account is involved.
    """
    with _real_get_sync_playwright()() as playwright:
        browser = playwright.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        page.set_content(_PAGE)
        try:
            yield page, context.new_cdp_session(page)
        finally:
            browser.close()


def test_panel_replaces_chat_without_changing_game_geometry(
    client: tuple[PageProtocol, CDPSessionProtocol],
) -> None:
    """The real DOM retains the game's frame, font, background and log."""
    page, cdp = client
    update_caption_panel(
        cdp,
        CaptionWordsDict(
            doing="Heading to equipment",
            why="It is going to collect the equipment it found nearby.",
        ),
    )
    result = narrow_json_to_dict(
        page.evaluate("""() => {
        const panel = document.getElementById('tankpit-bot-status');
        const select = document.querySelector('select');
        const p = panel.getBoundingClientRect(), s = select.getBoundingClientRect();
        const ps = getComputedStyle(panel), ss = getComputedStyle(select);
        return {
            sameRect: p.x === s.x && p.y === s.y && p.width === s.width && p.height === s.height,
            sameFont: ps.fontFamily === ss.fontFamily && ps.fontSize === ss.fontSize,
            sameColor: ps.color === ss.color && ps.backgroundColor === ss.backgroundColor,
            fits: panel.scrollHeight <= panel.clientHeight,
            chatInert: select.inert && select.tabIndex === -1 && ss.opacity === '0',
            log: document.querySelector('.history').innerText,
            words: panel.innerText,
        };
    }""")
    )
    assert result == {
        "sameRect": True,
        "sameFont": True,
        "sameColor": True,
        "fits": True,
        "chatInert": True,
        "log": "You hit orange-5",
        "words": "Heading to equipment\nIt is going to collect the equipment it found nearby.",
    }
    assert page.evaluate("document.body.innerText.indexOf('Attack the red')") == 17


def test_updates_are_text_only_and_follow_game_resize(
    client: tuple[PageProtocol, CDPSessionProtocol],
) -> None:
    """Repeated updates reuse one panel and follow the game's larger font."""
    page, cdp = client
    update_caption_panel(cdp, CaptionWordsDict(doing="First", why="First reason."))
    words = CaptionWordsDict(doing="<img src=x onerror=alert(1)>", why="Literal <b>text</b>.")
    update_caption_panel(cdp, words)
    page.evaluate("""() => {
        document.querySelector('.message-list').style.width = '300px';
        document.querySelector('select').style.fontSize = '16px';
    }""")
    page.wait_for_function(
        "getComputedStyle(document.getElementById('tankpit-bot-status')).fontSize === '16px'",
        timeout=5000,
    )
    assert page.evaluate("document.querySelectorAll('#tankpit-bot-status').length") == 1
    assert (
        page.evaluate(
            "document.querySelectorAll('#tankpit-bot-status img, #tankpit-bot-status b').length"
        )
        == 0
    )
    assert page.evaluate("document.getElementById('tankpit-bot-status').innerText") == (
        "<img src=x onerror=alert(1)>\nLiteral <b>text</b>."
    )


def test_missing_game_selector_fails_in_the_real_browser(
    client: tuple[PageProtocol, CDPSessionProtocol],
) -> None:
    """A game DOM change fails loudly instead of silently losing captions."""
    page, cdp = client
    page.set_content("<div>The client layout changed</div>")
    with pytest.raises(RuntimeError, match="CAPTION_PANEL_MISSING"):
        update_caption_panel(cdp, CaptionWordsDict(doing="Waiting", why="Waiting for a target."))


def test_dispatched_intent_and_death_reach_the_recorded_panel(
    client: tuple[PageProtocol, CDPSessionProtocol],
    tmp_path: Path,
) -> None:
    """The production tick publishers paint real browser pixels from bot state."""
    from tankpit_bot.bot.ai.intent import set_resource_target
    from tankpit_bot.bot.ai.scoring_types import BehaviorMode, ReasonKind, make_behavior_score
    from tankpit_bot.bot.ai.types import make_initial_ai_state
    from tankpit_bot.bot.base import Bot
    from tankpit_bot.bot.tick_body import _accept_dispatched_decision, _handle_own_deactivation
    from tankpit_bot.bot.tick_loop_types import make_tick_decision
    from tankpit_bot.bot.types import make_hold_command
    from tankpit_bot.state.types import make_self_state
    from tests.stream._capture_fixtures import stream_config

    page, cdp = client
    bot = Bot("https://test.tankpit.com/", stream_config=stream_config(tmp_path), headless=False)
    bot._cdp = cdp
    decision = make_tick_decision(
        command=make_hold_command(),
        behavior=make_behavior_score(
            BehaviorMode.COLLECT,
            100,
            4,
            5,
            ReasonKind.HARVEST_LEG_WALK,
        ),
        updated_ai_state=set_resource_target(make_initial_ai_state(), "equipment", 4, 5),
        desired_equipment=[],
    )
    _accept_dispatched_decision(bot, decision)
    assert page.evaluate(
        "document.getElementById('tankpit-bot-status').children[0].textContent"
    ) == ("Next: Collect equipment")
    _handle_own_deactivation(
        bot,
        make_self_state(
            tank_id=1,
            x=4,
            y=5,
            team=2,
            rank=0,
            fuel=0,
            leaderboard_position=1,
        ),
    )
    assert page.evaluate(
        "document.getElementById('tankpit-bot-status').children[0].textContent"
    ) == ("Destroyed")
    assert bot.world.world_state["self_state"] is None
    assert bot._respawn_deadline_ms > 0
