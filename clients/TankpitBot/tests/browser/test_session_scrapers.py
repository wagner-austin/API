"""Tests for BrowserSession scraper functionality."""

from __future__ import annotations

from tankpit_bot.browser import BrowserSession
from tankpit_bot.combat import CombatEvent


def test_browser_session_init_game_log_scraper() -> None:
    """Test _init_game_log_scraper creates scraper and can scrape."""
    from tests.test_dom_scraper import FakeCDPForScraper

    session = BrowserSession("https://example.com")
    cdp = FakeCDPForScraper("LOCATION: 1,2")

    session._init_game_log_scraper(cdp)

    # Poll returns entries, proving scraper was created and works
    entries = session._poll_game_log()
    assert len(entries) == 1
    assert entries[0]["text"] == "LOCATION: 1,2"
    assert entries[0]["category"] == "location"


def test_browser_session_poll_game_log_no_scraper() -> None:
    """Test _poll_game_log returns empty list when scraper not initialized."""
    session = BrowserSession("https://example.com")
    result = session._poll_game_log()
    assert result == []


def test_browser_session_poll_game_log_with_entries() -> None:
    """Test _poll_game_log returns new entries and logs them."""
    from tests.test_dom_scraper import FakeCDPForScraper

    session = BrowserSession("https://example.com")
    cdp = FakeCDPForScraper("LOCATION: 10,20\nYou hit red-1")

    session._init_game_log_scraper(cdp)

    # First poll should return 2 entries
    entries = session._poll_game_log()
    assert len(entries) == 2
    assert entries[0]["text"] == "LOCATION: 10,20"
    assert entries[1]["text"] == "You hit red-1"

    # Second poll should return empty (same content)
    entries = session._poll_game_log()
    assert len(entries) == 0


def test_browser_session_init_inventory_scraper() -> None:
    """Test _init_inventory_scraper creates scraper and can scrape."""
    from tests.test_dom_scraper import FakeCDPForScraper

    session = BrowserSession("https://example.com")
    cdp = FakeCDPForScraper("30 dual shots\n10 extra radars")

    session._init_inventory_scraper(cdp)

    # Poll returns empty on first call (initializes state)
    changes = session._poll_inventory()
    assert len(changes) == 0


def test_browser_session_poll_inventory_no_scraper() -> None:
    """Test _poll_inventory returns empty list when scraper not initialized."""
    session = BrowserSession("https://example.com")
    result = session._poll_inventory()
    assert result == []


def test_browser_session_poll_inventory_with_changes() -> None:
    """Test _poll_inventory returns changes and logs them."""
    from tests.test_dom_scraper import FakeCDPForScraper

    session = BrowserSession("https://example.com")
    cdp = FakeCDPForScraper("30 dual shots\n10 extra radars")

    session._init_inventory_scraper(cdp)

    # First poll initializes state
    session._poll_inventory()

    # Update fake with changed inventory
    cdp._return_value = "37 dual shots\n10 extra radars"

    # Second poll should return 1 change
    changes = session._poll_inventory()
    assert len(changes) == 1
    assert changes[0]["item"] == "dual_shots"
    assert changes[0]["delta"] == 7


def test_browser_session_init_combat_tracker() -> None:
    """Test _init_combat_tracker initializes tracker."""
    session = BrowserSession("https://example.com")
    # Before init, get_combat_events returns empty
    assert session._get_combat_events() == []

    session._init_combat_tracker()

    # After init, tracker can process events
    assert session._get_combat_events() == []
    # Verify it works by processing a line
    if session._combat_tracker:
        session._combat_tracker.process_log_line("You hit blue-7")
        assert len(session._get_combat_events()) == 1


def test_browser_session_get_combat_events_no_tracker() -> None:
    """Test _get_combat_events returns empty list when tracker not initialized."""
    session = BrowserSession("https://example.com")
    result = session._get_combat_events()
    assert result == []


def test_browser_session_get_combat_events_with_tracker() -> None:
    """Test _get_combat_events returns events from tracker."""
    session = BrowserSession("https://example.com")
    session._init_combat_tracker()

    # Process a combat line via the tracker (if initialized)
    if session._combat_tracker:
        session._combat_tracker.process_log_line("You hit blue-7")

    events = session._get_combat_events()
    assert len(events) == 1
    assert events[0] == CombatEvent(event_type="hit_by_player", attacker="player", target="blue-7")


def test_browser_session_poll_game_log_processes_combat() -> None:
    """Test _poll_game_log processes combat events when tracker initialized."""
    from tests.test_dom_scraper import FakeCDPForScraper

    session = BrowserSession("https://example.com")
    cdp = FakeCDPForScraper("You hit blue-7\nblue-7 hit you")

    session._init_game_log_scraper(cdp)
    session._init_combat_tracker()

    # Poll should process combat events
    entries = session._poll_game_log()
    assert len(entries) == 2

    # Combat tracker should have recorded events
    events = session._get_combat_events()
    assert len(events) == 2
    assert events[0] == CombatEvent(event_type="hit_by_player", attacker="player", target="blue-7")
    assert events[1] == CombatEvent(event_type="hit_by_enemy", attacker="blue-7", target="player")


def test_browser_session_poll_game_log_no_combat_without_tracker() -> None:
    """Test _poll_game_log skips combat processing when no tracker."""
    from tests.test_dom_scraper import FakeCDPForScraper

    session = BrowserSession("https://example.com")
    cdp = FakeCDPForScraper("You hit blue-7")

    session._init_game_log_scraper(cdp)
    # Do not init combat tracker

    # Poll should still return entries
    entries = session._poll_game_log()
    assert len(entries) == 1

    # No crash, no events
    assert session._get_combat_events() == []


def test_browser_session_poll_game_log_combat_event_not_none() -> None:
    """Test _poll_game_log calls log_event when combat event is not None."""
    from tankpit_bot.combat import CombatStats
    from tests.test_dom_scraper import FakeCDPForScraper

    session = BrowserSession("https://example.com")
    # Use combat lines that will parse successfully
    cdp = FakeCDPForScraper("You hit blue-7\nYou hit red-5\nYou hit green-3")

    session._init_game_log_scraper(cdp)
    session._init_combat_tracker()

    # Poll processes combat events - all 3 should be captured
    entries = session._poll_game_log()
    assert len(entries) == 3

    # Verify all combat events were recorded (confirming log_event path was taken)
    events = session._get_combat_events()
    assert len(events) == 3
    assert events[0] == CombatEvent(event_type="hit_by_player", attacker="player", target="blue-7")
    assert events[1] == CombatEvent(event_type="hit_by_player", attacker="player", target="red-5")
    assert events[2] == CombatEvent(event_type="hit_by_player", attacker="player", target="green-3")

    # Verify combat tracker stats are correct
    if session._combat_tracker:
        blue_stats = session._combat_tracker.get_stats("blue-7")
        expected = CombatStats(
            name="blue-7", hits_given=1, hits_received=0, deactivated=False, destroyed=False
        )
        assert blue_stats == expected


def test_browser_session_poll_game_log_combat_category_but_no_parse() -> None:
    """Test _poll_game_log handles combat-categorized but non-parseable lines."""
    from tests.test_dom_scraper import FakeCDPForScraper

    session = BrowserSession("https://example.com")
    # "You earned 10 points" contains "earned" which triggers combat category
    # but doesn't match any combat parsing patterns
    cdp = FakeCDPForScraper("You earned 10 points for hitting something")

    session._init_game_log_scraper(cdp)
    session._init_combat_tracker()

    # Poll should process the entry but not create combat events
    entries = session._poll_game_log()
    assert len(entries) == 1
    assert entries[0]["category"] == "combat"

    # No combat events should be created (parse_combat_line returns None)
    events = session._get_combat_events()
    assert len(events) == 0
