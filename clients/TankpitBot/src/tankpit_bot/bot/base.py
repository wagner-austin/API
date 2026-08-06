"""Bot — assembles SessionBase + CompletionsMixin + DispatchMixin.

Bot owns CDPService and CommandService via SessionBase composition.
State machine completions live in ``completions.py``, command dispatch
in ``bot_dispatch.py``. This module adds init, state access, game log,
account stats, and the run loop.
"""

from __future__ import annotations

from pathlib import Path

from platform_core.logging import get_logger

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import CDPSessionProtocol, PageProtocol
from tankpit_bot.bot.ai.types import (
    AIConfigDict,
    AIStateDict,
    make_default_ai_config,
    make_initial_ai_state,
)
from tankpit_bot.bot.bot_dispatch import DispatchMixin
from tankpit_bot.bot.command_service import CommandService
from tankpit_bot.bot.states import (
    make_initial_state_data,
)
from tankpit_bot.browser.cdp_service import CDPService
from tankpit_bot.browser.cdp_utils import get_current_time_ms
from tankpit_bot.browser.dom_scraper import (
    GameLogEntry,
    GameLogScraper,
    scrape_page_text,
)
from tankpit_bot.browser.flag_capture import FlagCaptureService
from tankpit_bot.browser.lifecycle import (
    cleanup_browser,
    gather_intel,
    navigate_and_login,
    wait_for_game_ready,
)
from tankpit_bot.browser.live_view import LiveViewService
from tankpit_bot.browser.session_storage import (
    STORAGE_STATE_PATH,
    load_storage_state,
    save_storage_state,
)
from tankpit_bot.diagnostics.account_stats import (
    emit_account_stats_sample,
    parse_account_stats,
)
from tankpit_bot.runtime_logging import emit_state
from tankpit_bot.service.frame_bus import FrameBus, FrameBusProtocol
from tankpit_bot.service.mode_bridge import ModeBridge, ModeBridgeProtocol
from tankpit_bot.service.status_bus import StatusBus, StatusBusProtocol
from tankpit_bot.sniffer.core import (
    _chrome_stream_display_args,
    _chrome_stream_no_viewport,
    _maximize_via_cdp,
)
from tankpit_bot.sniffer.world_state import get_world_state
from tankpit_bot.state import ContainerStateDict, SelfStateDict, WorldStateDict
from tankpit_bot.types import CapturedMessage, GameLogEntryWithTimestamp

log = get_logger(__name__)

# The C statistics panel paints incrementally: the "Statistics:" header
# can be in the DOM before the stat lines (a single 1500ms timed read
# landed in that gap and crashed sessions 20260611-004251/004405/012807).
# Poll the parse predicate instead of trusting one timed read.
_ACCOUNT_STATS_POLL_INTERVAL_MS = 300
_ACCOUNT_STATS_POLL_ATTEMPTS = 10
# Total wait budget for a single timed panel read (used by the simple
# capture path; equals one full poll budget).
_ACCOUNT_STATS_PANEL_RENDER_MS = _ACCOUNT_STATS_POLL_INTERVAL_MS * _ACCOUNT_STATS_POLL_ATTEMPTS
# The first-tick keypress itself can be swallowed by the client (run
# 20260611-013801: panel never opened across a full poll budget), so
# the startup capture retries on later ticks.
_ACCOUNT_STATS_MAX_CAPTURE_ATTEMPTS = 3


def _env_ai_config() -> AIConfigDict:
    """Build the session AI config with environment overrides applied.

    Overrides: ``TANKPIT_BOT_PRIORITY_TARGET`` (the priority hunt
    account) and ``TANKPIT_BOT_HUMAN_MIN_RANK`` /
    ``TANKPIT_BOT_HUMAN_MAX_RANK`` (the targetable human rank window)
    -- [[bot-behavior-contract]] §3.2.

    Returns:
        Default AI config with env-resolved fields filled in.
    """
    from tankpit_bot.bot.config import resolve_human_rank_window, resolve_priority_target

    min_rank, max_rank = resolve_human_rank_window()
    return AIConfigDict(
        **{
            **make_default_ai_config(),
            "priority_target_name": resolve_priority_target(),
            "human_target_min_rank": min_rank,
            "human_target_max_rank": max_rank,
        }
    )


class BotError(Exception):
    """Base exception for bot errors.

    All bot-specific exceptions inherit from this class,
    allowing callers to catch all bot errors with a single except clause.
    """


class ProtocolNotDiscoveredError(BotError):
    """Raised when the XOR protocol keys have not been discovered.

    The bot requires the magic key and static key to be discovered
    before it can send commands. This error is raised when attempting
    to send commands without valid protocol keys.
    """


class Bot(DispatchMixin):
    """Bot that sends commands and tracks game state with a state machine.

    Inheritance chain: Bot → DispatchMixin → CompletionsMixin → SessionBase.
    Each layer adds a focused concern: SessionBase owns CDPService/CommandService,
    CompletionsMixin adds state transitions, DispatchMixin adds command dispatch.
    Bot adds init, state access, game log, account stats, and the run loop.
    """

    def __init__(
        self,
        target_url: str,
        *,
        headless: bool = False,
        prefer_account: bool = False,
        cdp_service: CDPService | None = None,
        command_service: CommandService | None = None,
        mode_bridge: ModeBridgeProtocol | None = None,
        status_bus: StatusBusProtocol | None = None,
        frame_bus: FrameBusProtocol | None = None,
    ) -> None:
        """Initialize the bot.

        Args:
            target_url: URL to navigate to.
            headless: Whether to run browser in headless mode.
            prefer_account: Skip guest login and use account credentials.
            cdp_service: Injected CDPService. Created internally if None.
            command_service: Injected CommandService. Created internally if None.
            mode_bridge: Cross-thread channel the SPA writes mode
                overrides into. When ``None``, a fresh :class:`ModeBridge`
                is created — a standalone ``make bot`` session gets an
                inert bridge that no HTTP handler ever writes to, so
                ``drain`` returns ``None`` every tick and
                auto-arbitration runs unchanged. The service main
                (Phase A8) injects the shared instance the aiohttp
                thread owns.
            status_bus: Fan-out the tick loop publishes
                :class:`SessionStatusDict` frames into. When ``None``,
                a fresh :class:`StatusBus` is created — a standalone
                session gets a bus with zero subscribers, so publish
                is a no-op.
            frame_bus: Fan-out the screencast relay publishes JPEG
                frames into (2026-07-28 watch page). When ``None``, a
                fresh :class:`FrameBus` is created — a standalone
                session has zero subscribers, so the tick loop's
                demand check never starts the screencast.
        """
        super().__init__(
            target_url,
            headless=headless,
            prefer_account=prefer_account,
            cdp_service=cdp_service,
            command_service=command_service,
        )
        self._page: PageProtocol | None = None
        # One-shot latch for the in-game autoscroll enforcement; the
        # tick loop flips it on the first spawned tick.
        self._autoscroll_enforced: bool = False
        # Non-zero while waiting for the post-death respawn sync; the
        # tick loop exits ``deactivated`` if the deadline passes with
        # no fresh self state (a world with no respawn, e.g. the sim).
        self._respawn_deadline_ms: int = 0
        self._game_log_scraper: GameLogScraper | None = None
        self._game_log_witness: list[GameLogEntryWithTimestamp] = []
        self._shot_screenshot_seq: int = 0
        self._ai_state: AIStateDict = make_initial_ai_state(_env_ai_config())
        default_mode_bridge: ModeBridgeProtocol = ModeBridge()
        default_status_bus: StatusBusProtocol = StatusBus()
        self._mode_bridge: ModeBridgeProtocol = (
            mode_bridge if mode_bridge is not None else default_mode_bridge
        )
        self._status_bus: StatusBusProtocol = (
            status_bus if status_bus is not None else default_status_bus
        )
        default_frame_bus: FrameBusProtocol = FrameBus()
        self._frame_bus: FrameBusProtocol = (
            frame_bus if frame_bus is not None else default_frame_bus
        )
        # The in-page caster captures at its own cadence and delivers
        # frames over the CDP binding channel (Chrome's Local Network
        # Access gate hangs page→loopback fetches forever, 2026-07-29);
        # the binding handler publishes onto the shared frame bus.
        self._live_view = LiveViewService(publish=self._frame_bus.publish)
        # Human bug marker (2026-07-29): the HUD flag button delivers
        # clicks over its own CDP binding; the service turns each into
        # a human_flag diagnostic carrying the recent-tick ring.
        self._flag_capture = FlagCaptureService()
        # Gate for the C-panel account stats capture; fired from the
        # first HEALTHY tick rather than at bootstrap because the game
        # client ignores hotkeys until fully loaded (run 20260611-000x
        # captured panel_visible=False at startup). Failed attempts
        # retry on later ticks (bounded) since even a healthy-tick
        # keypress can be swallowed (run 20260611-013801).
        self._account_stats_captured = False
        self._account_stats_attempts = 0

    def _on_message_captured(self, message: CapturedMessage) -> None:
        """Buffer received messages with debug logging.

        Args:
            message: The captured message.
        """
        super()._on_message_captured(message)
        if message["direction"] == "received":
            log.debug("CDP_BUFFER: +1 (total=%d)", len(self._cdp_message_buffer))

    def _require_cdp(self) -> CDPSessionProtocol:
        """Return the attached CDP session or raise.

        Used by tick-loop code that must read the live page-client state.
        The tick loop's readiness gates ensure ``_cdp`` is attached well
        before any code reaches the capture point, so a missing session
        is an invariant violation rather than a normal pre-bootstrap
        state.
        """
        if self._cdp is None:
            raise RuntimeError("Bot has no CDP session attached")
        return self._cdp

    # =========================================================================
    # State Access
    # =========================================================================

    def get_world_state(self) -> WorldStateDict:
        """Get current world state.

        Returns:
            Current WorldStateDict with all tracked entities.
        """
        return get_world_state()

    def get_self_state(self) -> SelfStateDict | None:
        """Get self tank state (position, fuel, etc.).

        Returns:
            SelfStateDict if available, None if not yet tracked.
        """
        return get_world_state()["self_state"]

    def get_fuel(self) -> int:
        """Get current fuel (HP).

        Returns:
            Current fuel amount, or 0 if self_state not yet tracked.
        """
        state = self.get_self_state()
        return state["fuel"] if state is not None else 0

    def get_position(self) -> tuple[int, int] | None:
        """Get current position.

        Returns:
            Tuple of (x, y) coordinates, or None if not yet tracked.
        """
        state = self.get_self_state()
        if state is None:
            return None
        return (state["x"], state["y"])

    def get_containers(self) -> dict[str, ContainerStateDict]:
        """Get all known containers.

        Returns:
            Dict of container key ("x,y") to ContainerStateDict.
        """
        return get_world_state()["containers"]

    def get_fuel_containers(self) -> list[ContainerStateDict]:
        """Get all known fuel containers (not equipment).

        Returns:
            List of fuel containers with volume > 0.
        """
        containers = self.get_containers()
        return [c for c in containers.values() if c["is_fuel"] and c["volume"] > 0]

    def get_nearest_fuel_container(self) -> ContainerStateDict | None:
        """Get nearest fuel container to current position.

        Returns:
            Nearest ContainerStateDict, or None if no containers or no position.
        """
        pos = self.get_position()
        if pos is None:
            return None

        fuel_containers = self.get_fuel_containers()
        if not fuel_containers:
            return None

        # Sort by Manhattan distance
        my_x, my_y = pos
        fuel_containers.sort(key=lambda c: abs(c["x"] - my_x) + abs(c["y"] - my_y))
        return fuel_containers[0]

    # =========================================================================
    # Game Log
    # =========================================================================
    #
    # The bot inherits ``SessionBase`` -- a parallel hierarchy from the
    # ``BrowserSession`` used by the standalone sniffer -- so it owns
    # its own game-log scraper hooks. The DOM log is a WITNESS, not an
    # actor: every line it renders is the client's presentation of a
    # wire message the bot already decodes (0x41 Deactivation for
    # kills, 0x52 error codes for rejections -- capture replay
    # 2026-07-19, see wiki [[deactivation-format]]). The tick loop
    # polls it each tick and records the entries into the capture
    # artifact so the analyzer can diff the client's rendering against
    # the wire; nothing in the bot acts on them.

    def _init_game_log_scraper(self, cdp: CDPSessionProtocol) -> None:
        """Create the game log scraper for server feedback visibility.

        Args:
            cdp: Active CDP session for DOM access.
        """
        self._game_log_scraper = GameLogScraper(cdp)

    def _poll_game_log(self) -> list[GameLogEntry]:
        """Poll the game log for new entries since the last scrape.

        Returns:
            New log entries (kills, hits, empty containers, etc.).
        """
        scraper = self._game_log_scraper
        if scraper is None:
            return []
        return scraper.get_new_entries()

    def _record_game_log_witness(self, entries: list[GameLogEntry]) -> None:
        """Timestamp new game-log entries into the capture witness list.

        Args:
            entries: New log entries from this tick's poll, in order.
        """
        now = get_current_time_ms()
        for entry in entries:
            self._game_log_witness.append(
                GameLogEntryWithTimestamp(
                    timestamp_ms=now,
                    text=entry["text"],
                    category=entry["category"],
                )
            )

    def _capture_account_stats(self, phase: str) -> None:
        """Sample the in-game ``C`` statistics panel and emit it.

        The panel carries account-wide ground truth the wire never
        sends (lifetime play time, kills, deactivations, promotion
        points); the startup sample baselines every run so consecutive
        runs' deltas verify the wire 0x41 kill detection. The ``C`` key
        does not toggle a stateful panel -- each keypress emits a
        fresh ``Statistics:`` block into the in-game DOM log -- so a
        single press is enough to scrape, and a second press would
        only duplicate the block in the log without ``closing``
        anything.

        Args:
            phase: Capture point label (e.g. ``startup``).
        """
        if self._cdp is None or self._page is None:
            return
        for event_type in ("keyDown", "keyUp"):
            self._cdp.send(
                "Input.dispatchKeyEvent",
                {
                    "type": event_type,
                    "key": "c",
                    "code": "KeyC",
                    "windowsVirtualKeyCode": ord("C"),
                    "nativeVirtualKeyCode": ord("C"),
                },
            )
        self._page.wait_for_timeout(_ACCOUNT_STATS_PANEL_RENDER_MS)
        page_text = scrape_page_text(self._cdp)
        emit_account_stats_sample(parse_account_stats(page_text), phase=phase)

    def maybe_capture_account_stats_once(self) -> None:
        """Capture account stats on the first healthy tick, with bounded retries.

        The C-panel hotkey can be swallowed by the game client (run
        20260611-013801), so failed attempts retry on later ticks up to
        a bounded maximum.
        """
        if self._account_stats_captured:
            return
        if self._account_stats_attempts >= _ACCOUNT_STATS_MAX_CAPTURE_ATTEMPTS:
            return
        self._account_stats_attempts += 1
        self._capture_account_stats("startup")
        self._account_stats_captured = True

    # =========================================================================
    # Run Loop
    # =========================================================================

    def run(
        self,
        *,
        session_seconds: int,
        session_kills: int = 0,
        stop_file_path: Path,
    ) -> None:
        """Run the bot.

        Launches browser, logs in, joins game, and runs the game loop.
        The bot will scan for fuel and collect it automatically. The
        run ends gracefully -- capture saved, browser closed -- when
        the tick budget elapses or the stop file appears.

        Args:
            session_seconds: Bounded session length in seconds; zero
                or negative runs until externally stopped.
            stop_file_path: Sentinel file whose existence requests a
                graceful shutdown.

        Raises:
            RuntimeError: If Playwright is not installed.
        """
        from tankpit_bot.browser.cdp_utils import reset_cdp_time_offset
        from tankpit_bot.browser.types import PlaywrightNotInstalledError
        from tankpit_bot.sniffer.viewport import reset_viewport_tracking

        if _test_hooks.sync_playwright is None:
            raise PlaywrightNotInstalledError("Playwright is not installed.")

        self._start_timestamp_ms = get_current_time_ms()
        self._messages = []
        self._ws_urls = {}
        self._magic = None
        self._state_data = make_initial_state_data()
        self._ai_state = make_initial_ai_state(_env_ai_config())
        self._cdp_message_buffer = []

        launch_args = _chrome_stream_display_args()
        storage_state_path = load_storage_state(STORAGE_STATE_PATH)
        with _test_hooks.sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=self._headless,
                args=launch_args,
            )
            context = (
                browser.new_context(no_viewport=True, storage_state=storage_state_path)
                if _chrome_stream_no_viewport()
                else browser.new_context(storage_state=storage_state_path)
            )
            page = context.new_page()
            cdp = context.new_cdp_session(page)
            if _chrome_stream_no_viewport():
                _maximize_via_cdp(cdp)

            self._cdp = cdp
            self._page = page

            reset_cdp_time_offset()
            reset_viewport_tracking()

            self._setup_console_listener(cdp)
            self._setup_cdp_handlers(cdp)

            navigate_and_login(
                page,
                cdp,
                target_url=self._target_url,
                prefer_account=self._prefer_account,
                tank_name_prefix="Bot",
                auto_join_room=True,
            )

            wait_for_game_ready(page, self._messages)

            # Persist the freshly-issued auth cookies + localStorage
            # before the game loop can crash. Next launch of the bot
            # skips the tankpit login flow entirely and rejoins in
            # seconds instead of the ~5-10 s cold navigate + credential
            # sequence.
            save_storage_state(context, STORAGE_STATE_PATH)

            self._cdp_service.log_websocket_urls()
            self._static_key = gather_intel(page, cdp)

            self._init_game_log_scraper(cdp)

            log.info("Bot started, entering game loop")
            emit_state("%s", self.get_state())

            try:
                self._game_loop(
                    page,
                    session_seconds=session_seconds,
                    session_kills=session_kills,
                    stop_file_path=stop_file_path,
                )
            except KeyboardInterrupt:
                log.info("Bot interrupted by user")
            finally:
                self._send_graceful_quit()
                self._save_capture_session()
                self._detach_cdp_session()
                self._cdp = None
                self._page = None
                cleanup_browser(browser)

    def _send_graceful_quit(self) -> None:
        """Send the plain quit command so the server sees a lobby exit.

        Runs first in teardown, while the CDP session is still bound.
        A CLOSED browser makes the courtesy send definitionally
        pointless -- the socket drop already told the server we left
        -- so ``TargetClosedError`` is absorbed here with a log line.
        (The old docstring CLAIMED the send path handled a gone
        session; run bot-20260729-215151 proved it false: the browser
        died at 19 kills, the scorecard wrote cleanly, and then this
        send raised through teardown and turned a handled shutdown
        into exit code 2.)
        """
        from playwright._impl._errors import TargetClosedError

        self._commands.cdp = self._cdp
        try:
            self._commands.quit_game()
        except TargetClosedError:
            log.info("Graceful quit skipped: browser already closed (socket drop = lobby exit)")

    def _detach_cdp_session(self) -> None:
        """Detach the CDP session so no events race the browser close.

        The session's frame listeners stay registered until an explicit
        detach; after ``quit_game`` drops the game socket, late CDP
        events dispatched during ``browser.close()`` hit Playwright's
        sync bridge on a closing connection and log an ERROR-level
        callback traceback (seen at the end of the 20-kill soak
        bot-20260802-205105, zero bot frames in the stack). Detaching
        first unsubscribes the session; a target that is ALREADY gone
        makes the detach pointless, so its close-race error is
        absorbed with a log line — the same discipline as
        :meth:`_send_graceful_quit`.
        """
        from playwright._impl._errors import Error as PlaywrightError

        if self._cdp is None:
            return
        try:
            self._cdp.detach()
        except PlaywrightError as exc:
            log.info("CDP detach skipped: session already gone (%s)", exc)

    def _save_capture_session(self) -> None:
        """Save accumulated messages as a replayable capture session.

        Writes the capture session to the canonical bot artifact paths
        (latest + archive) so ``replay_bot.py`` can replay the run offline.
        """
        from platform_core.json_utils import dump_json_str

        from tankpit_bot.runtime_logging import get_bot_runtime_artifacts
        from tankpit_bot.types import CaptureSession, encode_capture_session

        artifacts = get_bot_runtime_artifacts()
        if artifacts is None:
            return

        session = CaptureSession(
            session_id=self._session_id,
            start_timestamp_ms=self._start_timestamp_ms,
            end_timestamp_ms=get_current_time_ms(),
            base_url=self._target_url,
            messages=self._messages,
            magic=self._magic,
            game_log=list(self._game_log_witness),
            tank_names={},
        )
        encoded = encode_capture_session(session)
        json_str = dump_json_str(encoded, compact=False, indent=2)
        _test_hooks.write_text(
            Path(artifacts["latest_capture_path"]),
            json_str,
        )
        _test_hooks.write_text(
            Path(artifacts["archive_capture_path"]),
            json_str,
        )
        log.info(
            "Saved capture session: %d messages -> %s",
            len(self._messages),
            artifacts["latest_capture_path"],
        )

    def _game_loop(
        self,
        page: PageProtocol,
        *,
        session_seconds: int,
        session_kills: int = 0,
        stop_file_path: Path,
    ) -> None:
        """Run the tick loop: sync, decide, execute on each server tick.

        Args:
            page: Playwright page for waiting between ticks.
            session_seconds: Bounded session length in seconds; zero
                or negative runs until externally stopped.
            stop_file_path: Sentinel file whose existence requests a
                graceful shutdown.
        """
        from tankpit_bot.bot.tick_loop import run_tick_loop

        run_tick_loop(
            self,
            page,
            session_seconds=session_seconds,
            session_kills=session_kills,
            stop_file_path=stop_file_path,
        )


__all__ = [
    "Bot",
    "BotError",
    "ProtocolNotDiscoveredError",
]
