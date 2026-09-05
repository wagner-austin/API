"""Bot — assembles the session mixin chain into the playable bot.

Bot owns CDPService and CommandService via SessionBase composition.
State machine completions live in ``completions.py``, command dispatch
in ``bot_dispatch.py``, world-state queries in ``state_access.py``, and
the DOM game-log witness in ``game_log_witness.py``. This module keeps
construction and the session run loop.
"""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

from platform_core.logging import get_logger

from tankpit_bot import _test_hooks
from tankpit_bot._test_hooks import (
    CDPSessionProtocol,
    GamePageProtocol,
    PageProtocol,
)
from tankpit_bot.bot.ai.types import (
    AIStateDict,
    make_initial_ai_state,
)
from tankpit_bot.bot.bot_dispatch import DispatchMixin
from tankpit_bot.bot.command_service import CommandService
from tankpit_bot.bot.config import env_ai_config, resolve_hud_overlay
from tankpit_bot.bot.game_log_witness import GameLogWitnessMixin
from tankpit_bot.bot.state_access import StateAccessMixin
from tankpit_bot.bot.states import make_initial_state_data
from tankpit_bot.bot.tank_registry import record_tank_sample
from tankpit_bot.browser.cdp_service import CDPService
from tankpit_bot.browser.cdp_utils import get_current_time_ms
from tankpit_bot.browser.client_structure import ClientStructureSurveyor
from tankpit_bot.browser.dom_scraper import (
    GameLogScraper,
)
from tankpit_bot.browser.flag_capture import FlagCaptureService
from tankpit_bot.browser.fullscreen import enter_game_fullscreen
from tankpit_bot.browser.lifecycle import (
    cleanup_browser,
    gather_intel,
    navigate_and_login,
    wait_for_game_ready,
)
from tankpit_bot.browser.room_join import resolve_room_name
from tankpit_bot.browser.session_storage import (
    load_storage_state,
    resolve_storage_state_path,
    save_storage_state,
)
from tankpit_bot.bus.mode_bridge import ModeBridge, ModeBridgeProtocol
from tankpit_bot.bus.status_bus import StatusBus, StatusBusProtocol
from tankpit_bot.diagnostics.entity_alignment import EntityAlignmentEmitter
from tankpit_bot.diagnostics.self_alignment import SelfAlignmentEmitter
from tankpit_bot.runtime_logging import emit_state
from tankpit_bot.sniffer.chrome_launch import (
    _chrome_stream_display_args,
    _chrome_stream_no_viewport,
    _fullscreen_via_cdp,
    _maximize_via_cdp,
)
from tankpit_bot.sniffer.world_service import WorldService
from tankpit_bot.stream.capture import DisplayCapture
from tankpit_bot.stream.types import StreamConfigDict
from tankpit_bot.types import CapturedMessage, GameLogEntryWithTimestamp

log = get_logger(__name__)


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


class Bot(GameLogWitnessMixin, StateAccessMixin, DispatchMixin):
    """Bot that sends commands and tracks game state with a state machine.

    Inheritance chain: Bot → GameLogWitnessMixin → StateAccessMixin →
    DispatchMixin → CompletionsMixin → SessionBase. Each layer adds one
    focused concern: SessionBase owns CDPService/CommandService,
    CompletionsMixin adds state transitions, DispatchMixin adds command
    dispatch, StateAccessMixin answers world-state questions, and
    GameLogWitnessMixin polls the DOM log and reads account stats. Bot
    itself is left with construction and the session run loop.
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
        world: WorldService | None = None,
        stream_config: StreamConfigDict | None = None,
    ) -> None:
        """Initialize the bot.

        Args:
            target_url: URL to navigate to.
            headless: Whether to run browser in headless mode.
            prefer_account: Skip guest login and use account credentials.
            cdp_service: Injected CDPService. Created internally if None.
            command_service: Injected CommandService. Created internally if None.
            world: Injected WorldService. Created internally if None.
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
            stream_config: Display-capture parameters. ``None`` means
                no capture — right for a desktop ``make run`` (the
                operator watches the real window) and for replay /
                scenario sessions, which have no viewers at all. The
                fleet's containers pass the resolved configuration and
                the run then owns an Xvfb + ffmpeg pair for its whole
                lifetime.

        Raises:
            ValueError: ``stream_config`` with ``headless=True`` — a
                capture records a rendered window, and a headless
                launch has none to record.
        """
        if headless and stream_config is not None:
            raise ValueError(
                "stream capture records a rendered window; run headed (TANKPIT_HEADLESS off)"
                " when TANKPIT_STREAM_VIDEO is set"
            )
        super().__init__(
            target_url,
            headless=headless,
            prefer_account=prefer_account,
            cdp_service=cdp_service,
            command_service=command_service,
            world=world,
        )
        # Narrower than the full page: event-loop pumping for the
        # poll-and-read flows plus the keyboard the key probe presses.
        # Naming what is actually called is what lets the simulator BE
        # the page ([[session-state-deglobalisation]]).
        self._page: GamePageProtocol | None = None
        # One-shot latch for the in-game autoscroll enforcement; the
        # tick loop flips it on the first spawned tick.
        self._autoscroll_enforced: bool = False
        # Non-zero while waiting for the post-death respawn sync; the
        # tick loop exits ``deactivated`` if the deadline passes with
        # no fresh self state (a world with no respawn, e.g. the sim).
        self._respawn_deadline_ms: int = 0
        self._game_log_scraper: GameLogScraper | None = None
        self._game_log_witness: list[GameLogEntryWithTimestamp] = []
        # Kill-banner points-verdict witness state (GameLogWitnessMixin):
        # the victim's name renders one line before the deactivation
        # line, so the pairing needs the trailing line remembered.
        self._kill_banner_victim: str = ""
        self._last_game_log_line: str = ""
        self._shot_screenshot_seq: int = 0
        self._ai_state: AIStateDict = make_initial_ai_state(env_ai_config())
        default_mode_bridge: ModeBridgeProtocol = ModeBridge()
        default_status_bus: StatusBusProtocol = StatusBus()
        self._mode_bridge: ModeBridgeProtocol = (
            mode_bridge if mode_bridge is not None else default_mode_bridge
        )
        self._status_bus: StatusBusProtocol = (
            status_bus if status_bus is not None else default_status_bus
        )
        # Display capture is a RUN-scoped concern: the Xvfb + ffmpeg
        # pair live exactly as long as one browser does, so run() owns
        # their lifecycle and the constructor only keeps the wish.
        self._stream_config = stream_config
        # Read once, like env_ai_config above: whether the in-page
        # diagnostic HUD (and the flag-click binding that rides it)
        # renders this session. The demo fleet turns it off — the card
        # sat on top of the game in every captured frame.
        self._hud_overlay = resolve_hud_overlay()
        # Human bug marker (2026-07-29): the HUD flag button delivers
        # clicks over its own CDP binding; the service turns each into
        # a human_flag diagnostic carrying the recent-tick ring.
        self._flag_capture = FlagCaptureService()
        # Belief-vs-truth alignment sampling. Each emitter remembers what
        # it last wrote so identical ticks add no artifact bulk; that
        # memory is one session's, not the process's
        # ([[session-state-deglobalisation]] step 3).
        self._self_alignment = SelfAlignmentEmitter()
        self._entity_alignment = EntityAlignmentEmitter()
        # The client-structure survey is written once per SESSION, so
        # its "already done" gate belongs to the session too
        # ([[session-state-deglobalisation]] step 5).
        self._client_structure = ClientStructureSurveyor()
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
        from tankpit_bot.browser.types import PlaywrightNotInstalledError

        if _test_hooks.sync_playwright is None:
            raise PlaywrightNotInstalledError("Playwright is not installed.")

        self._start_timestamp_ms = get_current_time_ms()
        self._messages = []
        self._ws_urls = {}
        self._magic = None
        self._state_data = make_initial_state_data()
        self._ai_state = make_initial_ai_state(env_ai_config())
        self._cdp_message_buffer = []

        # A streamed launch pins the device scale factor: the game
        # client's on-screen size comes entirely from DPR (fixed CSS
        # layout), and Xvfb's default of 1 shrank the picture into a
        # corner of the capture. The screen geometry in the stream
        # config was chosen for content AT this factor, so the two
        # travel together.
        launch_args = _chrome_stream_display_args() + (
            [f"--force-device-scale-factor={self._stream_config['scale']}"]
            if self._stream_config is not None
            else []
        )
        # Keyed by login identity so a fleet child selecting a
        # different account can never resume another account's session
        # (the 2026-08-13 arterial-as-artax incident).
        storage_cache = resolve_storage_state_path(self._prefer_account)
        storage_state_path = load_storage_state(storage_cache)
        with ExitStack() as stack:
            capture: DisplayCapture | None = None
            launch_env: dict[str, str] | None = None
            if self._stream_config is not None:
                capture = DisplayCapture(self._stream_config)
                # Registered BEFORE start_display: a display that
                # timed out waiting still has a server process to
                # stop. LIFO order means this runs after the
                # playwright context below has closed the browser, so
                # the display outlives its one client.
                stack.callback(capture.stop)
                capture.start_display()
                # Chromium finds its display the way every X client
                # does: DISPLAY in ITS environment. Playwright
                # replaces the child environment rather than merging,
                # so the overlay rides a full copy of the parent's.
                launch_env = dict(_test_hooks.child_environment())
                launch_env["DISPLAY"] = capture.display_env
            playwright = stack.enter_context(_test_hooks.sync_playwright())
            browser = playwright.chromium.launch(
                headless=self._headless,
                args=launch_args,
                env=launch_env,
            )
            # ONE teardown path, and it starts the statement after the
            # launch. The browser used to be closed at the tail of the
            # game loop's own ``finally``, which meant a failure during
            # bootstrap -- a login that never completes, a game that
            # never becomes ready, the storage-state write -- never
            # reached the teardown ladder. Exiting the
            # ``sync_playwright`` context kills the driver and the
            # browser with it, so no process outlived that path, but
            # the graceful close and its watchdogs were skipped.
            try:
                # No fixed viewport whenever the window itself is the
                # product: the Vibeshine streamed display, or a kiosk
                # window filling the capture screen. Playwright's
                # default 1280x720 viewport would clamp the content
                # inside either.
                context = (
                    browser.new_context(no_viewport=True, storage_state=storage_state_path)
                    if _chrome_stream_no_viewport() or self._stream_config is not None
                    else browser.new_context(storage_state=storage_state_path)
                )
                page = context.new_page()
                cdp = context.new_cdp_session(page)
                if _chrome_stream_no_viewport():
                    _maximize_via_cdp(cdp)

                # Fullscreen, not maximised, when the window IS the
                # picture: ffmpeg records the whole Xvfb screen, and a
                # maximised window would put the tab strip and address
                # bar in every frame of the public stream. Post-launch
                # CDP because ``--kiosk`` shapes Chromium's own startup
                # window, which Playwright suppresses (observed live
                # 2026-09-05: the flag applied to nothing).
                if capture is not None:
                    _fullscreen_via_cdp(cdp)

                self._cdp = cdp
                self._page = page

                self._setup_console_listener(cdp)
                self._setup_cdp_handlers(cdp)

                navigate_and_login(
                    page,
                    cdp,
                    self.world,
                    target_url=self._target_url,
                    prefer_account=self._prefer_account,
                    tank_name_prefix="Bot",
                    auto_join_room=True,
                )

                wait_for_game_ready(page, self._messages)

                # The encoder starts only once there is a game on the
                # screen: everything before this line is login flow,
                # and recording it would spend the live window's first
                # segments on a loading page. The game's own
                # fullscreen goes first, so the first segment is
                # already the game filling the frame rather than a
                # centred box in dead margin.
                if capture is not None:
                    enter_game_fullscreen(cdp)
                    capture.start_encoder()

                # Persist the freshly-issued auth cookies + localStorage
                # before the game loop can crash. Next launch of the bot
                # skips the tankpit login flow entirely and rejoins in
                # seconds instead of the ~5-10 s cold navigate + credential
                # sequence.
                save_storage_state(context, storage_cache)

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
                    # Bookkeeping before the socket drops. The panel sample
                    # this files was taken at STARTUP -- it is a login-time
                    # snapshot that does not move mid-session -- but the
                    # tank's name and colour only arrive on the wire after
                    # the first ticks, so the WRITE waits for identity even
                    # though the READING did not. No keypress, no tick.
                    #
                    # Inside the browser's try, not beside it: these are
                    # about a session that actually started, and a
                    # bootstrap that failed has no tank to sample and no
                    # capture worth saving.
                    record_tank_sample(self.world, resolve_room_name())
                    self._send_graceful_quit()
                    self._save_capture_session()
                    self._detach_cdp_session()
            finally:
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
