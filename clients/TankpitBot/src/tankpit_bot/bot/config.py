"""Bot-launch configuration resolved from the process environment.

Settings that launch a :class:`Bot` need to be read the same way from
every code path that launches one. Centralising the resolvers here
keeps :mod:`tankpit_bot.bot.entry` (the one-shot ``tankpit-bot`` CLI)
and :mod:`tankpit_bot.service.service_main` (what every fleet child
runs) in lockstep — an env-var-handling difference between the two used
to be a silent divergence risk, and ``headless`` proved it by sitting
as a hardcoded literal in BOTH until 2026-09-02.

Both resolvers read through :func:`tankpit_bot._test_hooks.get_env`, so
tests inject deterministic values without touching the real process
environment.
"""

from __future__ import annotations

from tankpit_bot import _test_hooks
from tankpit_bot.bot.ai.types import AIConfigDict, make_default_ai_config
from tankpit_bot.fleetshare.role import resolve_engagement_doctrine, resolve_fleet_role
from tankpit_bot.runtime_artifacts import bot_run_dir, resolve_bot_instance
from tankpit_bot.stream.types import StreamConfigDict, decode_stream_config

DEFAULT_TARGET_URL = "https://tankpit.com/"
"""Canonical tankpit URL used when ``TANKPIT_URL`` is unset or empty."""

_TRUTHY_VALUES: tuple[str, ...] = ("true", "1", "yes")
"""Accepted spellings of "yes" for every boolean env var read here.

Shared rather than duplicated per setting: two copies drift, and an
operator who learns that ``TANKPIT_PREFER_ACCOUNT=yes`` works is
entitled to have ``TANKPIT_HEADLESS=yes`` work the same way.
"""


def resolve_target_url() -> str:
    """Return the tankpit target URL, honouring the ``TANKPIT_URL`` env var.

    Returns:
        The env override when set to a non-empty value; otherwise the
        canonical :data:`DEFAULT_TARGET_URL`.
    """
    override = _test_hooks.get_env("TANKPIT_URL")
    if override is not None and override != "":
        return override
    return DEFAULT_TARGET_URL


def resolve_env_flag(name: str) -> bool:
    """Return True when the named environment variable spells an affirmative.

    One parser for every boolean setting in the process. The named
    resolvers below bind a variable to a meaning; this binds a spelling
    to a truth value, and there is exactly one of it so an operator who
    learns that ``yes`` works for one flag can rely on it for all of
    them.

    Args:
        name: Environment variable to read.

    Returns:
        True when the value (case-insensitive) is one of ``"true"``,
        ``"1"`` or ``"yes"``; False otherwise, including when unset.
    """
    raw = _test_hooks.get_env(name)
    if raw is None:
        return False
    return raw.lower() in _TRUTHY_VALUES


def resolve_prefer_account() -> bool:
    """Return True when ``TANKPIT_PREFER_ACCOUNT`` selects account login.

    Returns:
        True when the env var value (case-insensitive) matches any of
        ``"true"``, ``"1"``, or ``"yes"``; False otherwise (including
        when the env var is unset).
    """
    return resolve_env_flag("TANKPIT_PREFER_ACCOUNT")


def resolve_headless() -> bool:
    """Return True when ``TANKPIT_HEADLESS`` asks for a windowless browser.

    **Defaults to False, deliberately.** On a desktop the operator runs
    the fleet headed in order to watch the tanks play, and that is the
    point of running it there; a default of headless would take the
    window away from the machine whose whole job is showing it.

    A container is the case that needs the other answer. There is no X
    server in one, so a headed launch dies immediately with "Missing X
    server or $DISPLAY" and the child exits 1 about five seconds after
    spawn. ``docker-compose.yml`` has set ``TANKPIT_HEADLESS: "true"``
    all along; until this resolver existed nothing on the bot path read
    it, so the flag looked like the answer while being inert, and every
    containerized fleet bot died on launch.

    Returns:
        True when the env var value (case-insensitive) matches any of
        ``"true"``, ``"1"``, or ``"yes"``; False otherwise, including
        when the env var is unset.
    """
    return resolve_env_flag("TANKPIT_HEADLESS")


def resolve_human_rank_window() -> tuple[int, int]:
    """Return the targetable human rank window from the environment.

    ``TANKPIT_BOT_HUMAN_MIN_RANK`` / ``TANKPIT_BOT_HUMAN_MAX_RANK``
    bound which human ranks the bot may engage (integers, 0 recruit ..
    8 general). Defaults ``(1, 8)``: recruits protected, no ceiling.
    A main-map bot can raise the floor (lieutenant+ = 4) or lower the
    ceiling to leave high ranks alone (user doctrine 2026-07-28).
    Practice bots are farmed at any rank regardless of this window.

    Returns:
        ``(min_rank, max_rank)``.

    Raises:
        ValueError: If either value is not an integer.
    """
    raw_min = _test_hooks.get_env("TANKPIT_BOT_HUMAN_MIN_RANK")
    raw_max = _test_hooks.get_env("TANKPIT_BOT_HUMAN_MAX_RANK")
    return (
        int(raw_min) if raw_min is not None else 1,
        int(raw_max) if raw_max is not None else 8,
    )


DEFAULT_STREAM_SCALE = 2
"""Device scale factor for the streamed Chromium.

The game client lays out at a FIXED ~568x330 CSS pixels; its on-screen
size comes entirely from devicePixelRatio. The operator's desktop runs
~1.75 (which is where the caster era's 672x532 composite came from);
an Xvfb defaults to 1, and 1 is why the first live captures were a
small game floating in dead margin (measured 2026-09-05: content
568x330 centred in 704x544). Forcing 2 renders the client at crisp
double pixels, the same picture the desktop shows."""

DEFAULT_STREAM_WIDTH = 1280
"""Capture screen width. The client's layout at scale 2 measured
~1160 physical once live (a 1152 screen grew scrollbars — the
first-frame content-box measurement ran a few percent under the real
layout), so the default carries working margin rather than hugging a
number that moves with the client's own chrome."""

DEFAULT_STREAM_HEIGHT = 768
"""Capture screen height, same margin logic over the ~730 physical
the layout measured at scale 2."""

DEFAULT_STREAM_FPS = 30
"""Display sampling rate. The game paints at ~60 Hz but its motion is
readable at 30, and half the sampling is half the encode cost for a
per-bot pipeline that runs N times per container."""

DEFAULT_STREAM_BITRATE_KBPS = 1500
"""Encoder target. A megabit and a half of H.264 at 1152x672 is
generous for scaled 2D game content — and still ~4x less than the
678 KB/s the MJPEG pipeline this replaced was measured spending on a
quarter of the pixels."""

DEFAULT_STREAM_SEGMENT_SECONDS = 2
"""HLS segment length: the latency floor of the pipeline. Two seconds
keeps a viewer 2-6 s behind live, which a passive demo does not feel,
and keeps request cadence at one playlist poll + one segment per two
seconds per viewer."""


def resolve_hud_overlay() -> bool:
    """Return whether the in-page diagnostic HUD renders this session.

    ``TANKPIT_HUD_OVERLAY`` defaults ON — the HUD exists for a human
    watching the browser, and the desktop ``make run``/``make fleet``
    operator is exactly that human. The demo fleet's compose sets it
    off: its viewers are strangers watching the STREAM, the overlay
    card sits on top of the game in every captured frame (operator
    report, 2026-09-05), and the same numbers already reach the fleet
    page through the per-tick ``hud.json`` mirror, which this switch
    deliberately does not touch.

    Returns:
        False when the env var (case-insensitive) is set to anything
        other than ``"true"``, ``"1"`` or ``"yes"``; True otherwise,
        including when unset.
    """
    raw = _test_hooks.get_env("TANKPIT_HUD_OVERLAY")
    if raw is None:
        return True
    return raw.lower() in _TRUTHY_VALUES


def resolve_stream_config() -> StreamConfigDict | None:
    """Return the display-capture configuration, or ``None`` when off.

    ``TANKPIT_STREAM_VIDEO`` is the switch, set by the fleet's compose
    file and nothing else: a desktop ``make run`` has a real window to
    watch and no Xvfb to start, so unset means no capture and no other
    variable is read.

    The display number comes from ``TANKPIT_STREAM_DISPLAY`` when set,
    else from ``TANKPIT_BOT_SERVICE_PORT`` — the port the fleet
    manager already allocates uniquely per live child, which makes it
    a free unique X display number. No third source: a process with
    neither has no collision-free number to claim, and guessing one
    is how two bots end up recording each other's screens.

    Args:
        None.

    Returns:
        The validated configuration, or ``None`` when streaming is
        off.

    Raises:
        ValueError: Streaming is on but no display number is
            resolvable, or a numeric override is not an integer, or a
            value is outside its domain (via
            :func:`~tankpit_bot.stream.types.decode_stream_config`).
    """
    if not resolve_env_flag("TANKPIT_STREAM_VIDEO"):
        return None
    raw_display = _test_hooks.get_env("TANKPIT_STREAM_DISPLAY")
    if raw_display is None:
        raw_display = _test_hooks.get_env("TANKPIT_BOT_SERVICE_PORT")
    if raw_display is None:
        raise ValueError(
            "TANKPIT_STREAM_VIDEO is set but neither TANKPIT_STREAM_DISPLAY nor"
            " TANKPIT_BOT_SERVICE_PORT is; there is no unique display number to use"
        )
    raw_fps = _test_hooks.get_env("TANKPIT_STREAM_FPS")
    raw_bitrate = _test_hooks.get_env("TANKPIT_STREAM_BITRATE_KBPS")
    raw_scale = _test_hooks.get_env("TANKPIT_STREAM_SCALE")
    raw_width = _test_hooks.get_env("TANKPIT_STREAM_WIDTH")
    raw_height = _test_hooks.get_env("TANKPIT_STREAM_HEIGHT")
    hls_dir = bot_run_dir(resolve_bot_instance()) / "hls"
    return decode_stream_config(
        {
            "display": int(raw_display),
            "width": int(raw_width) if raw_width is not None else DEFAULT_STREAM_WIDTH,
            "height": int(raw_height) if raw_height is not None else DEFAULT_STREAM_HEIGHT,
            "scale": int(raw_scale) if raw_scale is not None else DEFAULT_STREAM_SCALE,
            "fps": int(raw_fps) if raw_fps is not None else DEFAULT_STREAM_FPS,
            "bitrate_kbps": (
                int(raw_bitrate) if raw_bitrate is not None else DEFAULT_STREAM_BITRATE_KBPS
            ),
            "segment_seconds": DEFAULT_STREAM_SEGMENT_SECONDS,
            "hls_dir": str(hls_dir),
        }
    )


def resolve_priority_target() -> str:
    """Return the priority hunt account from ``TANKPIT_BOT_PRIORITY_TARGET``.

    The named account outranks every other target at acquisition
    (case-insensitive; humans already outrank practice bots without
    any configuration -- [[bot-behavior-contract]] §3.2).

    Returns:
        The configured account name, or ``""`` when unset/empty.
    """
    raw = _test_hooks.get_env("TANKPIT_BOT_PRIORITY_TARGET")
    return raw if raw is not None else ""


__all__ = [
    "DEFAULT_STREAM_BITRATE_KBPS",
    "DEFAULT_STREAM_FPS",
    "DEFAULT_STREAM_HEIGHT",
    "DEFAULT_STREAM_SCALE",
    "DEFAULT_STREAM_SEGMENT_SECONDS",
    "DEFAULT_STREAM_WIDTH",
    "DEFAULT_TARGET_URL",
    "env_ai_config",
    "resolve_env_flag",
    "resolve_headless",
    "resolve_hud_overlay",
    "resolve_human_rank_window",
    "resolve_prefer_account",
    "resolve_priority_target",
    "resolve_stream_config",
    "resolve_target_url",
]


def resolve_weapon_resume_slack() -> int:
    """Return the weapons resume slack from ``TANKPIT_BOT_WEAPON_RESUME_SLACK``.

    Default 0 preserves the 2026-07-25 contract verbatim: HUNT entry
    requires duals and homings at exactly ``inventory_capacity(rank)``
    ("the bot never hunts below full stock, no exceptions"). Setting a
    positive slack N relaxes the resume bar to ``cap - N`` -- the same
    shape as the radar rule's fixed cap-5 floor -- because equipment
    has no map atlas, so every between-kill top-off to the exact cap
    forces a hop-scan discovery loop (the 2026-07-29 session where the
    user flagged that loop nine times from the HUD). The weapon
    EMERGENCY break (<4) is unaffected; this only moves the resume
    bar.

    Returns:
        Non-negative slack subtracted from the weapons resume cap.

    Raises:
        ValueError: If the env value is set but not a non-negative int.
    """
    raw = _test_hooks.get_env("TANKPIT_BOT_WEAPON_RESUME_SLACK")
    if raw is None:
        return 0
    value = int(raw)
    if value < 0:
        raise ValueError(f"TANKPIT_BOT_WEAPON_RESUME_SLACK must be >= 0, got {value}")
    return value


def env_ai_config() -> AIConfigDict:
    """Build the session AI config with environment overrides applied.

    Overrides: ``TANKPIT_BOT_PRIORITY_TARGET`` (the priority hunt
    account) and ``TANKPIT_BOT_HUMAN_MIN_RANK`` /
    ``TANKPIT_BOT_HUMAN_MAX_RANK`` (the targetable human rank window)
    -- [[bot-behavior-contract]] §3.2.

    Lives here rather than in ``bot/base.py`` because it is env-resolved
    launch configuration like every other resolver in this module, and
    because both ``base`` and ``run_session`` need it — keeping it in
    ``base`` forced the run loop to import the class that imports it.

    Returns:
        Default AI config with env-resolved fields filled in.
    """
    min_rank, max_rank = resolve_human_rank_window()
    return AIConfigDict(
        **{
            **make_default_ai_config(),
            "priority_target_name": resolve_priority_target(),
            "human_target_min_rank": min_rank,
            "human_target_max_rank": max_rank,
            "role": resolve_fleet_role(),
            "doctrine": resolve_engagement_doctrine(),
        }
    )
