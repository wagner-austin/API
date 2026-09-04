"""Runtime logging: which run is active, and the per-channel emitters.

Owns the ambient answer to "which run do my events belong to" and the
``emit_*`` channels that write them. The handler plumbing those channels
land on is :mod:`tankpit_bot.runtime_logging_handlers`; the record shape
and codec are :mod:`tankpit_bot.runtime_records`; the per-tick context
merged into every record is :mod:`tankpit_bot.runtime_context`.

**The active run is ambient, not global** ([[session-state-deglobalisation]]
step 10). The artifact slots are :class:`contextvars.ContextVar`s for the
same reason the tick context is: the ``emit_*`` channels are called from
hundreds of sites inside pure planner logic, so threading a parameter
would cost far more than the globals did, and a context variable still
isolates per thread and per async task -- which is what lets two
concurrent sessions in one process keep their own ``events.jsonl``
instead of the second silently detaching the first's handler.
"""

from __future__ import annotations

from contextvars import ContextVar
from pathlib import Path

from platform_core.logging import get_logger, stdlib_logging
from platform_core.rich_logging import setup_rich_logging

from tankpit_bot import _test_hooks
from tankpit_bot.runtime_artifacts import (
    BotRunArtifactsDict,
    ProbeRunArtifactsDict,
    SniffRunArtifactsDict,
    build_bot_run_artifacts,
    build_probe_run_artifacts,
    build_sniff_run_artifacts,
    make_run_stamp,
    resolve_bot_instance,
)
from tankpit_bot.runtime_context import get_runtime_context
from tankpit_bot.runtime_logging_handlers import (
    EMITTER_LOGGER_NAME,
    install_artifact_handlers,
    make_run_id,
    remove_artifact_handlers,
    reset_artifact_files,
    session_logger_name,
)
from tankpit_bot.runtime_records import _RESERVED_EVENT_KEYS, RuntimeLogExtraDict

_EMITTER_LOGGER = get_logger(EMITTER_LOGGER_NAME)

# One typed slot per run kind so each getter stays mypy-narrowed at its
# source, matching the idiom in :mod:`tankpit_bot.runtime_context`.
# Exactly one is ever set: ``_set_active_run`` clears the other two.
_BOT_ARTIFACTS: ContextVar[BotRunArtifactsDict | None] = ContextVar(
    "tankpit_bot_artifacts", default=None
)

_SNIFF_ARTIFACTS: ContextVar[SniffRunArtifactsDict | None] = ContextVar(
    "tankpit_sniff_artifacts", default=None
)

_PROBE_ARTIFACTS: ContextVar[ProbeRunArtifactsDict | None] = ContextVar(
    "tankpit_probe_artifacts", default=None
)

#: Identifies which per-run logger ``emit_*`` writes to. ``None`` before
#: any ``configure_*`` call, which is the unconfigured mode: events go to
#: the base emitter logger, where no event handler is mounted, so they
#: reach the console and the process text log but no ``events.jsonl``.
_ACTIVE_RUN_ID: ContextVar[str | None] = ContextVar("tankpit_active_run_id", default=None)


def _set_active_run(
    run_id: str,
    *,
    bot: BotRunArtifactsDict | None = None,
    sniff: SniffRunArtifactsDict | None = None,
    probe: ProbeRunArtifactsDict | None = None,
) -> None:
    """Make one run the ambient run for this thread or task.

    Exactly one artifacts argument is ever passed; the other two slots
    are cleared so the getters cannot report a stale run from a previous
    ``configure_*`` in the same context.

    Args:
        run_id: Run identity from :func:`make_run_id`.
        bot: Bot artifacts when configuring a bot run.
        sniff: Sniffer artifacts when configuring a sniff run.
        probe: Probe artifacts when configuring a probe run.
    """
    _BOT_ARTIFACTS.set(bot)
    _SNIFF_ARTIFACTS.set(sniff)
    _PROBE_ARTIFACTS.set(probe)
    _ACTIVE_RUN_ID.set(run_id)


def configure_bot_runtime_logging(stamp: str | None = None) -> BotRunArtifactsDict:
    """Configure console logging plus canonical bot artifact outputs.

    Args:
        stamp: Optional archive timestamp stamp for deterministic tests.

    Returns:
        Configured bot runtime artifacts.
    """
    resolved_stamp = stamp if stamp is not None else make_run_stamp()
    instance = resolve_bot_instance()
    artifacts = build_bot_run_artifacts(resolved_stamp, instance)
    setup_rich_logging(level="INFO")
    reset_artifact_files(
        Path(artifacts["latest_log_path"]),
        Path(artifacts["archive_log_path"]),
        Path(artifacts["latest_events_path"]),
        Path(artifacts["archive_events_path"]),
    )
    run_id = make_run_id("bot", resolved_stamp)
    install_artifact_handlers(
        run_id,
        "bot",
        Path(artifacts["latest_log_path"]),
        Path(artifacts["archive_log_path"]),
        Path(artifacts["latest_events_path"]),
        Path(artifacts["archive_events_path"]),
    )
    _set_active_run(run_id, bot=artifacts)
    # The run's own provenance, stamped as the artifact's first record
    # (board task 7e766d65): 539 archived runs cannot say what
    # produced them, and every future one now does. Session-scoped
    # like ``session_room_joined`` — no tick has happened yet. The
    # account name is DELIBERATELY absent (the tank_registry username
    # exposure decision is open; a stamp here would widen it into
    # every artifact).
    doctrine = _test_hooks.get_env("TANKPIT_DOCTRINE")
    room = _test_hooks.get_env("TANKPIT_ROOM")
    emit_diagnostic(
        diagnostic_kind="session_build",
        distribution_version=_test_hooks.read_distribution_version("tankpit-bot"),
        build_ref=_test_hooks.resolve_build_ref(),
        instance=instance,
        doctrine=doctrine if doctrine is not None else "",
        room=room if room is not None else "",
    )
    return artifacts


def configure_sniff_runtime_logging(stamp: str | None = None) -> SniffRunArtifactsDict:
    """Configure console logging plus canonical sniffer artifact outputs.

    Args:
        stamp: Optional archive timestamp stamp for deterministic tests.

    Returns:
        Configured sniffer runtime artifacts.
    """
    resolved_stamp = stamp if stamp is not None else make_run_stamp()
    artifacts = build_sniff_run_artifacts(resolved_stamp)
    setup_rich_logging(level="INFO")
    reset_artifact_files(
        Path(artifacts["latest_log_path"]),
        Path(artifacts["archive_log_path"]),
        Path(artifacts["latest_events_path"]),
        Path(artifacts["archive_events_path"]),
        Path(artifacts["latest_capture_path"]),
        Path(artifacts["latest_raw_capture_path"]),
        Path(artifacts["latest_summary_path"]),
    )
    run_id = make_run_id("sniff", resolved_stamp)
    install_artifact_handlers(
        run_id,
        "sniff",
        Path(artifacts["latest_log_path"]),
        Path(artifacts["archive_log_path"]),
        Path(artifacts["latest_events_path"]),
        Path(artifacts["archive_events_path"]),
    )
    _set_active_run(run_id, sniff=artifacts)
    return artifacts


def configure_probe_runtime_logging(
    probe_name: str,
    stamp: str | None = None,
    runs_root: str | None = None,
) -> ProbeRunArtifactsDict:
    """Configure console logging plus canonical probe artifact outputs.

    Args:
        probe_name: Probe identifier (``fuel``, ``equipment``, ``movement``,
            ``teleport``, ``enemy_teleport``, ``fuel_drill``). Embedded in
            archive filenames so multiple probe kinds share
            ``runs/probe/``.
        stamp: Optional archive timestamp stamp for deterministic tests.
        runs_root: Optional directory the artifacts land under, replacing
            the fixed ``runs/`` root. A cluster array must set it per
            task; see :func:`~tankpit_bot.runtime_artifacts.build_probe_run_artifacts`.

    Returns:
        Configured probe runtime artifacts.

    Raises:
        ValueError: When ``probe_name`` is empty (validated by
            :func:`tankpit_bot.runtime_artifacts.build_probe_run_artifacts`).
    """
    resolved_stamp = stamp if stamp is not None else make_run_stamp()
    artifacts = build_probe_run_artifacts(
        probe_name, resolved_stamp, None if runs_root is None else Path(runs_root)
    )
    setup_rich_logging(level="INFO")
    reset_artifact_files(
        Path(artifacts["latest_log_path"]),
        Path(artifacts["archive_log_path"]),
        Path(artifacts["latest_events_path"]),
        Path(artifacts["archive_events_path"]),
    )
    mode = f"probe:{probe_name}"
    run_id = make_run_id(mode, resolved_stamp)
    install_artifact_handlers(
        run_id,
        mode,
        Path(artifacts["latest_log_path"]),
        Path(artifacts["archive_log_path"]),
        Path(artifacts["latest_events_path"]),
        Path(artifacts["archive_events_path"]),
    )
    _set_active_run(run_id, probe=artifacts)
    return artifacts


def get_bot_runtime_artifacts() -> BotRunArtifactsDict | None:
    """Return this thread or task's bot runtime artifacts, if configured."""
    return _BOT_ARTIFACTS.get()


def get_sniff_runtime_artifacts() -> SniffRunArtifactsDict | None:
    """Return this thread or task's sniffer runtime artifacts, if configured."""
    return _SNIFF_ARTIFACTS.get()


def get_probe_runtime_artifacts() -> ProbeRunArtifactsDict | None:
    """Return this thread or task's probe runtime artifacts, if configured."""
    return _PROBE_ARTIFACTS.get()


def clear_runtime_logging_state() -> None:
    """Detach every artifact handler and forget the ambient run.

    The ambient run is a context variable, so it survives between tests
    on the same thread exactly as the tick context does — which is why
    the test suite still resets it explicitly. Production never calls
    this: a process either configures a run or never had one.
    """
    root = stdlib_logging.getLogger()
    remove_artifact_handlers(root)
    run_id = _ACTIVE_RUN_ID.get()
    if run_id is not None:
        remove_artifact_handlers(stdlib_logging.getLogger(session_logger_name(run_id)))
    _set_active_run_cleared()


def _set_active_run_cleared() -> None:
    """Clear all four ambient run slots."""
    _BOT_ARTIFACTS.set(None)
    _SNIFF_ARTIFACTS.set(None)
    _PROBE_ARTIFACTS.set(None)
    _ACTIVE_RUN_ID.set(None)


def emit_ai(
    message: str,
    *args: str | int | float | bool,
    **fields: str | int | float | bool,
) -> None:
    """Emit a structured AI event.

    Args:
        message: ``printf``-style message string.
        *args: Format arguments for the message string.
        **fields: Optional structured key/value payload spread into the
            JSONL event at the top level. Use for fields that tooling
            should query (e.g. ``combat_target_x``) so the smoke gate
            and ``bot-query`` reach them without parsing the message.
    """
    _emit_runtime_event("AI", message, *args, **fields)


def emit_sync(message: str, *args: str | int | float | bool) -> None:
    """Emit a structured synchronization event."""
    _emit_runtime_event("SYNC", message, *args)


def emit_state(message: str, *args: str | int | float | bool) -> None:
    """Emit a structured state-machine event."""
    _emit_runtime_event("STATE", message, *args)


def emit_wire(
    message: str,
    *args: str | int | float | bool,
    **fields: str | int | float | bool,
) -> None:
    """Emit a structured wire/protocol command event.

    Args:
        message: ``printf``-style message string.
        *args: Format arguments for the message string.
        **fields: Optional structured payload (e.g. ``action_kind``,
            ``target_x``) spread into the JSONL event at the top
            level so smoke + ``bot-query`` can reach them without
            parsing the message text.
    """
    _emit_runtime_event("WIRE", message, *args, **fields)


def emit_world(message: str, *args: str | int | float | bool) -> None:
    """Emit a structured world-state event."""
    _emit_runtime_event("WORLD", message, *args)


def emit_diagnostic(
    *,
    diagnostic_kind: str,
    **fields: str | int | float | bool,
) -> None:
    """Emit a structured diagnostic event.

    The ``DIAGNOSTIC`` channel carries observability-only emissions:
    target-selection breakdowns, attempt-window message timelines,
    invariant-violation reports, and the like. Every emit lands on
    ``runs/<mode>/latest.events.jsonl`` with ``diagnostic_kind`` plus
    any caller-supplied primitive fields spread at the top level, so
    queries against the JSONL can filter by kind and compare timing
    distributions across runs.

    Args:
        diagnostic_kind: Stable identifier for the diagnostic shape
            (``teleport_attempt``, ``fuel_target_selection``,
            ``action_phase_overlap``, ``map_positions_parsed``,
            ``movement_probe_map_already_showing``,
            ``command_dispatch_failure``, etc.). The kind names the
            payload schema; callers passing structured fields must
            match the kind's documented schema.
        **fields: Caller-supplied structured fields. Each field name
            must not collide with the reserved top-level event keys
            (timestamp, level, logger, mode, channel, message) --
            collision raises at encode time.
    """
    message = f"diagnostic_kind={diagnostic_kind}"
    _emit_runtime_event(
        "DIAGNOSTIC",
        message,
        diagnostic_kind=diagnostic_kind,
        **fields,
    )


def _merge_context_into_fields(
    fields: dict[str, str | int | float | bool],
) -> dict[str, str | int | float | bool]:
    """Return a new field dict with the active runtime context attached.

    Context fields are written first; explicit ``fields`` win on
    collision so call-site arguments override the per-tick context.

    Args:
        fields: Explicit fields passed to the emit_* call.

    Returns:
        New dict containing the context fields (when set) plus the
        original ``fields`` overrides.
    """
    merged: dict[str, str | int | float | bool] = {}
    context = get_runtime_context()
    tick_n = context.get("tick_n")
    if tick_n is not None:
        merged["tick_n"] = tick_n
    bot_state = context.get("bot_state")
    if bot_state is not None:
        merged["bot_state"] = bot_state
    in_flight_action_kind = context.get("in_flight_action_kind")
    if in_flight_action_kind is not None:
        merged["in_flight_action_kind"] = in_flight_action_kind
    merged.update(fields)
    return merged


def _emit_runtime_event(
    channel: str,
    message: str,
    *args: str | int | float | bool,
    **fields: str | int | float | bool,
) -> None:
    """Emit a runtime event to both console logs and JSONL artifacts.

    The active runtime context (``tick_n`` / ``bot_state`` /
    ``in_flight_action_kind`` set via :func:`set_runtime_context`) is
    merged into the structured payload before write. Explicit
    ``fields`` override the context fields on collision.

    Args:
        channel: Event channel such as ``AI`` or ``WORLD``.
        message: ``printf``-style message string without the channel prefix.
        *args: Format arguments for the message string.
        **fields: Structured key/value payload spread into the JSONL event
            at the top level. Must not collide with the reserved event keys
            (timestamp, level, logger, mode, channel, message).

    Raises:
        ValueError: When a field name collides with a reserved event
            key. Validated HERE, at the call, not only in the JSONL
            handler: unit tests run without the handler attached, so
            a handler-only check let a covered ``level=`` emit ship
            and crash both fleet bots on the first live 0x4E
            decoration announcement (2026-08-26 05:11:17).
    """
    for key in fields:
        if key in _RESERVED_EVENT_KEYS:
            raise ValueError(f"runtime event field name {key!r} collides with reserved record key")
    formatted = message % args if args else message
    extra = RuntimeLogExtraDict(
        runtime_channel=channel,
        runtime_message=formatted,
        runtime_fields=_merge_context_into_fields(dict(fields)),
    )
    _active_emitter_logger().info("%s: %s", channel, formatted, extra=extra)


def _active_emitter_logger() -> stdlib_logging.Logger:
    """Return the logger this thread or task's events belong to.

    With a run configured this is that run's own logger, which carries
    its event handler; the record still propagates up to the root text
    handler and the console. With no run configured it is the base
    emitter logger, which has no event handler — so an unconfigured
    process logs to console and text but writes no ``events.jsonl``,
    which is what it did before the run became ambient.

    Returns:
        The logger to emit this record on.
    """
    run_id = _ACTIVE_RUN_ID.get()
    if run_id is None:
        return _EMITTER_LOGGER
    return stdlib_logging.getLogger(session_logger_name(run_id))


__all__ = [
    "clear_runtime_logging_state",
    "configure_bot_runtime_logging",
    "configure_probe_runtime_logging",
    "configure_sniff_runtime_logging",
    "emit_ai",
    "emit_diagnostic",
    "emit_state",
    "emit_sync",
    "emit_wire",
    "emit_world",
    "get_bot_runtime_artifacts",
    "get_probe_runtime_artifacts",
    "get_sniff_runtime_artifacts",
]
