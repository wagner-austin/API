"""Fleet domain: the instance registry behind the HTTP surface.

Owns spawn/adopt/stop/restart/remove/drain over one bot child process
per instance name, and nothing about HTTP. Each bot is a CHILD
PROCESS, so an orchestrator dying can never kill a live tank -- and
because they outlive it, the manager both drains them on the way out
and adopts the survivors on the way back in ([[fleet-lifecycle]]).

What an operator may ASK for -- accounts, rooms, colours, roles, the
port -- is :mod:`tankpit_bot.service.fleet_config`; this module is
what is actually running.
"""

from __future__ import annotations

from platform_core.json_utils import JSONObject
from platform_core.logging import get_logger

from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot.runtime_artifacts import _INSTANCE_NAME, bot_run_dir
from tankpit_bot.service import _test_hooks as service_hooks
from tankpit_bot.service.constants import FLEET_CHILD_PORT_BASE, FLEET_CHILD_PORT_COUNT
from tankpit_bot.service.fleet_adoption import adopt_recorded_bots
from tankpit_bot.service.fleet_bot import (
    FleetBotDict,
    _child_environment,
    _ManagedBot,
)
from tankpit_bot.service.fleet_config import (
    configured_accounts,
    derive_instance,
    resolve_doctrine,
    resolve_human_min_rank,
    resolve_role,
    resolve_troop,
)
from tankpit_bot.service.fleet_error import FleetError
from tankpit_bot.service.fleet_record import (
    FleetProcessRecordDict,
    forget_process_record,
    write_process_record,
)
from tankpit_bot.service.fleet_telemetry import FleetTelemetry

log = get_logger(__name__)


class FleetManager:
    """Spawn and track one bot process per instance name."""

    def __init__(self) -> None:
        """Start with an empty registry and a fresh boot identity."""
        self._bots: dict[str, _ManagedBot] = {}
        self._telemetry = FleetTelemetry()
        self._draining = False
        self._boot_id = str(top_hooks.get_current_time_ms())

    @property
    def boot_id(self) -> str:
        """Identify THIS manager process to its clients.

        The control page keeps per-instance state (rows, HUD cards,
        cached summaries) keyed by names it learned from a previous
        poll. When the manager restarts, every one of those names is
        meaningless to the new process, and a page that kept polling
        them just drew 404s at itself. It compares this value instead
        and reloads when it changes.

        Two managers cannot share a value in practice because they
        cannot share a port: the second one fails to bind before it
        can serve anything.

        Returns:
            An opaque per-process identifier.
        """
        return self._boot_id

    def adopt(self) -> list[str]:
        """Re-attach to bots left running by a previous manager.

        Called once at boot, before serving. Bots whose processes are
        gone have their records cleared; survivors join the registry
        as if this manager had spawned them, carrying their original
        account, role, room, troop, bounds and start time.

        Returns:
            The adopted instance names, in sorted order.

        Raises:
            OSError: If a spawn record cannot be read.
            InvalidJsonError: If a spawn record is not valid JSON.
            JSONTypeError: If a spawn record is malformed.
        """
        for bot in adopt_recorded_bots():
            self._bots[bot.instance] = bot
        adopted = sorted(self._bots)
        if adopted:
            log.info("Fleet: adopted %d running bot(s): %s", len(adopted), ", ".join(adopted))
        return adopted

    def _allocate_service_port(self) -> int:
        """Pick a service port no live child is holding.

        Lowest free port first, so a port is reused as soon as the child
        that held it dies rather than the range marching upward until it
        is exhausted by a long-lived fleet.

        Only LIVE children reserve a port: a dead row keeps its port in
        its report so the number that served its video stays readable,
        but it must not keep the port out of circulation.

        Returns:
            A port in ``[FLEET_CHILD_PORT_BASE, +FLEET_CHILD_PORT_COUNT)``.

        Raises:
            FleetError: If every port in the range is held by a live
                child. Refused rather than wrapped around, because two
                children on one port would serve each other's video.
        """
        taken = {bot.service_port for bot in self._bots.values() if bot.process.is_running()}
        for offset in range(FLEET_CHILD_PORT_COUNT):
            port = FLEET_CHILD_PORT_BASE + offset
            if port not in taken:
                return port
        raise FleetError(
            f"no free child service port in [{FLEET_CHILD_PORT_BASE}, "
            f"{FLEET_CHILD_PORT_BASE + FLEET_CHILD_PORT_COUNT}) - "
            f"{len(taken)} live children hold every one"
        )

    def spawn(
        self,
        *,
        instance: str,
        account: str,
        kills: int,
        seconds: int,
        role: str = "",
        room: str = "",
        troop: str = "",
        doctrine: str = "",
    ) -> FleetBotDict:
        """Spawn one bot child process under an instance namespace.

        Args:
            instance: Instance name; validated against the same
                pattern as ``resolve_bot_instance`` so a bad name is
                rejected here, not by a crashed child.
            account: ``TANKPIT_ACCOUNT`` selector; empty uses the
                accounts.json default.
            kills: Kill bound (0 unbounded).
            seconds: Seconds bound (0 unbounded).
            room: ``TANKPIT_ROOM`` selector; empty keeps the child's
                default (Practice). Cross-room fleets stay safe: the
                knowledge exchange merges same-room reports only
                (2026-08-26).
            role: Fleet role selector; empty means fighter — the full
                doctrine is the primary configuration, a gatherer is
                an explicit operator choice ([[fleet-coordination]]).
            troop: Tank color name; empty keeps the account's own
                default tank for the map. Accounts hold one tank per
                color, so this picks which tank plays.

        Returns:
            The spawned instance's report row.

        Raises:
            FleetError: If the name is invalid, already registered and
                alive, the bounds are negative, or the role or troop
                is not a known one.
        """
        if not instance:
            instance = derive_instance(account)
        if not _INSTANCE_NAME.match(instance):
            raise FleetError(
                f"instance {instance!r} is not a valid instance name "
                "(lowercase alphanumeric plus -_, max 32 chars)"
            )
        if kills < 0 or seconds < 0:
            raise FleetError("bounds must be non-negative")
        resolved_role = resolve_role(role)
        resolved_troop = resolve_troop(troop)
        resolved_doctrine = resolve_doctrine(doctrine)
        configured = configured_accounts()
        if account and account not in configured:
            known = ", ".join(configured) or "none configured"
            raise FleetError(
                f"account {account!r} is not in accounts.json (accounts are "
                f"config, not free text; configured: {known})"
            )
        resolved_account = account or (configured[0] if configured else "")
        conflict = self.live_accounts().get(resolved_account)
        if conflict is not None:
            raise FleetError(
                f"account {resolved_account!r} already has a live bot "
                f"({conflict['instance']!r}, pid {conflict['pid']}) - the game refuses "
                "a second login on the same account"
            )
        existing = self._bots.get(instance)
        if existing is not None and existing.process.is_running():
            raise FleetError(
                f"instance {instance!r} is already running (pid {existing.process.pid})"
            )
        service_port = self._allocate_service_port()
        process = service_hooks.spawn_bot_process(
            _child_environment(
                instance=instance,
                kills=kills,
                seconds=seconds,
                resolved_role=resolved_role,
                account=account,
                room=room,
                troop=resolved_troop,
                doctrine=resolved_doctrine,
                human_min_rank=resolve_human_min_rank(room),
                service_port=service_port,
            )
        )
        bot = _ManagedBot(
            instance=instance,
            service_port=service_port,
            account=account,
            role=resolved_role,
            room=room,
            troop=resolved_troop,
            doctrine=resolved_doctrine,
            kills=kills,
            seconds=seconds,
            started_ms=top_hooks.get_current_time_ms(),
            process=process,
        )
        self._bots[instance] = bot
        self._record_spawn(bot)
        log.info(
            "Fleet: spawned instance %r pid %d (role=%s kills=%d seconds=%d)",
            instance,
            process.pid,
            resolved_role,
            kills,
            seconds,
        )
        return bot.report()

    def _record_spawn(self, bot: _ManagedBot) -> None:
        """Persist what a future manager needs to find this bot again.

        A child that has already exited by the time it is asked for
        its identity leaves NO record, on purpose: there is nothing
        for a later manager to adopt, and a record naming a dead pid
        would only be re-checked and discarded on every future boot.

        Args:
            bot: The freshly registered bot.

        Returns:
            None.
        """
        created_at = service_hooks.process_identity(bot.process.pid)
        if created_at is None:
            log.warning(
                "Fleet: instance %r (pid %d) exited before it could be recorded; "
                "a later manager will not be able to adopt it",
                bot.instance,
                bot.process.pid,
            )
            return
        write_process_record(
            FleetProcessRecordDict(
                instance=bot.instance,
                account=bot.account,
                role=bot.role,
                room=bot.room,
                troop=bot.troop,
                doctrine=bot.doctrine,
                kills=bot.kills,
                seconds=bot.seconds,
                started_ms=bot.started_ms,
                pid=bot.process.pid,
                created_at=created_at,
                service_port=bot.service_port,
            )
        )

    def report(self) -> list[FleetBotDict]:
        """Return every registered instance's current state.

        Returns:
            Report rows sorted by instance name.
        """
        return [self._bots[name].report() for name in sorted(self._bots)]

    def stop(self, instance: str) -> FleetBotDict:
        """Request a graceful stop: write the instance's stop sentinel.

        The bot's tick loop polls the sentinel and exits at the next
        boundary with a full teardown — scorecard, capture save, and
        archive all happen, exactly as a bounded session ends.

        Args:
            instance: Registered instance name.

        Returns:
            The instance's report row after the request.

        Raises:
            FleetError: If the instance is not registered.
        """
        bot = self._bots.get(instance)
        if bot is None:
            raise FleetError(f"unknown instance {instance!r}")
        self._request_stop(bot)
        return bot.report()

    def _request_stop(self, bot: _ManagedBot) -> None:
        """Write one bot's stop sentinel.

        Args:
            bot: The registered bot to ask to stop.

        Returns:
            None.
        """
        sentinel = bot_run_dir(bot.instance) / "STOP"
        top_hooks.write_text(sentinel, "")
        log.info("Fleet: stop requested for %r (sentinel %s)", bot.instance, sentinel)

    def live_accounts(self) -> dict[str, FleetBotDict]:
        """Map every account holding a live bot to that bot's report row.

        The game refuses a second login on one account, so an account
        with a live tank is spoken for. Two callers need that fact and
        must not disagree about it: :meth:`spawn` REFUSES a request for
        a taken account, and the public demo surface PICKS an untaken
        one rather than asking. Resolving an empty selector to the
        configured default therefore happens here, once — when spawn
        and the demo each resolved it themselves, a demo bot could pick
        the default account under its own name and be refused by the
        spawn it had just decided was safe.

        Returns:
            Resolved username -> the report row of the live bot holding
            it. An account nothing is running under is absent. Empty
            when nothing is live, and also when no accounts are
            configured at all: an unresolvable selector names no
            account and so reserves none.
        """
        configured = configured_accounts()
        default = configured[0] if configured else ""
        held: dict[str, FleetBotDict] = {}
        for bot in self._bots.values():
            resolved = bot.account or default
            if resolved and bot.process.is_running():
                held[resolved] = bot.report()
        return held

    def live_instances(self) -> list[str]:
        """Return the instances whose processes are still running.

        Returns:
            Instance names in sorted order.
        """
        return sorted(name for name, bot in self._bots.items() if bot.process.is_running())

    def draining(self) -> bool:
        """Report whether a shutdown drain has been requested.

        Returns:
            True once :meth:`request_drain` has been called.
        """
        return self._draining

    def request_drain(self) -> list[str]:
        """Ask every live bot to stop, so the manager can exit cleanly.

        This is a DRAIN, never a kill. Each bot gets the same stop
        sentinel a single ``stop`` writes, and tears down the same way
        -- scorecard, capture save, archive, and the quit-to-lobby
        that stops the tank being left exposed in a live game. A tank
        killed outright instead would lose its rank.

        Idempotent: calling it again re-writes the sentinels, which a
        bot mid-teardown ignores.

        Returns:
            The instances asked to stop, in sorted order. Empty means
            nothing was running and the manager can exit immediately.
        """
        self._draining = True
        draining = []
        for instance in self.live_instances():
            self._request_stop(self._bots[instance])
            draining.append(instance)
        log.info(
            "Fleet: drain requested; %d bot(s) tearing down: %s",
            len(draining),
            ", ".join(draining) or "none",
        )
        return draining

    def restart(self, instance: str) -> FleetBotDict:
        """Respawn a finished instance with the parameters it had.

        ALL of them: account, bounds, role, room, and troop. The room was
        missing here from the day it was added to spawn (2026-08-26)
        until 2026-08-28, so a restart silently relocated the bot to
        the default Practice room — the row said ``World``, the child
        joined Practice, and only the run log disagreed.

        The fleet never silently kills: restarting a LIVE instance is
        refused — stop it first, let the teardown run, then restart.

        Args:
            instance: Registered instance name.

        Returns:
            The respawned instance's report row.

        Raises:
            FleetError: If the instance is unknown or still alive.
        """
        bot = self._bots.get(instance)
        if bot is None:
            raise FleetError(f"unknown instance {instance!r}")
        if bot.process.is_running():
            raise FleetError(f"instance {instance!r} is still running; stop it first")
        return self.spawn(
            instance=instance,
            account=bot.account,
            kills=bot.kills,
            seconds=bot.seconds,
            role=bot.role,
            room=bot.room,
            troop=bot.troop,
        )

    def stats_gate(self, instance: str) -> None:
        """Refuse telemetry reads for unregistered instances.

        Args:
            instance: Candidate instance name.

        Raises:
            FleetError: If the instance is not registered.
        """
        if instance not in self._bots:
            raise FleetError(f"unknown instance {instance!r}")

    def live_service_port(self, instance: str) -> int:
        """Return the service port of a RUNNING instance.

        Liveness is part of the answer, not a courtesy check. A dead
        bot's port goes straight back into circulation
        (:meth:`_allocate_service_port` reserves live children only), so
        by the time anyone asks about a finished instance that number
        may already belong to a different bot. Relaying to it would
        serve one bot's video under another's name -- the exact
        confusion the allocator exists to prevent.

        Args:
            instance: Candidate instance name.

        Returns:
            The port this instance's own service is serving on.

        Raises:
            FleetError: If the instance is unregistered, or is
                registered but no longer running.
        """
        bot = self._bots.get(instance)
        if bot is None:
            raise FleetError(f"unknown instance {instance!r}")
        if not bot.process.is_running():
            raise FleetError(f"instance {instance!r} is not running")
        return bot.service_port

    def stats(self, instance: str) -> JSONObject:
        """Summarize a registered instance's latest run from its events.

        The digest reduction (:mod:`fleet_telemetry`) — the same truth
        table ``make digest`` prints, reduced to the fields the control
        page shows, cached so 1 s page polling costs one events parse
        per cache window. Works on live runs and on crashed ones.

        Args:
            instance: Registered instance name.

        Returns:
            ``{"available": False}`` when the instance has produced no
            events yet, else the summary with ``"available": True``.

        Raises:
            FleetError: If the instance is not registered.
        """
        if instance not in self._bots:
            raise FleetError(f"unknown instance {instance!r}")
        return self._telemetry.stats(instance)

    def activity(self, instance: str) -> JSONObject:
        """Return the live tail of a registered instance's run.

        Current bot state, tick, fuel, and the last few AI/WORLD/STATE
        lines (:mod:`fleet_telemetry`) — what the bot is doing right
        now, for the control page's activity feed.

        Args:
            instance: Registered instance name.

        Returns:
            ``{"available": False}`` before the first events, else the
            tail with ``"available": True``.

        Raises:
            FleetError: If the instance is not registered.
        """
        if instance not in self._bots:
            raise FleetError(f"unknown instance {instance!r}")
        return self._telemetry.activity(instance)

    def remove(self, instance: str) -> FleetBotDict:
        """Drop a finished instance from the registry.

        Args:
            instance: Registered instance name.

        Returns:
            The removed instance's final report row.

        Raises:
            FleetError: If the instance is unknown or still alive —
                the fleet never silently kills; stop it first.
        """
        bot = self._bots.get(instance)
        if bot is None:
            raise FleetError(f"unknown instance {instance!r}")
        if bot.process.is_running():
            raise FleetError(f"instance {instance!r} is still running; stop it first")
        del self._bots[instance]
        forget_process_record(instance)
        self._telemetry.forget(instance)
        return bot.report()


__all__ = [
    "FleetError",
    "FleetManager",
]
