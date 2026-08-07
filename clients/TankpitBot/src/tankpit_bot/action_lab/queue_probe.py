"""The queue probe: measure how the server orders queued commands.

Holds the :class:`QueueProbe` session, the run summary, and the entry
point. The experiments it runs are
:mod:`tankpit_bot.action_lab.queue_experiments`.

``run_single_experiment`` is reached through the experiments module
rather than imported by name, so the tests that swap it keep a working
injection seam.
"""

from __future__ import annotations

from platform_core.logging import get_logger

from tankpit_bot.action_lab import queue_experiments
from tankpit_bot.action_lab.probe_base import ProbeBase
from tankpit_bot.action_lab.probe_entrypoint import run_and_save_standard_probe_session
from tankpit_bot.action_lab.probe_runtime import (
    ProbeCommandReadyContextDict,
    execute_live_probe_bootstrap,
)
from tankpit_bot.action_lab.probe_session import build_probe_session_envelope
from tankpit_bot.action_lab.queue_probe_types import (
    QueueExperimentKind,
    QueueExperimentResultDict,
    QueueProbeSessionDict,
    encode_queue_probe_session,
)

log = get_logger(__name__)

_DEFAULT_EXPERIMENT_KINDS: list[QueueExperimentKind] = [
    "shoot_then_pickup",
    "shoot_then_shoot",
    "move_then_pickup",
]


class QueueProbe(ProbeBase):
    """Live probe that tests server command queue behavior."""

    def execute_probe(
        self,
        *,
        initial_sync_timeout_ms: int,
        experiment_timeout_ms: int,
        experiment_kinds: list[QueueExperimentKind],
    ) -> QueueProbeSessionDict:
        """Run the live queue probe session.

        Args:
            initial_sync_timeout_ms: Maximum wait for initial sync.
            experiment_timeout_ms: Maximum wait per experiment.
            experiment_kinds: Which experiments to run.

        Returns:
            Complete session with all experiment results.

        Raises:
            ValueError: If timeout is non-positive or kinds is empty.
        """
        if experiment_timeout_ms <= 0:
            raise ValueError("experiment_timeout_ms must be positive")
        if not experiment_kinds:
            raise ValueError("experiment_kinds must not be empty")

        def _run_ready_session(
            context: ProbeCommandReadyContextDict,
        ) -> QueueProbeSessionDict:
            experiments: list[QueueExperimentResultDict] = []
            for kind in experiment_kinds:
                result = queue_experiments.run_single_experiment(
                    self,
                    kind,
                    timeout_ms=experiment_timeout_ms,
                )
                experiments.append(result)
                log.info(
                    "Experiment %s: %s (primary %sms, secondary %sms, gap %dms)",
                    kind,
                    result["status"],
                    result["primary"]["elapsed_ms"],
                    result["secondary"]["elapsed_ms"],
                    result["inter_send_delay_ms"],
                )

            first_started_ms = experiments[0]["primary"]["sent_ms"] if experiments else None
            envelope = build_probe_session_envelope(
                self,
                context=context,
                first_attempt_started_ms=first_started_ms,
            )
            return QueueProbeSessionDict(
                session_id=envelope.session_id,
                start_timestamp_ms=envelope.start_timestamp_ms,
                end_timestamp_ms=envelope.end_timestamp_ms,
                base_url=envelope.base_url,
                spawn_x=envelope.spawn_x,
                spawn_y=envelope.spawn_y,
                capture_session_path="",
                initial_sync_timeout_ms=initial_sync_timeout_ms,
                experiment_timeout_ms=experiment_timeout_ms,
                startup_timing=envelope.startup_timing,
                experiments=experiments,
            )

        return execute_live_probe_bootstrap(
            self,
            initial_sync_timeout_ms=initial_sync_timeout_ms,
            run_ready_session=_run_ready_session,
        )


def format_queue_probe_summary(session: QueueProbeSessionDict) -> str:
    """Format a human-readable summary of a queue probe session.

    Args:
        session: Completed queue probe session.

    Returns:
        Multi-line summary string.
    """
    lines = [
        f"Queue probe session {session['session_id']}",
        f"  Spawn: ({session['spawn_x']}, {session['spawn_y']})",
        f"  Experiments: {len(session['experiments'])}",
    ]
    for exp in session["experiments"]:
        p_ms = exp["primary"]["elapsed_ms"]
        s_ms = exp["secondary"]["elapsed_ms"]
        lines.append(
            f"  {exp['kind']}: {exp['status']}"
            f" (primary={p_ms}ms, secondary={s_ms}ms, gap={exp['inter_send_delay_ms']}ms)"
        )
    return "\n".join(lines)


def _create_queue_probe(
    target_url: str,
    *,
    headless: bool,
    prefer_account: bool,
) -> QueueProbe:
    """Factory for creating QueueProbe instances with injected services.

    Args:
        target_url: Browser target URL.
        headless: Whether to run headless.
        prefer_account: Whether to prefer account login.

    Returns:
        New QueueProbe instance with factory-wired services.
    """
    from tankpit_bot.action_lab.probe_factory import create_probe

    probe = create_probe(
        QueueProbe,
        target_url,
        headless=headless,
        prefer_account=prefer_account,
    )
    assert isinstance(probe, QueueProbe)
    return probe


def run_queue_probe(
    target_url: str,
    output_path: str,
    *,
    headless: bool = False,
    prefer_account: bool = False,
    initial_sync_timeout_ms: int = 10000,
    experiment_timeout_ms: int = 5000,
    experiment_kinds: list[QueueExperimentKind] | None = None,
) -> QueueProbeSessionDict:
    """Run a live queue probe and save the session JSON.

    Args:
        target_url: Browser target URL.
        output_path: JSON output path.
        headless: Whether to run headless.
        prefer_account: Whether to prefer account login.
        initial_sync_timeout_ms: Maximum wait for initial sync.
        experiment_timeout_ms: Maximum wait per experiment.
        experiment_kinds: Which experiments to run (defaults to all three).

    Returns:
        Completed and persisted session payload.
    """
    kinds = experiment_kinds if experiment_kinds is not None else _DEFAULT_EXPERIMENT_KINDS

    def _run_session(probe: QueueProbe) -> QueueProbeSessionDict:
        return probe.execute_probe(
            initial_sync_timeout_ms=initial_sync_timeout_ms,
            experiment_timeout_ms=experiment_timeout_ms,
            experiment_kinds=kinds,
        )

    return run_and_save_standard_probe_session(
        probe_factory=_create_queue_probe,
        run_session=_run_session,
        encoder=encode_queue_probe_session,
        summary_formatter=format_queue_probe_summary,
        target_url=target_url,
        output_path=output_path,
        headless=headless,
        prefer_account=prefer_account,
    )


__all__ = [
    "QueueProbe",
    "format_queue_probe_summary",
    "log",
    "run_queue_probe",
]
