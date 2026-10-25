"""What each mode of ``fleet-runners`` does, and how it prints it.

Support module, not a command: :mod:`fleet.cli.runners` parses the flags,
refuses the combinations it cannot honour and calls exactly one of these.
Split out on 2026-10-20 (MCPs board task 1e9ab738) when the prune mode took
the command module to 595 lines: the parsing and the acts are two concerns,
the way their suites already were (``test_runner_cli`` and
``test_runner_cli_acts``).
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence

from platform_core.json_utils import dump_json_str
from platform_core.logging import get_logger

from fleet.contracts.runners import HostRunnerSpec, RunnerSpec, encode_runner_spec
from fleet.core import (
    _test_hooks,
    runner_audit,
    runner_load,
    runner_onboard,
    runner_rebuild,
    runner_registrations,
    runner_render,
)
from fleet.core.runner_audit_types import AuditFinding

_log = get_logger(__name__)

#: What :func:`log_findings` prefixes a roster-wide row with, where a host's
#: rows carry its name.
ROSTER_LABEL = "roster"


def audit(spec: RunnerSpec, hosts: Sequence[HostRunnerSpec]) -> int:
    """Audit every selected host and its repositories' registrations, and print the verdicts.

    Args:
        spec: The whole roster, which every registration is held to.
        hosts: The hosts to audit, whose installs name the repositories
            whose registrations are read.

    Returns:
        0 when every check on every reachable host passed, every host was
        reachable and every registration is declared; 1 otherwise.

    Raises:
        AppError: ``RUNNER_AUDIT_UNPARSABLE`` from the scorer -- a transcript
            this tooling cannot score is a fault in the tooling, not a fleet
            condition, and must stop the audit rather than print as a line --
            and ``RUNNER_GITHUB_UNREADABLE`` from a registrations read.
    """
    outcomes = [(host["name"], runner_audit.attempt_audit_host(host)) for host in hosts]
    faults = 0
    for name, outcome in outcomes:
        findings = outcome["findings"]
        if findings is None:
            _log.info("%s UNREACHABLE %s", name, outcome["reason"])
            faults += 1
            continue
        faults += log_findings(name, findings)
    repos = sorted({install["repo"] for host in hosts for install in host["installs"]})
    readings = runner_registrations.read_registrations(spec, repos)
    faults += log_findings(
        ROSTER_LABEL, [runner_registrations.registration_finding(each) for each in readings]
    )
    if faults:
        _log.info("%d fault(s); the fleet does not match its roster", faults)
        return 1
    return 0


def render(host: HostRunnerSpec, out_dir: str) -> int:
    """Write the converge scripts for one host and print the run order.

    Args:
        host: The host to render for.
        out_dir: Directory receiving ``base.sh`` and ``provision.sh``.

    Returns:
        0 always: rendering is a local act that either completes or raises.
    """
    rendered = runner_render.render_provision(host)
    base_path = pathlib.Path(out_dir) / "base.sh"
    provision_path = pathlib.Path(out_dir) / "provision.sh"
    _test_hooks.write_text(base_path, rendered["base_script"])
    _test_hooks.write_text(provision_path, rendered["provision_script"])
    _log.info("wrote %s", base_path)
    _log.info("wrote %s", provision_path)
    _log.info("RUN ORDER for %s:", host["name"])
    _log.info(
        "  1. base.sh on the host as root, then reboot if it printed %s",
        runner_render.REBOOT_MARKER,
    )
    _log.info("  2. provision.sh on the host as root, with each RUNNER_TOKEN_* exported")
    for step in rendered["manual_steps"]:
        _log.info("  %s", step)
    _log.info("  3. fleet-runners --spec <roster> --host %s", host["name"])
    return 0


def sample(hosts: Sequence[HostRunnerSpec], out_path: str) -> int:
    """Append one load reading per host to the record file.

    Args:
        hosts: The hosts to sample.
        out_path: The JSONL record to append to.

    Returns:
        0 always: an unreachable host RAISES here rather than scoring, per
        :func:`fleet.core.runner_load.sample_host` -- a recorded zero from a
        host the sampler could not see would be the healthiest possible
        reading taken at the exact moment it knows nothing.
    """
    for host in hosts:
        reading = runner_load.sample_host(host, at=_test_hooks.now())
        _test_hooks.append_text(pathlib.Path(out_path), runner_load.render_sample_line(reading))
        _log.info(
            "%s total=%d %s",
            host["name"],
            reading["total"],
            " ".join(
                f"{install['runner_name']}({install['repo'].split('/')[1]})="
                f"{install['descendants']}"
                for install in reading["installs"]
            ),
        )
    return 0


def report(record_path: str, *, cores: int, host_name: str | None) -> int:
    """Print the distribution over a sample record.

    Args:
        record_path: The JSONL record ``--sample`` appended to.
        cores: The reference core count. A reference line, not a threshold.
        host_name: Restrict to one host's samples, or None for all.

    Returns:
        0 always: the report judges nothing, so it has no failing verdict
        to exit with. See :class:`fleet.core.runner_load.LoadReport`.

    Raises:
        ValueError: From the reporter, on an empty selection or bad cores.
        JSONTypeError: From the record decoder, on a corrupt line.
    """
    samples = runner_load.decode_sample_file(_test_hooks.read_text(pathlib.Path(record_path)))
    if host_name is not None:
        samples = [each for each in samples if each["host"] == host_name]
    load = runner_load.report_over_cores(samples, cores=cores)
    _log.info("samples: %d", load["samples"])
    _log.info("total descendants: min=%d max=%d", load["minimum"], load["maximum"])
    _log.info(
        "over %d cores: %.1f%% of samples (longest consecutive streak: %d)",
        cores,
        load["over_fraction"] * 100.0,
        load["longest_over_streak"],
    )
    _log.info("at or under %d cores: %.1f%% of samples", cores, load["under_fraction"] * 100.0)
    return 0


def onboard(spec: RunnerSpec, spec_path: str, host: HostRunnerSpec, repo: str) -> int:
    """Onboard a repo onto a host, rewrite the roster, print the audit.

    Args:
        spec: The whole roster; the host entry inside it is mutated by the
            onboarder and the whole document is rewritten on success.
        spec_path: Where the roster lives, for the rewrite.
        host: The target host's entry within ``spec``.
        repo: The repository, ``owner/name``.

    Returns:
        0 when the post-onboard audit is clean; 1 when it reports drift --
        the runner exists and the roster records it either way, and the
        drift lines say exactly what to repair.

    Raises:
        AppError: As :func:`fleet.core.runner_onboard.onboard` describes.
    """
    plan, findings = runner_onboard.onboard(host, repo)
    _test_hooks.write_text(
        pathlib.Path(spec_path), dump_json_str(encode_runner_spec(spec), indent=2) + "\n"
    )
    install = plan["install"]
    _log.info(
        "onboarded %s runner %s (labels: %s)",
        repo,
        install["runner_name"],
        ",".join(install["labels"]),
    )
    _log.info("roster rewritten: %s", spec_path)
    faults = log_findings(host["name"], findings)
    if faults:
        _log.info("%d fault(s) after onboarding; the drift lines name the repairs", faults)
        return 1
    _log.info("onboarding verified: every check green, %s covered from birth", repo)
    return 0


def rebuild(host: HostRunnerSpec) -> int:
    """Rebuild one runner host and print each stage and the audit.

    Args:
        host: The host to rebuild.

    Returns:
        0 when the post-rebuild audit is clean; 1 when it reports drift, a
        manual asset being the usual one, named by its own line.

    Raises:
        AppError: As :func:`fleet.core.runner_rebuild.rebuild` describes.
    """
    rebuilt = runner_rebuild.rebuild(host)
    for step in rebuilt["steps"]:
        _log.info("%s rebuild: %s", host["name"], step)
    for step in rebuilt["manual_steps"]:
        _log.info("%s %s", host["name"], step)
    faults = log_findings(host["name"], rebuilt["findings"])
    if faults:
        _log.info("%d fault(s) after the rebuild; the drift lines name the repairs", faults)
        return 1
    _log.info("rebuild verified: every check green on %s", host["name"])
    return 0


def prune(spec: RunnerSpec) -> int:
    """Delete the roster's undeclared offline registrations and print each act.

    Args:
        spec: The roster.

    Returns:
        0 when no undeclared registration is left; 1 when GitHub reads one
        online or busy, which is named and kept.

    Raises:
        AppError: As :func:`fleet.core.runner_registrations.prune_registrations`
            describes.
    """
    pruned = runner_registrations.prune_registrations(spec)
    for registration in pruned["removed"]:
        _log.info(
            "%s PRUNED %s %s",
            ROSTER_LABEL,
            registration["repo"],
            runner_registrations.describe_registration(registration),
        )
    for registration in pruned["kept"]:
        _log.info(
            "%s KEPT %s %s -- a live machine no roster host declares: declare it, or stop its "
            "runner and prune again",
            ROSTER_LABEL,
            registration["repo"],
            runner_registrations.describe_registration(registration),
        )
    _log.info(
        "prune: examined %d registration(s); removed %d, kept %d",
        pruned["examined"],
        len(pruned["removed"]),
        len(pruned["kept"]),
    )
    return 1 if pruned["kept"] else 0


def log_findings(label: str, findings: Sequence[AuditFinding]) -> int:
    """Print audit verdicts under the name of what they judge.

    Args:
        label: The audited host's name, or :data:`ROSTER_LABEL` for the
            roster-wide registrations rows.
        findings: Its verdicts.

    Returns:
        How many drifted.
    """
    faults = 0
    for finding in findings:
        if finding["ok"]:
            _log.info("%s OK %s", label, finding["check_id"])
        else:
            _log.info(
                "%s DRIFT %s -- %s (%s)",
                label,
                finding["check_id"],
                finding["detail"],
                finding["reason"],
            )
            faults += 1
    return faults


__all__ = [
    "ROSTER_LABEL",
    "audit",
    "log_findings",
    "onboard",
    "prune",
    "rebuild",
    "render",
    "report",
    "sample",
]
