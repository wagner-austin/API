"""Rendering the converge scripts for a CI runner host.

``fleet-runners --render`` writes two artifacts and prints the run order,
both run on the host as root:

  base.sh       the roster's packages (docker and the GPU driver among
                them), docker enabled, and the runner account in its group.
  provision.sh  the hygiene units, the assets, the CI slice, the reaper,
                the job scopes, the action cache, and every runner install,
                configured and started.

ONE PROVISION, EVERY CALLER. The rebuild (:mod:`fleet.core.runner_rebuild`)
and the onboarding (:mod:`fleet.core.runner_onboard`) run the same
:func:`render_provision_script` the render writes, with the registration
tokens they minted written into its head. Every step is idempotent, so
onboarding one repository re-runs the whole provision over a host that
holds the rest: the host-wide units that list every install (the action
cache's fill and adopt services) then name the new one too, which a
provision of the new install alone left out.

A TOKEN IS READ ONLY WHERE IT IS USED. Registration tokens expire within
the hour and only ``config.sh`` reads one, for an install with no
``.runner`` yet, so the check that its variable is set sits inside that
branch: a re-run over configured runners needs none.

THE HYGIENE PAYLOAD LIVES HERE, NOT ON A BOX. ``ci-clean`` ran only on
lavender's disk until 2026-09-08, which made it exactly the folklore this
package exists to end: :mod:`fleet.core.ci_clean` is its source of truth, and
the rendered provision installs byte-identical copies everywhere.
"""

from __future__ import annotations

import re
from typing import Final

from typing_extensions import TypedDict

from fleet.contracts.runners import FileAsset, HostRunnerSpec, RunnerInstall
from fleet.core.ci_clean import CI_CLEAN_SCRIPT, CI_CLEAN_SERVICE, CI_CLEAN_TIMER
from fleet.core.runner_action_cache import render_fill_lines
from fleet.core.runner_cache_adopt import (
    render_adopt_lines,
    render_cache_adopt_now_lines,
    render_cache_drop_in_lines,
)
from fleet.core.runner_cpuset_render import render_ci_cpuset_lines
from fleet.core.runner_install import (
    RUNNER_ACCOUNT,
    RUNNER_VERSION,
    install_root_for,
    token_variable,
)
from fleet.core.runner_job_scope_render import render_job_hook_lines, render_job_scope_lines
from fleet.core.runner_reaper_render import render_kill_mode_lines, render_reaper_lines
from fleet.core.runner_recovery import render_recovery_lines
from fleet.core.runner_slice_render import (
    render_runner_slice_drop_in_lines,
    render_runner_slice_move_lines,
    render_slice_unit_lines,
)
from fleet.core.runner_unit_files import render_unit_file_lines
from fleet.core.script_values import scriptable

#: The deadline for one root payload, in seconds: the base installs the
#: GPU driver and a build toolchain, and the provision downloads a runner
#: release per install. Half an hour bounds a hung download without cutting
#: a slow honest one short.
STAGE_TIMEOUT_SECONDS: Final = 1800

#: The line the base prints when a package it installed asks for a reboot.
REBOOT_MARKER: Final = "FLEET_REBOOT_REQUIRED"

#: The file a package that needs a reboot writes, a new kernel module above all.
REBOOT_REQUIRED_PATH: Final = "/var/run/reboot-required"

#: A Debian package name: lowercase letters, digits and ``+-.``, starting
#: with a letter or digit, so a name is embedded in the apt line verbatim.
_APT_PACKAGE: Final = re.compile(r"[a-z0-9][a-z0-9+.-]*")

#: Where ``pipx install`` puts its shims for the runner account, appended to
#: each runner's ``.path`` (see :func:`render_install_lines`).
LOCAL_BIN: Final = f"/home/{RUNNER_ACCOUNT}/.local/bin"

#: Every root payload's first lines.
SCRIPT_HEADER: Final = ("#!/usr/bin/env bash", "set -euo pipefail")

#: Where the hygiene service and its daily timer are written.
CI_CLEAN_SERVICE_PATH: Final = "/etc/systemd/system/ci-clean.service"
CI_CLEAN_TIMER_PATH: Final = "/etc/systemd/system/ci-clean.timer"

#: The provision's last line: one hygiene pass, after every runner install,
#: so the cache generations it reports exist. The timer alone left a host
#: provisioned after 04:00 with no run for up to a day, and the runner host's
#: suite holds each host to a successful run within 26 hours: lavender joined
#: on 2026-10-19 and its first check failed on "ci-clean.timer has never
#: fired". ``start`` waits for the oneshot, so a failed pass fails the provision.
FIRST_HYGIENE_PASS: Final = "systemctl start ci-clean.service"


class RenderedProvision(TypedDict):
    """Everything ``fleet-runners --render`` produces for one host.

    Attributes:
        base_script: ``base.sh``, run as root first.
        provision_script: ``provision.sh``, run as root next with each
            ``RUNNER_TOKEN_*`` an unconfigured install needs exported.
        manual_steps: One line per asset no script can fetch, printed loudly.
            Empty only for a host with no such assets.
    """

    base_script: str
    provision_script: str
    manual_steps: list[str]


def render_base_script(spec: HostRunnerSpec) -> str:
    """The base stage for one host, run as root.

    Docker, the GPU driver and the fleet toolchain are the fleet node
    provision's (MCPs scripts/host/fleet-node/provision.sh), which every
    runner host has run, so the base refuses a host whose docker daemon is
    not active rather than installing a second one beside it.

    Args:
        spec: The host's roster entry.

    Returns:
        The bash text: refuses a host with no active docker, installs the
        roster's packages, creates the runner account in the docker group,
        proves the account can reach docker, and prints
        :data:`REBOOT_MARKER` when a package asked for a reboot.

    Raises:
        ValueError: When a package name is not a plain Debian package name.
    """
    packages = spec["base"]["apt_packages"]
    for package in packages:
        if not _APT_PACKAGE.fullmatch(package):
            raise ValueError(f"apt package {package!r} is not a plain package name")
    lines = [
        *SCRIPT_HEADER,
        "export DEBIAN_FRONTEND=noninteractive",
        "systemctl is-active --quiet docker || { echo 'docker is not active; run the fleet "
        "node provision (MCPs scripts/host/fleet-node/provision.sh) first' >&2; exit 1; }",
        "apt-get update -qq",
        f"apt-get install -y -qq {' '.join(packages)}",
        f"id {RUNNER_ACCOUNT} >/dev/null 2>&1 || useradd -m -s /bin/bash {RUNNER_ACCOUNT}",
        f"usermod -aG docker {RUNNER_ACCOUNT}",
        f"runuser -u {RUNNER_ACCOUNT} -- docker ps >/dev/null",
        f"if [ -e {REBOOT_REQUIRED_PATH} ]; then echo {REBOOT_MARKER}; fi",
        f"echo 'base ready: packages installed, docker reachable as {RUNNER_ACCOUNT}'",
    ]
    return "\n".join(lines) + "\n"


def render_asset_lines(asset: FileAsset) -> list[str]:
    """Provision lines for one fetchable asset.

    Args:
        asset: The asset, with ``manual`` False. The contract guarantees a
            non-writable one carries its own ``provision_command``.

    Returns:
        The bash lines that create it, idempotently: an asset already
        present is left alone, because the fetch belongs to first provision
        and drift afterwards is the audit's to report.

    Raises:
        ValueError: When the path cannot be embedded verbatim.
    """
    path = scriptable(asset["path"], label="asset path")
    if asset["writable"]:
        return [
            f"mkdir -p '{path}'",
            f"chown {RUNNER_ACCOUNT}:{RUNNER_ACCOUNT} '{path}'",
        ]
    return [
        f"if [ ! -e '{path}' ]; then",
        f"    {asset['provision_command']}",
        "fi",
    ]


def render_install_lines(install: RunnerInstall) -> list[str]:
    """Bash lines that install one runner, run as root.

    The install directory comes from the roster's own ``workdir`` -- the
    ``_work`` parent -- not from any derived name: an earlier version
    numbered directories by loop index, which would have installed to a
    path the roster never declared, and the audit would then have verified
    the declaration while the runner lived somewhere else.

    The release's own ``bin/installdependencies.sh`` installs the libraries
    the agent links, once, beside the unpack. ``--replace`` takes back a
    registration of the same name. :data:`LOCAL_BIN` goes onto the runner's
    ``.path`` AFTER config.sh, which writes that file, and BEFORE the service
    starts, because runsvc.sh reads it once at start and exports it as every
    job step's PATH, and a systemd-started runner has no login shell to add
    anything: without it, a job that installs poetry with pipx dies one step
    later at ``poetry: command not found``, exit 127 (measured on lavender).

    An install whose ``.runner`` file exists is already configured and
    config.sh is skipped, with the check of its token, and one whose
    ``.service`` file exists already has its unit and ``svc.sh install`` is
    skipped: both refuse a second run, and a rebuild re-runs every stage
    over a host that may be half built. ``svc.sh start`` always runs;
    starting a running unit is a no-op.

    Every drop-in is written before that start, because a new runner takes
    a queued job within seconds of it (the slice drop-in's docstring has
    the measurement); only the moves of a unit an older provision started
    come after it, and each restarts nothing that started with its drop-in.

    Args:
        install: The install to configure.

    Returns:
        The bash lines that download, configure and start it.

    Raises:
        ValueError: When the install directory cannot be embedded verbatim.
    """
    token_var = token_variable(install["repo"])
    directory = scriptable(install_root_for(install), label="workdir")
    labels = ",".join(install["labels"])
    return [
        f"mkdir -p '{directory}'",
        f"if [ ! -f '{directory}/config.sh' ]; then",
        f"    curl -fsSL -o '{directory}/runner.tar.gz' "
        f"https://github.com/actions/runner/releases/download/v{RUNNER_VERSION}/"
        f"actions-runner-linux-x64-{RUNNER_VERSION}.tar.gz",
        f"    tar -xzf '{directory}/runner.tar.gz' -C '{directory}'",
        f"    rm '{directory}/runner.tar.gz'",
        f"    '{directory}/bin/installdependencies.sh'",
        "fi",
        f"chown -R {RUNNER_ACCOUNT}:{RUNNER_ACCOUNT} '{directory}'",
        f"if [ ! -f '{directory}/.runner' ]; then",
        f'    : "${{{token_var}:?set {token_var} to a fresh registration token for '
        f'{install["repo"]}; fleet-runners --rebuild and --onboard mint it with gh}}"',
        f"    runuser -u {RUNNER_ACCOUNT} -- bash -c \"cd '{directory}' && ./config.sh "
        f'--unattended --url https://github.com/{install["repo"]} --token \\"${{{token_var}}}\\" '
        f'--name {install["runner_name"]} --labels {labels} --replace"',
        "fi",
        f"grep -qF ':{LOCAL_BIN}' '{directory}/.path' || "
        f"sed -i 's#$#:{LOCAL_BIN}#' '{directory}/.path'",
        f"[ -f '{directory}/.service' ] || (cd '{directory}' && ./svc.sh install {RUNNER_ACCOUNT})",
        *render_recovery_lines(install),
        *render_kill_mode_lines(install),
        *render_job_hook_lines(install),
        *render_cache_drop_in_lines(install),
        *render_runner_slice_drop_in_lines(install),
        f"(cd '{directory}' && ./svc.sh start)",
        *render_cache_adopt_now_lines(install),
        *render_runner_slice_move_lines(install),
    ]


def render_provision_script(
    spec: HostRunnerSpec, tokens: dict[str, str], reserved_cores: int
) -> str:
    """The provision for one host, run as root.

    Args:
        spec: The host's roster entry.
        tokens: Registration tokens by repository, exported at the head of
            the script; empty when whoever runs it exports them, as the
            render's run order says.
        reserved_cores: The logical cores the host's roster file reserves
            for its runners (``dispatch.budget.reserved_cores``), the size
            of the CI slice's CPU set (:mod:`fleet.core.runner_cpuset_render`).

    Returns:
        The complete ``provision.sh`` text.

    Raises:
        ValueError: When ``reserved_cores`` is below 1, or a token or a
            roster value cannot be embedded
            verbatim.
    """
    lines: list[str] = [
        *SCRIPT_HEADER,
        "# provision.sh, rendered by fleet-runners; run as root after base.sh.",
        "# Re-run after any change: every step is idempotent.",
        *(
            f"export {token_variable(repo)}='{scriptable(token, label='registration token')}'"
            for repo, token in sorted(tokens.items())
        ),
        "",
        "# --- hygiene: ci-clean script, service and daily timer ---",
        "cat > /usr/local/bin/ci-clean <<'CI_CLEAN_EOF'",
        CI_CLEAN_SCRIPT.rstrip("\n"),
        "CI_CLEAN_EOF",
        "chmod +x /usr/local/bin/ci-clean",
        *render_unit_file_lines(
            CI_CLEAN_SERVICE_PATH,
            CI_CLEAN_SERVICE,
            announcement="hygiene unit set: ci-clean.service",
        ),
        *render_unit_file_lines(
            CI_CLEAN_TIMER_PATH, CI_CLEAN_TIMER, announcement="hygiene unit set: ci-clean.timer"
        ),
        "systemctl enable --now ci-clean.timer",
        "",
        "# --- host assets ---",
    ]
    for asset in spec["assets"]:
        if not asset["manual"]:
            lines += render_asset_lines(asset)
    lines += ["", "# --- the runners' share of the host (MCPs board task 45a4f22b) ---"]
    lines += render_slice_unit_lines(spec["ci_slice"])
    lines += render_ci_cpuset_lines(reserved_cores)
    lines += ["", "# --- what a finished job leaves behind (MCPs board task 53528106) ---"]
    lines += render_reaper_lines(spec)
    lines += ["", "# --- each job outside its runner's listener (MCPs board task 715b9ec5) ---"]
    lines += render_job_scope_lines()
    lines += ["", "# --- the actions jobs use, kept on the host (MCPs board task ce2b92e0) ---"]
    lines += render_fill_lines(spec)
    lines += render_adopt_lines(spec)
    lines += ["", "# --- runner installs ---"]
    for install in spec["installs"]:
        lines += render_install_lines(install)
        lines.append("")
    lines += [
        "# --- the first hygiene pass, run now rather than at the timer's next 04:00 ---",
        FIRST_HYGIENE_PASS,
    ]
    return "\n".join(lines) + "\n"


def manual_steps(spec: HostRunnerSpec) -> list[str]:
    """The assets no script may fetch, each a loud line.

    Args:
        spec: The host's roster entry.

    Returns:
        One ``PLACE BY HAND`` line per manual asset.
    """
    return [
        f"PLACE BY HAND: {asset['path']} -- {asset['reason']}. This asset can never be "
        "fetched by a script; place it, then re-run fleet-runners audit."
        for asset in spec["assets"]
        if asset["manual"]
    ]


def render_provision(spec: HostRunnerSpec, reserved_cores: int) -> RenderedProvision:
    """Everything needed to converge a host onto its roster entry.

    Args:
        spec: The host's roster entry.
        reserved_cores: The logical cores its roster file reserves for its
            runners, as :func:`render_provision_script` takes them.

    Returns:
        Both scripts, the provision expecting its tokens exported, and the
        manual steps no script can perform.

    Raises:
        ValueError: As :func:`render_provision_script` describes.
    """
    return RenderedProvision(
        base_script=render_base_script(spec),
        provision_script=render_provision_script(spec, {}, reserved_cores),
        manual_steps=manual_steps(spec),
    )


__all__ = [
    "CI_CLEAN_SERVICE_PATH",
    "CI_CLEAN_TIMER_PATH",
    "LOCAL_BIN",
    "REBOOT_MARKER",
    "REBOOT_REQUIRED_PATH",
    "SCRIPT_HEADER",
    "STAGE_TIMEOUT_SECONDS",
    "RenderedProvision",
    "manual_steps",
    "render_asset_lines",
    "render_base_script",
    "render_install_lines",
    "render_provision",
    "render_provision_script",
]
