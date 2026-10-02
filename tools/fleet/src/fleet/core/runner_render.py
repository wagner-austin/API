"""Rendering the converge scripts for a new CI host.

``fleet-runners render`` writes two artifacts and prints the run order:

  provision.ps1   Windows side, rendered by
                  :mod:`fleet.core.runner_windows_provision`. Writes
                  ``.wslconfig`` when the roster declares a memory floor,
                  registers + starts the keepalive scheduled task when one
                  is declared, and installs every Windows-side runner.
  provision.sh    Linux side, run INSIDE the distro as root. Installs the
                  hygiene units, creates writable assets, clones fetchable
                  ones, and configures + installs every runner instance.

RENDERED, NOT EXECUTED, DELIBERATELY. Every mutating act on a fleet machine
in this workspace is operator-visible before it runs -- the same posture as
the fleet lock and the no-preemptive-shutdown rule -- and two of the inputs
cannot be automated at all: runner registration tokens expire hourly and are
minted by a human in the GitHub UI, and the licensed game tree can never be
fetched by any script. The rendered scripts are exact, not templates: every
value in them came from the roster, and the audit afterwards is the proof
they were run.

THE HYGIENE PAYLOAD LIVES HERE, NOT ON A BOX. ``ci-clean`` ran only on
lavender's disk until 2026-09-08, which made it exactly the folklore this
package exists to end: this module is its source of truth, and the rendered
provision installs byte-identical copies everywhere.
"""

from __future__ import annotations

from typing_extensions import TypedDict

from fleet.contracts.runners import FileAsset, HostRunnerSpec, RunnerInstall
from fleet.core.runner_install import RUNNER_VERSION, install_root_for, token_variable
from fleet.core.runner_reaper_render import render_kill_mode_lines, render_reaper_lines
from fleet.core.runner_recovery import render_wsl_recovery_lines
from fleet.core.runner_slice_render import render_runner_slice_lines, render_slice_unit_lines
from fleet.core.runner_windows_provision import render_windows_provision_script

#: Where ``pipx install`` puts its shims for the runner account, appended to
#: each WSL runner's ``.path`` (see :func:`render_wsl_install_lines`).
LOCAL_BIN = "/home/gharunner/.local/bin"

#: Where WSL mounts ``nvidia-smi`` and the GPU driver libraries from the
#: Windows driver, also appended to each WSL runner's ``.path``. An
#: interactive ``wsl`` shell has it on PATH; a runner configured under
#: ``sudo`` does not, and on the rebuilt lavender Model-Trainer's job died at
#: ``nvidia-smi: command not found`` while the host audit, which ran
#: nvidia-smi from an interactive shell, passed (board task 1aa6a021).
WSL_LIB = "/usr/lib/wsl/lib"

#: The entries every WSL runner's ``.path`` must carry, in append order.
RUNNER_PATH_ENTRIES: tuple[str, ...] = (LOCAL_BIN, WSL_LIB)

#: The daily hygiene script, installed to /usr/local/bin/ci-clean.
#:
#: DOCKER FIRST, AND UNGATED. The Actions runner removes a job's service
#: containers but never their anonymous volumes, and by 2026-09-25 lavender
#: held 8,571 of them, 536 GB, which filled its disk and remounted every
#: runner's root read-only (board task 9e5c7a17). The workflows now mount a
#: tmpfs over each declared volume; this is the backstop for whatever image or
#: job leaks next. It runs ahead of the Runner.Worker gate because Docker
#: counts its own references -- nothing a running container holds is pruned --
#: and because a gated pass skips whenever any one of a host's runners is busy,
#: and lavender has eight.
#:
#: WHAT IT DELIBERATELY KEEPS: hashed poetry venvs whose source project still
#: exists (they are per-package-per-runner working state, not duplicates),
#: everything under any _work tree (in-project venvs persist there by design),
#: named docker volumes, and every tagged image, which is the pull cache.
#: WHAT IT PRUNES: containers stopped for a day, anonymous volumes no container
#: references, dangling images, venvs whose recorded source paths are all
#: gone, poetry wheel artifacts untouched for 60+ days, tmp download fragments
#: a killed job left behind, and the pip cache. The poetry half refuses to run
#: while a Runner.Worker is executing -- a venv mid-use must never race a
#: delete.
CI_CLEAN_SCRIPT = """#!/usr/bin/env bash
# ci-clean: daily hygiene for a CI runner host. Rendered from
# fleet.core.runner_render -- edit THERE, never here.
set -euo pipefail

# Docker, ungated: it never prunes what a running container references.
# `volume prune` takes anonymous volumes only; named ones need --all.
docker container prune -f --filter until=24h
docker volume prune -f
docker image prune -f

if pgrep -f 'Runner.Worker' >/dev/null 2>&1; then
    echo "ci-clean: a runner job is executing; skipping this pass."
    exit 0
fi

VENVS="$HOME/.cache/pypoetry/virtualenvs"
ARTIFACTS="$HOME/.cache/pypoetry/artifacts"

reclaimed_kb=0

if [ -d "$VENVS" ]; then
    for env in "$VENVS"/*/; do
        [ -d "$env" ] || continue
        mapfile -t srcs < <(cat "$env"lib/python*/site-packages/*.pth 2>/dev/null \\
            | grep -E '^/' | sort -u)
        [ "${#srcs[@]}" -gt 0 ] || continue  # unknown provenance: keep
        alive=0
        for s in "${srcs[@]}"; do
            [ -e "$s" ] && alive=1 && break
        done
        if [ "$alive" -eq 0 ]; then
            size_kb=$(du -sk "$env" | cut -f1)
            reclaimed_kb=$((reclaimed_kb + size_kb))
            echo "ci-clean: orphan venv $(basename "$env") (${size_kb}KB)"
            rm -rf "$env"
        fi
    done
fi

if [ -d "$ARTIFACTS" ]; then
    before_kb=$(du -sk "$ARTIFACTS" | cut -f1)
    find "$ARTIFACTS" -type f -name 'tmp*' -mmin +1440 -delete 2>/dev/null || true
    find "$ARTIFACTS" -type f -mtime +60 -delete 2>/dev/null || true
    find "$ARTIFACTS" -type d -empty -delete 2>/dev/null || true
    after_kb=$(du -sk "$ARTIFACTS" | cut -f1)
    reclaimed_kb=$((reclaimed_kb + before_kb - after_kb))
fi

if [ -d "$HOME/.cache/pip" ]; then
    pip_kb=$(du -sk "$HOME/.cache/pip" | cut -f1)
    rm -rf "$HOME/.cache/pip"
    reclaimed_kb=$((reclaimed_kb + pip_kb))
fi

echo "ci-clean: reclaimed $((reclaimed_kb / 1024))MB this pass."
"""

#: systemd unit for the hygiene pass.
CI_CLEAN_SERVICE = """[Unit]
Description=CI runner host hygiene (docker leftovers, poetry orphans, stale artifacts, pip cache)

[Service]
Type=oneshot
User=gharunner
ExecStart=/usr/local/bin/ci-clean
"""

#: systemd timer for the hygiene pass: daily, because a week of a leak
#: bounded only by the disk is what filled lavender.
CI_CLEAN_TIMER = """[Unit]
Description=Daily CI hygiene pass

[Timer]
OnCalendar=*-*-* 04:00
Persistent=true

[Install]
WantedBy=timers.target
"""


class RenderedProvision(TypedDict):
    """Everything ``fleet-runners render`` produces for one host.

    Attributes:
        windows_script: ``provision.ps1``, run on the Windows host first.
        linux_script: ``provision.sh``, run inside the distro as root next.
        manual_steps: One line per asset no script can fetch, printed loudly.
            Empty only for a host with no such assets.
    """

    windows_script: str
    linux_script: str
    manual_steps: list[str]


def _render_asset_lines(asset: FileAsset) -> list[str]:
    """Provision lines for one fetchable asset.

    Args:
        asset: The asset, with ``manual`` False. The contract guarantees a
            non-writable one carries its own ``provision_command``.

    Returns:
        The bash lines that create it, idempotently: an asset already
        present is left alone, because the fetch belongs to first provision
        and drift afterwards is the audit's to report.
    """
    if asset["writable"]:
        return [
            f"mkdir -p '{asset['path']}'",
            f"chown gharunner:gharunner '{asset['path']}'",
        ]
    command = asset["provision_command"]
    return [
        f"if [ ! -e '{asset['path']}' ]; then",
        f"    {command}",
        "fi",
    ]


def render_wsl_install_lines(install: RunnerInstall) -> list[str]:
    """Bash lines that install one WSL-side runner.

    The install directory comes from the roster's own ``workdir`` -- the
    ``_work`` parent -- not from any derived name: an earlier version
    numbered directories by loop index, which would have installed to a
    path the roster never declared, and the audit would then have verified
    the declaration while the runner lived somewhere else.

    ``--replace`` takes back a registration of the same name, as on the
    Windows side. :data:`RUNNER_PATH_ENTRIES` go onto the runner's ``.path``
    AFTER config.sh, which writes that file, and BEFORE the service starts,
    because runsvc.sh reads it once at start and exports it as every job
    step's PATH, and a systemd-started runner has no login shell to add
    anything. ``~/.local/bin`` is where ``pipx install poetry`` puts its
    shims: without it, a job that installs poetry dies one step later at
    ``poetry: command not found``, exit 127 (measured on lavender; MCPs
    scripts/ops/provision-wsl-runner.ps1 carried it by hand).
    ``/usr/lib/wsl/lib`` is where WSL mounts nvidia-smi (see
    :data:`WSL_LIB`).

    Args:
        install: The install to configure, with ``side`` ``"wsl"``.

    An install whose ``.runner`` file exists is already configured and
    config.sh is skipped, and one whose ``.service`` file exists already
    has its unit and ``svc.sh install`` is skipped: both refuse a second
    run, and ``--rebuild`` re-runs every stage over a host that may be
    half built. ``svc.sh start`` always runs; starting a running unit is
    a no-op.

    Returns:
        The bash lines that download, configure and start it.
    """
    token_var = token_variable(install["repo"])
    directory = install_root_for(install)
    labels = ",".join(install["labels"])
    return [
        f': "${{{token_var}:?set {token_var} to a fresh registration token for '
        f'{install["repo"]} (Settings > Actions > Runners > New self-hosted runner)}}"',
        f"mkdir -p {directory}",
        f"if [ ! -f {directory}/config.sh ]; then",
        f"    curl -fsSL -o {directory}/runner.tar.gz "
        f"https://github.com/actions/runner/releases/download/v{RUNNER_VERSION}/"
        f"actions-runner-linux-x64-{RUNNER_VERSION}.tar.gz",
        f"    tar -xzf {directory}/runner.tar.gz -C {directory}",
        f"    rm {directory}/runner.tar.gz",
        "fi",
        f"chown -R gharunner:gharunner {directory}",
        f"if [ ! -f {directory}/.runner ]; then",
        f'    sudo -u gharunner bash -c "cd {directory} && ./config.sh --unattended '
        f'--url https://github.com/{install["repo"]} --token \\"${{{token_var}}}\\" '
        f'--name {install["runner_name"]} --labels {labels} --replace"',
        "fi",
        *(
            f"grep -qF ':{entry}' {directory}/.path || sed -i 's#$#:{entry}#' {directory}/.path"
            for entry in RUNNER_PATH_ENTRIES
        ),
        f"[ -f {directory}/.service ] || (cd {directory} && ./svc.sh install gharunner)",
        *render_wsl_recovery_lines(install),
        *render_kill_mode_lines(install),
        f"(cd {directory} && ./svc.sh start)",
        *render_runner_slice_lines(install),
    ]


def _render_linux_script(spec: HostRunnerSpec) -> str:
    """The in-distro provision for one host.

    Args:
        spec: The host's roster entry.

    Returns:
        The complete ``provision.sh`` text, to run inside the distro as root.
    """
    lines: list[str] = [
        "#!/usr/bin/env bash",
        "# provision.sh -- Linux side. Rendered by fleet-runners; run inside the",
        f"# distro as root: wsl -d {spec['wsl_distro']} -- bash /mnt/c/.../provision.sh",
        "# Re-run after any change: every step is idempotent.",
        "set -euo pipefail",
        "",
        "id gharunner >/dev/null 2>&1 || useradd -m gharunner",
        "command -v pipx >/dev/null 2>&1 || { apt-get update && apt-get install -y pipx; }",
        "",
        "# --- hygiene: ci-clean script, service and daily timer ---",
        "cat > /usr/local/bin/ci-clean <<'CI_CLEAN_EOF'",
        CI_CLEAN_SCRIPT.rstrip("\n"),
        "CI_CLEAN_EOF",
        "chmod +x /usr/local/bin/ci-clean",
        "cat > /etc/systemd/system/ci-clean.service <<'UNIT_EOF'",
        CI_CLEAN_SERVICE.rstrip("\n"),
        "UNIT_EOF",
        "cat > /etc/systemd/system/ci-clean.timer <<'TIMER_EOF'",
        CI_CLEAN_TIMER.rstrip("\n"),
        "TIMER_EOF",
        "systemctl daemon-reload",
        "systemctl enable --now ci-clean.timer",
        "",
        "# --- host assets ---",
    ]
    for asset in spec["assets"]:
        if not asset["manual"]:
            lines += _render_asset_lines(asset)
    lines += ["", "# --- the runners' share of the VM (MCPs board task 45a4f22b) ---"]
    lines += render_slice_unit_lines(spec["ci_slice"])
    lines += ["", "# --- what a finished job leaves behind (MCPs board task 53528106) ---"]
    lines += render_reaper_lines()
    lines += ["", "# --- runner installs ---"]
    # wsl-side installs only: windows-side ones are provision.ps1's, in
    # their own execution environment.
    for install in (i for i in spec["installs"] if i["side"] == "wsl"):
        lines += render_wsl_install_lines(install)
        lines.append("")
    return "\n".join(lines) + "\n"


def render_provision(spec: HostRunnerSpec) -> RenderedProvision:
    """Everything needed to converge a host onto its roster entry.

    Args:
        spec: The host's roster entry.

    Returns:
        Both scripts and the manual steps no script can perform.
    """
    manual = [
        f"PLACE BY HAND: {asset['path']} -- {asset['reason']}. This asset can never be "
        "fetched by a script; place it, then re-run fleet-runners audit."
        for asset in spec["assets"]
        if asset["manual"]
    ]
    return RenderedProvision(
        windows_script=render_windows_provision_script(spec, {}),
        linux_script=_render_linux_script(spec),
        manual_steps=manual,
    )


__all__ = [
    "CI_CLEAN_SCRIPT",
    "CI_CLEAN_SERVICE",
    "CI_CLEAN_TIMER",
    "LOCAL_BIN",
    "RUNNER_PATH_ENTRIES",
    "WSL_LIB",
    "RenderedProvision",
    "render_provision",
    "render_wsl_install_lines",
]
