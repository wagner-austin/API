"""Rendering the converge scripts for a new CI host.

``fleet-runners render`` writes two artifacts and prints the run order:

  provision.ps1   Windows side. Writes ``.wslconfig`` when the roster
                  declares a memory floor, and registers + starts the
                  keepalive scheduled task when one is declared.
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

#: The GitHub Actions runner release the rendered provision installs.
#:
#: Pinned so two hosts provisioned a week apart run the same agent; measured
#: as the version the live lavender installs run (2026-09-08). Bumping it is
#: a one-line change reviewed like any other.
RUNNER_VERSION = "2.337.0"

#: The weekly hygiene script, installed to /usr/local/bin/ci-clean.
#:
#: WHAT IT DELIBERATELY KEEPS: hashed poetry venvs whose source project still
#: exists (they are per-package-per-runner working state, not duplicates), and
#: everything under any _work tree (in-project venvs persist there by design).
#: WHAT IT PRUNES: venvs whose recorded source paths are all gone, poetry
#: wheel artifacts untouched for 60+ days, tmp download fragments a killed job
#: left behind, and the pip cache. It refuses to run while a Runner.Worker is
#: executing -- a venv mid-use must never race a delete.
CI_CLEAN_SCRIPT = """#!/usr/bin/env bash
# ci-clean: weekly hygiene for a CI runner host. Rendered from
# fleet.core.runner_render -- edit THERE, never here.
set -euo pipefail

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
Description=CI runner host hygiene (poetry orphans, stale artifacts, pip cache)

[Service]
Type=oneshot
User=gharunner
ExecStart=/usr/local/bin/ci-clean
"""

#: systemd timer for the hygiene pass.
CI_CLEAN_TIMER = """[Unit]
Description=Weekly CI hygiene pass

[Timer]
OnCalendar=Sun 04:00
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


def token_variable(repo: str) -> str:
    """The environment variable a repo's registration token arrives in.

    Args:
        repo: The repository, ``owner/name`` form.

    Returns:
        The variable name, e.g. ``RUNNER_TOKEN_API`` for ``wagner-austin/API``.
    """
    name = repo.split("/")[1]
    sanitized = "".join(c if c.isalnum() else "_" for c in name.upper())
    return f"RUNNER_TOKEN_{sanitized}"


def render_windows_install_lines(install: RunnerInstall) -> list[str]:
    """PowerShell lines that install one WINDOWS-side runner.

    Lifted from the tree-bot install of 2026-09-09 (the last hand-rolled
    runner on this fleet, by operator mandate) and from the layout its
    siblings already share: an install directory beside
    ``C:\\actions-runner`` named for the repo, the pinned runner release,
    and ``config.cmd --runasservice`` so the service survives reboots.

    Args:
        install: The install to provision, with ``side`` ``"windows"``.

    Returns:
        The PowerShell lines. The registration token arrives in the same
        per-repo environment variable the bash side uses; config.cmd
        refuses a repeat registration loudly, which is the correct answer
        to running this twice.
    """
    token_var = token_variable(install["repo"])
    directory = _install_root_for(install)
    labels = ",".join(install["labels"])
    return [
        f"if (-not $env:{token_var}) {{ throw 'set {token_var} to a fresh registration "
        f"token for {install['repo']}' }}",
        f"New-Item -ItemType Directory -Force -Path '{directory}' | Out-Null",
        f"if (-not (Test-Path '{directory}\\config.cmd')) {{",
        f"    $zip = '{directory}\\runner.zip'",
        "    Invoke-WebRequest -Uri "
        f"'https://github.com/actions/runner/releases/download/v{RUNNER_VERSION}/"
        f"actions-runner-win-x64-{RUNNER_VERSION}.zip' -OutFile $zip",
        f"    Expand-Archive -LiteralPath $zip -DestinationPath '{directory}' -Force",
        "    Remove-Item $zip",
        "}",
        f"& '{directory}\\config.cmd' --unattended "
        f"--url https://github.com/{install['repo']} --token $env:{token_var} "
        f"--name {install['runner_name']} --labels {labels} --runasservice",
    ]


def render_windows_python_toolcache_lines(
    install: RunnerInstall, python_versions: tuple[str, ...]
) -> list[str]:
    """PowerShell lines that seed a Windows runner's Python tool cache.

    On self-hosted Windows, ``actions/setup-python`` INSTALLS when the tool
    cache misses -- via the python.org all-users installer, whose registry
    writes the runner's service account rightly lacks (measured on
    tree-bot run 34394258775: SecurityException in Remove-Item Registry).
    Seeding the cache turns setup-python's install path into its find
    path. The source is the NuGet CPython package: full xcopy-deployable
    Python with pip, no installer, no registry -- exactly the property a
    service-account tool cache needs. The find path wants only
    ``Python/<ver>/x64/python.exe`` plus an ``x64.complete`` marker.

    Args:
        install: The windows-side install whose tool cache to seed.
        python_versions: Exact versions to seed, e.g. ``("3.11.9",)``.
            Exact on purpose: NuGet carries only versions python.org built
            for Windows, and a floating spec here would re-create the
            drift this module exists to prevent.

    Returns:
        The PowerShell lines. Idempotent: an already-seeded version is
        left alone.
    """
    root = _install_root_for(install)
    lines: list[str] = [
        f"$ToolDir = '{root}\\_work\\_tool'",
    ]
    for version in python_versions:
        lines += [
            f"$Target = Join-Path $ToolDir 'Python\\{version}\\x64'",
            "if (-not (Test-Path (Join-Path $Target 'python.exe'))) {",
            "    $Work = Join-Path $env:TEMP ([guid]::NewGuid().ToString('N'))",
            "    New-Item -ItemType Directory -Path $Work | Out-Null",
            "    $Zip = Join-Path $Work 'python.nupkg.zip'",
            "    Invoke-WebRequest -Uri "
            f"'https://www.nuget.org/api/v2/package/python/{version}' "
            "-OutFile $Zip -UseBasicParsing",
            "    Expand-Archive -LiteralPath $Zip -DestinationPath $Work -Force",
            "    New-Item -ItemType Directory -Force -Path $Target | Out-Null",
            "    Copy-Item -Path (Join-Path $Work 'tools\\*') -Destination $Target -Recurse -Force",
            "    & (Join-Path $Target 'python.exe') -m pip --version",
            f"    if ($LASTEXITCODE -ne 0) {{ throw 'pip missing in seeded {version}' }}",
            "    New-Item -ItemType File -Force -Path "
            f"(Join-Path $ToolDir 'Python\\{version}\\x64.complete') | Out-Null",
            "    Remove-Item -Recurse -Force $Work",
            "}",
        ]
    return lines


def _install_root_for(install: RunnerInstall) -> str:
    """The install directory an install's workdir sits under.

    Args:
        install: The install.

    Returns:
        The parent of the ``_work`` tree, backslashed for a windows-side
        install (command lines and cmdlets both take that form) and POSIX
        for a wsl one.
    """
    root = install["workdir"].rsplit("/", 1)[0]
    if install["side"] == "windows":
        return root.replace("/", "\\")
    return root


def _render_windows_script(spec: HostRunnerSpec) -> str:
    """The Windows-side provision for one host.

    Args:
        spec: The host's roster entry.

    Returns:
        The complete ``provision.ps1`` text: the ``.wslconfig`` floor and
        keepalive task when declared, then every windows-side runner
        install. Documents emptiness rather than being omitted when the
        roster declares none of those, so the run order the CLI prints
        holds for every host.
    """
    lines: list[str] = [
        "# provision.ps1 -- Windows side. Rendered by fleet-runners; run as the",
        "# machine's interactive user. Re-run after any change: every step is",
        "# idempotent except runner registration, which config.cmd refuses to",
        "# repeat -- loudly, which is the correct answer to running it twice.",
        "$ErrorActionPreference = 'Stop'",
    ]
    floor = spec["wslconfig_min_memory_gb"]
    if floor is not None:
        lines += [
            "$WslConfig = @(",
            "  '[wsl2]',",
            f"  'memory={floor}GB',",
            "  'swap=8GB'",
            ")",
            'Set-Content -LiteralPath "$env:USERPROFILE\\.wslconfig" '
            "-Value $WslConfig -Encoding ascii",
            "Write-Output 'wrote .wslconfig; the ceiling applies after wsl --shutdown'",
        ]
    keepalive = spec["keepalive_task"]
    if keepalive is not None:
        lines += [
            f"schtasks /create /f /tn '{keepalive}' /sc onstart "
            f"/tr 'wsl -d {spec['wsl_distro']} -- sleep infinity' /rl highest",
            f"schtasks /run /tn '{keepalive}'",
            f"Write-Output 'keepalive task {keepalive} registered and started'",
        ]
    windows_installs = [i for i in spec["installs"] if i["side"] == "windows"]
    for install in windows_installs:
        lines.append("")
        lines += render_windows_install_lines(install)
    if floor is None and keepalive is None and not windows_installs:
        lines.append("Write-Output 'nothing declared for the Windows side of this host'")
    return "\n".join(lines) + "\n"


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

    Args:
        install: The install to configure, with ``side`` ``"wsl"``.

    Returns:
        The bash lines that download, configure and start it.
    """
    token_var = token_variable(install["repo"])
    directory = _install_root_for(install)
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
        f'sudo -u gharunner bash -c "cd {directory} && ./config.sh --unattended '
        f'--url https://github.com/{install["repo"]} --token \\"${{{token_var}}}\\" '
        f'--name {install["runner_name"]} --labels {labels}"',
        f"(cd {directory} && ./svc.sh install gharunner && ./svc.sh start)",
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
        "# --- hygiene: ci-clean script, service and weekly timer ---",
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
        windows_script=_render_windows_script(spec),
        linux_script=_render_linux_script(spec),
        manual_steps=manual,
    )


__all__ = [
    "CI_CLEAN_SCRIPT",
    "CI_CLEAN_SERVICE",
    "CI_CLEAN_TIMER",
    "RUNNER_VERSION",
    "RenderedProvision",
    "render_provision",
    "render_windows_install_lines",
    "render_windows_python_toolcache_lines",
    "render_wsl_install_lines",
    "token_variable",
]
