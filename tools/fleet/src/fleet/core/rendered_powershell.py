"""Every PowerShell script tools/fleet renders, named as its committed copy is.

These scripts are shipped verbatim to Windows hosts and run there with
``powershell -File``, so they are code under the operator's PowerShell rule
(MCPs board task d69786fa, A2): executed by a Pester suite under MCPs'
harness, not string-matched. The harness finds scripts through
``git ls-files``, so each render is committed under ``rendered/`` as
``<name>.ps1``: ``make render-powershell`` writes them from this registry
(:mod:`fleet.cli.render_powershell`), ``tests/test_rendered_powershell.py``
fails when a committed copy differs from its renderer's output or has no
entry here, and ``tests/pester`` executes each copy to every command and
branch. Nothing is added to the harness to see them: a committed render is a
tracked script like any other, and one no suite runs is PS-NO-SUITE.

Scripts that depend on a host are rendered from the real roster,
``runners.json``, once per host, so the committed copy is the text that host
receives. Scripts that depend on a dispatch are rendered for one example
dispatch (:data:`EXAMPLE_STAGE_ROOT`, :data:`EXAMPLE_RUN_ID`): every location
they read is a parameter defaulting to the rendered path, so the suite runs
the committed text against a directory it laid out, and a real dispatch's
copy differs from the example only in those defaults.
"""

from __future__ import annotations

from typing_extensions import TypedDict

from fleet.contracts.node import NodePlatform
from fleet.contracts.runners import RunnerSpec
from fleet.core import (
    dialect,
    names,
    retire,
    runner_audit,
    runner_base_render,
    runner_distro,
    runner_load,
    runner_rebuild,
    runner_windows_provision,
    verdict,
)
from fleet.core.dialect_windows import WindowsDialect

#: Where the committed renders live, relative to tools/fleet.
RENDERED_DIRECTORY = "rendered"

#: The example dispatch's stage root: the Windows nodes' own.
EXAMPLE_STAGE_ROOT = "C:/fleet/stage"

#: The example dispatch's id, in the shape fleet-run mints.
EXAMPLE_RUN_ID = "MCPs-packages-maketools-1790000000"

#: The commit the example dispatch's companion was archived from.
EXAMPLE_COMPANION_SHA = "5f389cb3d9bdd2e9b49e8df6683a6fee71a359b2"

#: The example dispatch's project, inside its export.
EXAMPLE_PROJECT = "packages/maketools"

#: The test workers the example dispatch was granted.
EXAMPLE_WORKERS = 4

#: The example dispatch's install steps.
EXAMPLE_INSTALL: tuple[tuple[str, ...], ...] = (("npm", "ci"),)

#: The payload stem the example distro driver runs: the rebuild's Linux base.
EXAMPLE_DISTRO_STEM = "fleet-rebuild-linux-base"


class RenderedScript(TypedDict):
    """One rendered script and the name its committed copy carries.

    Attributes:
        name: The file stem under :data:`RENDERED_DIRECTORY`.
        text: The complete script, exactly as a host receives it.
    """

    name: str
    text: str


def _dialect_scripts() -> list[RenderedScript]:
    """The Windows dialect's scripts, for the example dispatch.

    Returns:
        One entry per script, named ``dialect-<act>``.
    """
    spoken = WindowsDialect()
    target = names.dispatch_directory(EXAMPLE_STAGE_ROOT, EXAMPLE_RUN_ID)
    companion = names.companion_directory(EXAMPLE_STAGE_ROOT, "MCPs")
    return [
        RenderedScript(name="dialect-capacity-probe", text=spoken.capacity_probe_script()),
        RenderedScript(name="dialect-toolchain-probe", text=spoken.toolchain_probe_script()),
        RenderedScript(name="dialect-observe-sessions", text=spoken.observe_sessions_script()),
        RenderedScript(name="dialect-make-directory", text=spoken.make_directory_script(target)),
        RenderedScript(
            name="dialect-reset-directory", text=spoken.reset_directory_script(companion)
        ),
        RenderedScript(name="dialect-digest", text=spoken.digest_script(target)),
        RenderedScript(name="dialect-result", text=spoken.result_script(target)),
        RenderedScript(
            name="dialect-log-tail",
            text=spoken.log_tail_script(target, verdict.LOG_TAIL_LINES),
        ),
        RenderedScript(
            name="dialect-launch", text=spoken.launch_script(target=target, run_id=EXAMPLE_RUN_ID)
        ),
        RenderedScript(
            name="dialect-stop", text=spoken.stop_script(target=target, run_id=EXAMPLE_RUN_ID)
        ),
        RenderedScript(
            name="dialect-retire",
            text=retire.script_for(
                NodePlatform.WINDOWS, stage_root=EXAMPLE_STAGE_ROOT, run_id=EXAMPLE_RUN_ID
            ),
        ),
        RenderedScript(
            name="dialect-extract",
            text=spoken.checked_script(
                dialect.extract_commands(f"{target}/{names.ARCHIVE_NAME}", target)
            ),
        ),
        RenderedScript(
            name="dialect-init-repository",
            text=spoken.checked_script(dialect.init_repository_commands(target, EXAMPLE_RUN_ID)),
        ),
        RenderedScript(
            name="dialect-companion-repository",
            text=spoken.checked_script(
                dialect.companion_repository_commands(companion, EXAMPLE_COMPANION_SHA)
            ),
        ),
        RenderedScript(
            name="dialect-build",
            text=spoken.build_script(
                target=target,
                path=EXAMPLE_PROJECT,
                workers=EXAMPLE_WORKERS,
                install=EXAMPLE_INSTALL,
                cache_root=names.cache_root(EXAMPLE_STAGE_ROOT),
                isolated_docker=False,
            ),
        ),
    ]


def render_all(roster: RunnerSpec) -> list[RenderedScript]:
    """Render every script, in a stable order.

    Args:
        roster: The runner roster the host-specific scripts are rendered for.

    Returns:
        One entry per committed file.

    Raises:
        ValueError: When two entries would share a file name, or a roster
            value cannot be embedded verbatim.
    """
    scripts = [
        RenderedScript(
            name="rebuild-boot-instant", text=runner_rebuild.render_boot_instant_script()
        ),
        RenderedScript(name="rebuild-restart", text=runner_rebuild.render_restart_script()),
        RenderedScript(name="load-forest", text=runner_load.WINDOWS_FOREST_SCRIPT),
        *_dialect_scripts(),
    ]
    for host in roster["hosts"]:
        scripts += [
            RenderedScript(
                name=f"rebuild-terminate-{host['name']}",
                text=runner_rebuild.render_terminate_script(host["wsl_distro"]),
            ),
            RenderedScript(
                name=f"base-windows-{host['name']}",
                text=runner_base_render.render_windows_base_script(host),
            ),
            RenderedScript(
                name=f"base-import-{host['name']}",
                text=runner_base_render.render_import_script(host),
            ),
            RenderedScript(
                name=f"audit-{host['name']}", text=runner_audit.render_audit_script(host)
            ),
            RenderedScript(
                name=f"distro-driver-{host['name']}",
                text=runner_distro.render_distro_driver(host, EXAMPLE_DISTRO_STEM),
            ),
            RenderedScript(
                name=f"provision-windows-{host['name']}",
                text=runner_windows_provision.render_windows_provision_script(host, {}),
            ),
        ]
        windows_installs = [i for i in host["installs"] if i["side"] == "windows"]
        if windows_installs:
            scripts.append(
                RenderedScript(
                    name=f"onboard-windows-{host['name']}",
                    text=runner_windows_provision.render_windows_onboard_script(
                        windows_installs[0], {}
                    ),
                )
            )
    names = [script["name"] for script in scripts]
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ValueError(f"two rendered scripts would share a file name: {', '.join(repeated)}")
    return scripts


__all__ = [
    "EXAMPLE_COMPANION_SHA",
    "EXAMPLE_DISTRO_STEM",
    "EXAMPLE_INSTALL",
    "EXAMPLE_PROJECT",
    "EXAMPLE_RUN_ID",
    "EXAMPLE_STAGE_ROOT",
    "EXAMPLE_WORKERS",
    "RENDERED_DIRECTORY",
    "RenderedScript",
    "render_all",
]
