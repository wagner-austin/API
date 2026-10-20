"""Each Windows runner's Python tool cache, as lines of the host's install and its audit.

MCPs board task 1e9ab738. On self-hosted Windows, ``actions/setup-python``
INSTALLS a version its tool cache misses, through the python.org installer,
whose products Windows Installer registers machine-wide under one product
code per version. loki runs one runner per repository, each with its own
tool cache under its own ``_work``, so two runners setting up one version
install one product into two directories: measured on loki on 2026-10-20,
chat's and corvis-stick's reruns both set up 3.11.9 at once, setup.ps1
removed the other's registry entries before its own install, and chat's
failed with "Error happened during Python installation". Seeding the cache
turns setup-python's install path into its find path, which reads only
``Python/<version>/x64`` and an ``x64.complete`` marker beside it.

THE SEED IS THE NUGET CPYTHON PACKAGE, lifted from the Windows provision
0531d9be2 deleted with lavender's Windows side: a whole xcopy-deployable
Python with no installer and no registry. It carries pip as a module but no
``Scripts`` directory, and setup-python puts ``Scripts`` on PATH, so a
workflow's bare ``pip`` fails on a seed that stops at the copy (measured on
MCPs run 36254365786, session-audit: CommandNotFoundException for pip). pip
is therefore reinstalled from the package's own bundled wheel, offline,
which writes the entry points, and ``Scripts\\pip.exe`` is both the
idempotency test and the verification.

EVERY EFFECT IS A PARAMETER: the package's URL, which the Pester suite
serves as a ``file://`` archive it built, and python.exe's file name, which
it points at a stand-in. :mod:`fleet.core.runner_windows_render` places the
lines inside its loop over the installs, before each runner's service lines.
"""

from __future__ import annotations

from fleet.contracts.runner_windows import WindowsRunnerInstall
from fleet.core.runner_audit_types import ExpectedCheck

#: The NuGet CPython package, ``{0}`` the exact version.
PYTHON_PACKAGE_URL = "https://www.nuget.org/api/v2/package/python/{0}"

#: Why each tool cache row matters.
SEED_REASON = (
    "setup-python installs a version its tool cache misses with the python.org installer, "
    "whose products are registered machine-wide, so two of loki's runners setting up one "
    "version break each other's install (chat's 3.11.9, 2026-10-20)"
)

#: The param-block lines both scripts take, each ending in a comma.
PARAMETER_LINES: tuple[str, ...] = (
    f"    [string]$PythonPackageUrl = '{PYTHON_PACKAGE_URL}',",
    "    [string]$PythonName = 'python.exe',",
)

#: The install's lines for one runner, inside its loop over ``$Installs``,
#: whose ``$root`` is the runner's directory.
INSTALL_LINES: tuple[str, ...] = (
    "    $tools = Join-Path $root '_work\\_tool'",
    "    foreach ($version in @($install.Python)) {",
    '        $target = Join-Path $tools "Python\\$version\\x64"',
    "        if (-not (Test-Path -LiteralPath (Join-Path $target 'Scripts\\pip.exe'))) {",
    "            $work = Join-Path $env:TEMP ([guid]::NewGuid().ToString('N'))",
    "            [void][IO.Directory]::CreateDirectory($work)",
    "            $package = Join-Path $work 'python.nupkg.zip'",
    "            Invoke-WebRequest -Uri ($PythonPackageUrl -f $version) -OutFile $package "
    "-UseBasicParsing",
    "            Expand-Archive -LiteralPath $package -DestinationPath $work -Force",
    "            [void][IO.Directory]::CreateDirectory($target)",
    "            Copy-Item -Path (Join-Path $work 'tools\\*') -Destination $target -Recurse -Force",
    "            $wheel = @(Get-ChildItem -LiteralPath "
    "(Join-Path $target 'Lib\\ensurepip\\_bundled') -Filter 'pip-*.whl')",
    "            if ($wheel.Count -ne 1) {",
    '                throw "RUNNER_PYTHON_SEED_FAILED: $version holds $($wheel.Count) bundled pip '
    'wheels, not one"',
    "            }",
    "            & (Join-Path $target $PythonName) -m pip install --force-reinstall --no-deps "
    "--no-index --no-warn-script-location --disable-pip-version-check $wheel[0].FullName",
    "            if ($LASTEXITCODE -ne 0) {",
    '                throw "RUNNER_PYTHON_SEED_FAILED: the pip reinstall in $version exited '
    '$LASTEXITCODE"',
    "            }",
    "            if (-not (Test-Path -LiteralPath (Join-Path $target 'Scripts\\pip.exe'))) {",
    '                throw "RUNNER_PYTHON_SEED_FAILED: the pip reinstall in $version wrote no '
    'Scripts\\pip.exe"',
    "            }",
    "            [void](New-Item -ItemType File -Force -Path "
    '(Join-Path $tools "Python\\$version\\x64.complete"))',
    "            Remove-Item -LiteralPath $work -Recurse -Force",
    '            "python: seeded $version into $tools for $($install.Repo)"',
    "        }",
    "    }",
)

#: The audit's rows for one runner, inside its loop over ``$Installs``.
AUDIT_LINES: tuple[str, ...] = (
    "    $tools = Join-Path $install.Root '_work\\_tool'",
    "    foreach ($version in @($install.Python)) {",
    '        $pip = Join-Path $tools "Python\\$version\\x64\\Scripts\\pip.exe"',
    '        $complete = Join-Path $tools "Python\\$version\\x64.complete"',
    "        if ((Test-Path -LiteralPath $pip) -and (Test-Path -LiteralPath $complete)) {",
    '            "CHECK python:${service}:$version OK"',
    "        } else {",
    '            "CHECK python:${service}:$version DRIFT the tool cache lacks $pip or $complete"',
    "        }",
    "    }",
)


def expected_checks(install: WindowsRunnerInstall) -> list[ExpectedCheck]:
    """The tool cache rows the audit prints for one runner, in order.

    Args:
        install: The runner.

    Returns:
        One ``python:<service>:<version>`` row per declared version.
    """
    return [
        ExpectedCheck(check_id=f"python:{install['service']}:{version}", reason=SEED_REASON)
        for version in install["python_toolcache"]
    ]


__all__ = [
    "AUDIT_LINES",
    "INSTALL_LINES",
    "PARAMETER_LINES",
    "PYTHON_PACKAGE_URL",
    "SEED_REASON",
    "expected_checks",
]
