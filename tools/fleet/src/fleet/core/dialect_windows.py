"""The PowerShell dialect: every script a Windows node is handed.

These were the ONLY scripts until 2026-09-20, spread over the launch, staging,
probe, toolchain and cancel modules; they moved here unchanged in text when
the Linux dialect arrived, so that the two could be read side by side and
held to one protocol. Each method's docstring keeps the incident its text was
written against, because every one of these was found by reading a task's XML
or a transcript off a node rather than by reasoning about the string.

THE ONE RULE, AND IT COST A DISPATCH TO RELEARN. :mod:`fleet.core.remote`
forbids interpolating a remote command, because quotes do not survive the
trip through the local shell, ssh and ``cmd`` into ``powershell``. That rule
was applied to ssh and then broken one layer further in: the first launch
passed the whole build to ``New-ScheduledTaskAction -Argument '-Command "cd
''{path}''; ..."'``, and PowerShell ended the single-quoted argument at the
first inner quote. So the build is its OWN FILE, sent and run by path, and the
registration interpolates one path and no code.
"""

from __future__ import annotations

from fleet.core import names, windows_build, windows_task
from fleet.core.powershell_text import STRICT_HEADER, system32_parameter
from fleet.core.script_values import scriptable
from fleet.core.windows_log_tail import windows_log_tail_script
from fleet.core.windows_toolchain_probe import TOOLCHAIN_PROBE_SCRIPT

#: How the node is asked to run a script file it has just been handed.
#:
#: ``-NoProfile`` because a profile is the node owner's, and a dispatch that
#: inherited it would run different code on different machines for reasons
#: nobody recorded. ``-ExecutionPolicy Bypass`` because the script arrived over
#: ssh and is unsigned by construction.
POWERSHELL_INVOCATION = ("powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File")

#: How a script's body is written on the far side.
#:
#: Streamed from stdin rather than passed as an argument, so no shell between
#: here and the disk can interpret it. ``-LiteralPath`` because a path is a
#: path: without it PowerShell treats ``[`` and ``]`` as wildcards.
#:
#: THE OUTER QUOTES ARE LOAD-BEARING AND THEIR ABSENCE WAS A REAL BUG. Windows
#: OpenSSH hands a remote command to ``cmd.exe``, not to PowerShell. Unquoted,
#: cmd sees the ``|`` as ITS OWN pipe, runs ``powershell -Command $input``, and
#: pipes the result to ``Set-Content`` -- which cmd does not have. Measured
#: 2026-09-04 on the first real dispatch: ``'Set-Content' is not recognized as
#: an internal or external command``. Quoting makes the whole thing one
#: argument to powershell.
#:
#: The directory is created in the same command because the alternative is a
#: second round trip that would itself need a path to already exist. Paths
#: given to this must be ABSOLUTE and literal: a ``$env:TEMP`` inside the
#: single quotes below is not expanded by PowerShell, so it would create a
#: directory named ``$env:TEMP`` rather than resolving one.
WRITE_COMMAND = (
    'powershell -NoProfile -Command "'
    "New-Item -ItemType Directory -Force -Path (Split-Path -Parent '{path}') | Out-Null; "
    "$input | Set-Content -LiteralPath '{path}' -Encoding utf8"
    '"'
)

#: The capacity probe, verbatim. Nothing is substituted into it by this
#: package, so there is no value that could carry a quote into a shell. The
#: braces are PowerShell's own format operator, evaluated on the far side.
#:
#: KEY=VALUE LINES, not JSON: the node renders with string formatting and a
#: malformed number would produce malformed JSON that fails with a parser
#: message instead of a field name. Not positional, because a reordered
#: script would silently swap two numbers.
CAPACITY_PROBE_SCRIPT = """\
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$os = Get-CimInstance Win32_OperatingSystem
$drive = Get-PSDrive C
"free_ram_gb={0:N3}" -f ($os.FreePhysicalMemory / 1MB)
"free_disk_gb={0:N3}" -f ($drive.Free / 1GB)
"logical_cores={0}" -f (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
"""


#: The session-observer script, verbatim. Nothing is substituted into it.
#:
#: Written for :mod:`fleet.core.observe` (MCPs board task 5a3865bf) and
#: moved here unchanged when the Linux dialect got its own (board task
#: cd5010c4). ``platform`` is the literal ``win32`` because this text runs
#: under PowerShell on Windows by construction. ``hostname`` is lowercased
#: because that is how the harness spells ``pidDomain`` (measured on
#: austinpc: ``$env:COMPUTERNAME`` is ``AUSTINPC``, the record says
#: ``win32:austinpc``). Records are emitted UNTOUCHED so the decode on the
#: hub sees exactly the bytes the harness wrote, and an absent directory is
#: an empty array rather than an error. The directory is a parameter whose
#: default is the profile's, so the Pester suite reads one it laid out
#: (MCPs board task d69786fa).
OBSERVE_SESSIONS_SCRIPT = """\
param(
    [string]$SessionsDirectory = "$HOME\\.claude\\sessions"
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$records = @()
if (Test-Path -LiteralPath $SessionsDirectory) {
    foreach ($file in Get-ChildItem -LiteralPath $SessionsDirectory -Filter '*.json' -File) {
        $records += , (Get-Content -Raw -LiteralPath $file.FullName | ConvertFrom-Json)
    }
}
$document = [pscustomobject]@{
    platform = 'win32'
    hostname = $env:COMPUTERNAME.ToLowerInvariant()
    records  = @($records)
}
$document | ConvertTo-Json -Depth 8 -Compress
"""


#: The tools a checked script may run: the PowerShell variable each is named
#: by, and its param-block line.
_CHECKED_TOOLS: dict[str, tuple[str, str]] = {
    "tar": ("Tar", system32_parameter("Tar", "tar.exe")),
    "git": ("Git", "[string]$Git = 'git'"),
}


def _located_script(parameter: str, location: str, body: tuple[str, ...]) -> str:
    """A script whose one remote location is a parameter.

    The location is the parameter's default, a plain string, so a host runs
    the script with no arguments exactly as before, and the Pester suite
    that executes the committed render (:mod:`fleet.core.rendered_powershell`,
    MCPs board task d69786fa) points it at a directory it laid out.

    Args:
        parameter: The parameter's name, PascalCase.
        location: The absolute remote path it defaults to.
        body: The statements after the strict header, reading the parameter.

    Returns:
        The script's text.

    Raises:
        ValueError: When the location cannot be embedded verbatim.
    """
    default = scriptable(location, label=parameter.lower())
    lines = ["param(", f"    [string]${parameter} = '{default}'", ")", *STRICT_HEADER, *body]
    return "\n".join(lines) + "\n"


class WindowsDialect:
    """The PowerShell rendering of every remote act."""

    def script_path(self, directory: str, stem: str) -> str:
        """Name a ``.ps1`` under a remote directory.

        Args:
            directory: Absolute remote directory.
            stem: The script's name without its extension.

        Returns:
            The absolute path.
        """
        return f"{directory}/{stem}.ps1"

    def fleet_directory(self, user: str) -> str:
        """The provisioned account's ``.fleet`` directory.

        Args:
            user: The provisioned account.

        Returns:
            ``C:/Users/<user>/.fleet``, literal and absolute: the write
            command expands nothing, so ``$env:USERPROFILE`` would name a
            directory called that.
        """
        return f"C:/Users/{user}/.fleet"

    def observe_sessions_script(self) -> str:
        """The constant session-observer script.

        Returns:
            :data:`OBSERVE_SESSIONS_SCRIPT`.
        """
        return OBSERVE_SESSIONS_SCRIPT

    def write_command(self, path: str) -> str:
        """The remote command that writes standard input to a path.

        Args:
            path: Absolute remote path; its parent is created.

        Returns:
            :data:`WRITE_COMMAND` with the path in place.
        """
        return WRITE_COMMAND.format(path=path)

    def invocation(self) -> tuple[str, ...]:
        """How a script file is run by path.

        Returns:
            :data:`POWERSHELL_INVOCATION`.
        """
        return POWERSHELL_INVOCATION

    def echo_command(self, text: str) -> str:
        """One line that prints a literal.

        Args:
            text: The literal, which contains no quote characters.

        Returns:
            ``Write-Output`` of it, single-quoted.
        """
        return f"Write-Output '{text}'"

    def checked_script(self, commands: tuple[tuple[str, ...], ...]) -> str:
        """Render commands as a script that ends at the first failure.

        TWO MECHANISMS BECAUSE POWERSHELL HAS TWO KINDS OF FAILURE, and
        neither one alone catches the other. The strict header's ``Stop``
        makes a CMDLET's error terminating; a NATIVE program's non-zero
        status is not an error at all to PowerShell, and with ``-File`` it is
        not the script's status either. So every command runs through one
        ``Invoke-Step`` that reads the exit code straight after the call and
        exits with it, and the check is written once per script rather than
        once per command.

        EACH TOOL IS A PARAMETER (:data:`_CHECKED_TOOLS`): ``tar`` is
        System32's own by absolute path, because a shell whose ``PATH`` puts
        Git's ``usr/bin`` first resolves a bare ``tar`` to GNU tar, which
        reads ``C:`` as a remote host; ``git`` is whichever the node's
        ``PATH`` names, as it always was. The Pester suite over the committed
        render passes stand-ins for both (MCPs board task d69786fa).

        Args:
            commands: Commands in order, each an argument vector whose first
                word names a tool in :data:`_CHECKED_TOOLS`.

        Returns:
            The script's text.

        Raises:
            ValueError: When a command names another tool, or a word cannot
                be embedded verbatim.
        """
        tools: list[str] = []
        for command in commands:
            if command[0] not in tools:
                tools.append(command[0])
        unknown = [tool for tool in tools if tool not in _CHECKED_TOOLS]
        if unknown:
            raise ValueError(
                f"a checked script runs only {', '.join(sorted(_CHECKED_TOOLS))}, "
                f"not {', '.join(unknown)}"
            )
        parameters = ",\n".join(f"    {_CHECKED_TOOLS[tool][1]}" for tool in tools)
        steps = [
            f"Invoke-Step ${_CHECKED_TOOLS[command[0]][0]} @("
            + ", ".join(f"'{scriptable(word, label='argument')}'" for word in command[1:])
            + ")"
            for command in commands
        ]
        lines = [
            "param(",
            parameters,
            ")",
            *STRICT_HEADER,
            "function Invoke-Step {",
            "    param([string]$Tool, [string[]]$Arguments)",
            "    & $Tool @Arguments",
            "    if ($LASTEXITCODE -ne 0) {",
            "        exit $LASTEXITCODE",
            "    }",
            "}",
            *steps,
        ]
        return "\n".join(lines) + "\n"

    def make_directory_script(self, target: str) -> str:
        """The script that creates a dispatch's directory.

        ``[IO.Directory]::CreateDirectory`` RATHER THAN ``New-Item``, and
        the reason is a parameter that does not exist. This read ``New-Item
        -ItemType Directory -Force -LiteralPath`` from the day the dialect
        was written; ``New-Item`` has no ``-LiteralPath`` (``Set-Content``,
        ``Remove-Item`` and ``Test-Path`` do, which is how it looked right),
        and measured on sedona 2026-09-22 it fails with "A parameter cannot
        be found that matches parameter name 'LiteralPath'". Nothing noticed
        for two weeks because the write command (:data:`WRITE_COMMAND`)
        creates a file's parent itself, so every directory this script was
        meant to make was made by the next step anyway -- until a companion,
        whose directory is filled by ``tar`` rather than by a write, and
        whose ``tar`` then reported "could not chdir".

        The .NET call rather than ``-Path``, which is the parameter that
        does exist, because ``-Path`` reads ``[`` and ``]`` as wildcards:
        the literal-path intent behind the original was right and only its
        spelling was wrong. It also creates parents and is silent about a
        directory that is already there, which is what ``-Force`` was for.

        Args:
            target: Absolute remote directory for this dispatch.

        Returns:
            The script's text. No exit-code check follows the call, because
            nothing in it is a native program: a .NET failure throws, and the
            strict header's Stop ends the script on it.
        """
        return _located_script(
            "Directory", target, ("[IO.Directory]::CreateDirectory($Directory) | Out-Null",)
        )

    def reset_directory_script(self, target: str) -> str:
        """The script that empties a companion's directory and creates it.

        ``-Force`` on the removal is load-bearing rather than defensive: the
        directory being replaced holds a git repository this package made
        there, and every loose object under ``.git/objects`` is written
        READ-ONLY. Without it the second run carrying a companion stops on
        the first object file.

        Args:
            target: Absolute remote directory to replace.

        Returns:
            The script's text, guarded by ``Test-Path`` because
            ``Remove-Item`` of a path that does not exist is an error and the
            first run on a node is exactly that case.
        """
        return _located_script(
            "Directory",
            target,
            (
                "if (Test-Path -LiteralPath $Directory) {",
                "    Remove-Item -Recurse -Force -LiteralPath $Directory",
                "}",
                "[IO.Directory]::CreateDirectory($Directory) | Out-Null",
            ),
        )

    def retire_script(self, *, target: str, retained: str, scripts: tuple[str, ...]) -> str:
        """Keep a settled run's transcript, then remove its directory and scripts.

        Every location is a parameter defaulting to the rendered path, so a
        node runs it with no arguments and the Pester suite over its committed
        render points it at a directory it laid out. Removing its own file is
        safe: ``powershell -File`` has read the whole script before the first
        statement runs.

        Args:
            target: The dispatch's absolute remote directory.
            retained: Where its transcript is kept.
            scripts: The scripts it left under the stage root.

        Returns:
            The script's text. Each removal is guarded by ``Test-Path``,
            because ``Remove-Item`` of a path that is not there is an error and
            a retire run a second time meets exactly that; ``-Force`` removes
            the read-only objects of the repository staged there.

        Raises:
            ValueError: When a path cannot be embedded verbatim.
        """
        # One plain string parameter per script, gathered in the body: a
        # list as a parameter's default is an expression the coverage harness
        # counts as a command, reached only by a run that takes the default.
        parameters = [f"$Script{index}" for index in range(len(scripts))]
        declared = [
            f"    [string]$Target = '{scriptable(target, label='target')}'",
            f"    [string]$Log = '{scriptable(names.log_path(target), label='log')}'",
            f"    [string]$Retained = '{scriptable(retained, label='retained')}'",
            *(
                f"    [string]{parameter} = '{scriptable(script, label='script')}'"
                for parameter, script in zip(parameters, scripts, strict=True)
            ),
        ]
        body = [
            "param(",
            ",\n".join(declared),
            ")",
            *STRICT_HEADER,
            "[IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($Retained)) | Out-Null",
            "if (Test-Path -LiteralPath $Log) {",
            "    Move-Item -Force -LiteralPath $Log -Destination $Retained",
            "}",
            "if (Test-Path -LiteralPath $Target) {",
            "    Remove-Item -Recurse -Force -LiteralPath $Target",
            "}",
            f"foreach ($script in @({', '.join(parameters)})) {{",
            "    if (Test-Path -LiteralPath $script) {",
            "        Remove-Item -Force -LiteralPath $script",
            "    }",
            "}",
        ]
        return "\n".join(body) + "\n"

    def digest_script(self, target: str) -> str:
        """Print the landed archive's SHA-256, extracting nothing.

        Args:
            target: Absolute remote directory for this dispatch.

        Returns:
            The script's text. ``Get-FileHash`` reports upper case, lowered
            here so the output is exactly what the sender compares.
        """
        return _located_script(
            "Target",
            target,
            (
                "(Get-FileHash -Algorithm SHA256 -LiteralPath "
                f'"$Target/{names.ARCHIVE_NAME}").Hash.ToLower()',
            ),
        )

    def build_script(
        self,
        *,
        target: str,
        path: str,
        workers: int,
        install: tuple[tuple[str, ...], ...],
        cache_root: str,
        isolated_docker: bool,
    ) -> str:
        """Ready the tree, run the recipe in the project, write its status last.

        Args:
            target: Absolute remote directory holding the export, its root.
            path: The project's directory inside the export, ``""`` for the
                root.
            workers: Test workers the capacity check granted.
            install: The project's declared install steps, argv each.
            cache_root: The node's cache directory.
            isolated_docker: True for a project that declares the ``docker``
                tag, which no Windows node carries.

        Returns:
            :func:`fleet.core.windows_build.build_script`'s text, which says
            why every native run goes through cmd.exe and why the caches are
            the node's.

        Raises:
            ValueError: When ``isolated_docker`` is True. No Windows node
                declares a rootless daemon (its toolchain probe never answers
                ``docker=yes``), so the capacity check places no docker
                project here; a build rendered for one anyway would run its
                containers as the runner, which is what the isolation exists
                to prevent, so it is refused rather than rendered.
        """
        if isolated_docker:
            raise ValueError(
                f"the project at {path!r} declares the docker tag, and a Windows node has no "
                "rootless daemon to isolate its build on; it runs only on a node whose "
                "execdocker user carries one"
            )
        return windows_build.build_script(
            target=target, path=path, workers=workers, install=install, cache_root=cache_root
        )

    def log_tail_script(self, target: str, lines: int) -> str:
        """Print the last lines of the build's transcript, or nothing.

        Args:
            target: Absolute remote directory holding the export.
            lines: How many lines from the end.

        Returns:
            The script's text, a bounded read from the end of the
            transcript in whichever encoding it carries
            (:mod:`fleet.core.windows_log_tail` says which, and why not
            ``Get-Content -Tail``). An absent transcript prints nothing
            rather than an error: the collector has already read the result
            file, so an empty tail is a build that wrote no transcript, and
            the verdict says so.
        """
        return windows_log_tail_script(names.log_path(target), lines)

    def launch_script(self, *, target: str, run_id: str) -> str:
        """Register a scheduled task for the build, start it, and prove it began.

        Args:
            target: Absolute remote directory holding the staged tree.
            run_id: The dispatch, which names its own task.

        Returns:
            :func:`fleet.core.windows_task.launch_script`'s text, which says
            why Task Scheduler and why it waits for the build's process id.
        """
        return windows_task.launch_script(target=target, run_id=run_id)

    def result_script(self, target: str) -> str:
        """Print the status and the epoch second it was written, or nothing.

        IT REPORTS *WHEN* AS WELL AS *WHAT*. Whether a run was safe is a
        question about whether its lease covered the whole of it, answerable
        only against the moment the build ended -- which the node knows and
        nobody else does. Measured 2026-09-04, a run that finished three
        minutes inside its window was refused twenty minutes later for having
        been collected late.

        The epoch is computed by subtracting the Unix epoch from a UTC
        timestamp rather than with ``-UFormat %s``, which in PowerShell 5.1
        converts from LOCAL time and would put every node's answer out by its
        own offset.

        Args:
            target: Absolute remote directory holding the staged tree.

        Returns:
            The script's text.
        """
        return _located_script(
            "Target",
            target,
            (
                f'$result = "$Target/{names.RESULT_NAME}"',
                "if (Test-Path -LiteralPath $result) {",
                "    $file = Get-Item -LiteralPath $result",
                "    $code = (Get-Content -Raw -LiteralPath $result).Trim()",
                "    $epoch = [int]($file.LastWriteTimeUtc - [datetime]'1970-01-01').TotalSeconds",
                '    "$code $epoch"',
                "}",
            ),
        )

    def stop_script(self, *, target: str, run_id: str) -> str:
        """End the build's process tree, then stop and unregister the task.

        Args:
            target: Absolute remote directory holding the staged tree and the
                build's recorded process id.
            run_id: The dispatch.

        Returns:
            :func:`fleet.core.windows_task.stop_script`'s text, which says
            why the tree is killed first and only by a verified id.
        """
        return windows_task.stop_script(target=target, run_id=run_id)

    def capacity_probe_script(self) -> str:
        """The constant capacity probe.

        Returns:
            :data:`CAPACITY_PROBE_SCRIPT`.
        """
        return CAPACITY_PROBE_SCRIPT

    def toolchain_probe_script(self) -> str:
        """The constant toolchain probe.

        Returns:
            :data:`fleet.core.windows_toolchain_probe.TOOLCHAIN_PROBE_SCRIPT`.
        """
        return TOOLCHAIN_PROBE_SCRIPT


__all__ = [
    "CAPACITY_PROBE_SCRIPT",
    "OBSERVE_SESSIONS_SCRIPT",
    "POWERSHELL_INVOCATION",
    "WRITE_COMMAND",
    "WindowsDialect",
]
