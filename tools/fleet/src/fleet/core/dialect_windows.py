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

from fleet.contracts.project import MAKE_TARGET
from fleet.core import names

#: Task Scheduler's ``SCHED_S_TASK_HAS_NOT_RUN``, 0x00041303.
#:
#: The status a registered task reports until it has run once. It is the
#: signal the launch script waits to stop seeing, because
#: ``Start-ScheduledTask`` reports a failure to start as a NON-TERMINATING
#: error: PowerShell prints it, exits 0, and the dispatch records a run that
#: does not exist. Measured 2026-09-04 -- the ledger said ``running`` for a
#: task whose ``LastRunTime`` was still the 1999 sentinel.
TASK_HAS_NOT_RUN = 267011

#: How long the node waits for a started task to leave that state.
#:
#: Generous because it is bounding a Task Scheduler round trip and not any
#: work: the task only has to BEGIN. A build that starts and fails in the
#: first second still leaves this state, so the wait ends on the first status
#: change rather than on success.
LAUNCH_TIMEOUT_SECONDS = 30

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
$os = Get-CimInstance Win32_OperatingSystem
$drive = Get-PSDrive C
"free_ram_gb={0:N3}" -f ($os.FreePhysicalMemory / 1MB)
"free_disk_gb={0:N3}" -f ($drive.Free / 1GB)
"logical_cores={0}" -f (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors
"""

#: The toolchain probe, verbatim.
#:
#: ``--version`` is asked of each tool and the first line kept, because git
#: and poetry both print several. A tool that is present but declines to
#: answer yields an empty version rather than a failure: absence and silence
#: are different states, and only the first stops a dispatch.
#:
#: ``pip`` is the one manager that is not an executable on PATH but a module
#: of the interpreter, so it is asked as ``python -m pip --version`` behind the
#: same ``$python`` the tool loop reported: without the guard a node with no Python
#: would print a CommandNotFound error where a line was expected. Its output
#: is collected whole and the first line taken afterwards, NOT piped through
#: ``Select-Object -First 1`` like the others: that stops the pipeline early,
#: PowerShell 5.1 then reports the native process as exit -1, and a present
#: pip read as absent (measured on the hub 2026-09-20).
#:
#: A ``python`` that resolves under ``Microsoft\\WindowsApps`` is reported
#: ABSENT. On a node whose PATH has no real interpreter ahead of it, that is
#: the App Execution Alias stub, which answers ``--version`` with "Python was
#: not found; run without arguments to install from the Microsoft Store" and
#: exit 9009. Read as present, that sentence became the version, the node was
#: offered no install, and every slime check lavender took died at its
#: Makefile's first python call (fleet board task e62c8120, 2026-09-22/23).
#: The real Store build lives under the same directory and is refused too,
#: deliberately: it sandboxes ``%LOCALAPPDATA%`` writes and broke poetry's
#: venv on sedona (:mod:`fleet.contracts.toolchain`).
TOOLCHAIN_PROBE_SCRIPT = """\
$python = Get-Command python -ErrorAction SilentlyContinue
if ($python -and $python.Source -like '*\\Microsoft\\WindowsApps\\*') { $python = $null }
foreach ($tool in @('python','poetry','git','make','node','tar','winget','choco')) {
  $found = $python
  if ($tool -ne 'python') { $found = Get-Command $tool -ErrorAction SilentlyContinue }
  if ($found) {
    $raw = (& $tool --version 2>&1 | Select-Object -First 1)
    $text = ($raw | Out-String).Trim() -replace '[\\r\\n]', ' '
    "$tool=yes=$text"
  } else {
    "$tool=no="
  }
}
if ($python) {
  $lines = @(& python -m pip --version 2>&1)
  if ($LASTEXITCODE -eq 0) {
    "pip=yes=" + (($lines[0] | Out-String).Trim() -replace '[\\r\\n]', ' ')
  } else {
    "pip=no="
  }
} else {
  "pip=no="
}
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
#: an empty array rather than an error.
OBSERVE_SESSIONS_SCRIPT = """\
$ErrorActionPreference = 'Stop'
$dir = Join-Path $HOME '.claude\\sessions'
$records = @()
if (Test-Path -LiteralPath $dir) {
    foreach ($file in Get-ChildItem -LiteralPath $dir -Filter '*.json' -File) {
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

    def checked_script(self, commands: tuple[str, ...]) -> str:
        """Render commands as a script that ends at the first failure.

        TWO MECHANISMS BECAUSE POWERSHELL HAS TWO KINDS OF FAILURE, and
        neither one alone catches the other. ``$ErrorActionPreference =
        'Stop'`` makes a CMDLET's error terminating, which exits the script
        non-zero; a NATIVE program's non-zero status is not an error at all
        to PowerShell, and with ``-File`` it is not the script's status
        either, so each command is followed by an explicit check.

        ``-gt 0`` rather than ``-ne 0`` because ``$LASTEXITCODE`` is unset
        until the first native command runs, and ``$null -ne 0`` is true:
        with ``-ne`` a script whose first command is a cmdlet would exit
        before its second. A stale value cannot survive into a later check,
        since any non-zero status exits at the check that follows it.

        Args:
            commands: Command lines, in order.

        Returns:
            The script's text.
        """
        lines = ["$ErrorActionPreference = 'Stop'"]
        for command in commands:
            lines.append(command)
            lines.append("if ($LASTEXITCODE -gt 0) { exit $LASTEXITCODE }")
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
            The script's text.
        """
        return self.checked_script((f"[IO.Directory]::CreateDirectory('{target}') | Out-Null",))

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
        return self.checked_script(
            (
                f"if (Test-Path -LiteralPath '{target}') "
                f"{{ Remove-Item -Recurse -Force -LiteralPath '{target}' }}",
                f"[IO.Directory]::CreateDirectory('{target}') | Out-Null",
            )
        )

    def reassemble_script(self, target: str) -> str:
        """Decode the one-line base64 and print the SHA-256, extracting nothing.

        Args:
            target: Absolute remote directory for this dispatch.

        Returns:
            The script's text. ``Get-Content -Raw`` because the archive
            travels as one line; wrapping would make the decode depend on how
            the writer chose to fold it.
        """
        return (
            f"$encoded = Get-Content -Raw -LiteralPath '{target}/{names.ENCODED_NAME}'\n"
            f"$bytes = [Convert]::FromBase64String($encoded.Trim())\n"
            f"[IO.File]::WriteAllBytes('{target}/{names.ARCHIVE_NAME}', $bytes)\n"
            f"(Get-FileHash -Algorithm SHA256 -LiteralPath "
            f"'{target}/{names.ARCHIVE_NAME}').Hash.ToLower()\n"
        )

    def build_script(
        self,
        *,
        target: str,
        path: str,
        workers: int,
        install: tuple[tuple[str, ...], ...],
        cache_root: str,
    ) -> str:
        """Ready the tree, run the recipe in the project, write its status last.

        ``$LASTEXITCODE`` rather than ``$?`` because the recipe is a native
        program: PowerShell sets ``$?`` false whenever a native command writes
        to a redirected stderr, which ``make`` does routinely on a passing run.

        THE CACHES ARE THE NODE'S, NOT THE RUN'S. A clean export carries no
        ``node_modules``, no ``.venv`` and no browsers, and installing them
        from the network on every run would make a twelve-minute suite a
        thirty-minute one. The three package managers each honour one
        environment variable naming their cache, so the build points all
        three at ``<stage_root>/cache`` (:func:`fleet.core.names.cache_root`),
        which every run on the node shares and no run directory contains:
        the first run fills it and every later run restores from it, which
        is what "restores dependencies from the node's cache" means here.

        THE INSTALL STEPS RUN AT THE EXPORT ROOT, before the recipe, each
        appending to the same transcript; the first that exits non-zero ends
        the build with ITS status in the result file, so a dependency that
        will not install reads as an install failure in the log and never as
        the suite's fault. Each token was held to the source grammar at
        decode (:mod:`fleet.contracts.source`), so it is written bare.

        Args:
            target: Absolute remote directory holding the export, its root.
            path: The project's directory inside the export, ``""`` for the
                root.
            workers: Test workers the capacity check granted.
            install: The project's declared install steps, argv each.
            cache_root: The node's cache directory.

        Returns:
            The script's text. Its first act records its own process id in
            :data:`~fleet.core.names.PID_NAME`, for :meth:`stop_script`, and
            its last writes the recipe's exit status to the result file.
        """
        log = names.log_path(target)
        result = f"{target}/{names.RESULT_NAME}"
        lines = [
            f"$PID | Set-Content -LiteralPath '{target}/{names.PID_NAME}'",
            "$ErrorActionPreference = 'Continue'",
            f"$env:npm_config_cache = '{cache_root}/npm'",
            f"$env:POETRY_CACHE_DIR = '{cache_root}/pypoetry'",
            f"$env:PLAYWRIGHT_BROWSERS_PATH = '{cache_root}/ms-playwright'",
            f"$env:PYTEST_XDIST_AUTO_NUM_WORKERS = '{workers}'",
            f"Set-Location -LiteralPath '{target}'",
        ]
        for step in install:
            command = " ".join(step)
            lines.append(f"Write-Output '$ {command}' *>> '{log}'")
            lines.append(f"{command} *>> '{log}'")
            lines.append(
                f"if ($LASTEXITCODE -ne 0) {{ $LASTEXITCODE | Set-Content -LiteralPath "
                f"'{result}'; exit 0 }}"
            )
        lines.append(f"Set-Location -LiteralPath '{names.recipe_directory(target, path)}'")
        lines.append(f"make {MAKE_TARGET} *>> '{log}'")
        lines.append(f"$LASTEXITCODE | Set-Content -LiteralPath '{result}'")
        return "\n".join(lines) + "\n"

    def log_tail_script(self, target: str, lines: int) -> str:
        """Print the last lines of the build's transcript, or nothing.

        Args:
            target: Absolute remote directory holding the export.
            lines: How many lines from the end.

        Returns:
            The script's text. An absent transcript prints nothing rather
            than an error: the collector has already read the result file,
            so an empty tail is a build that wrote no transcript, and the
            verdict says so.
        """
        log = names.log_path(target)
        return (
            f"if (Test-Path -LiteralPath '{log}') {{\n"
            f"  Get-Content -Tail {lines} -LiteralPath '{log}'\n"
            f"}}\n"
        )

    def launch_script(self, *, target: str, run_id: str) -> str:
        """Register a scheduled task for the build, start it, and prove it began.

        WHY TASK SCHEDULER AND NOT AN SSH CHILD. Windows OpenSSH assigns the
        session's process tree to a job object precisely so the tree dies when
        the connection ends, and a process cannot be moved out of a job object
        once it is in one (``memory/reference_long_runs_need_task_scheduler.md``).

        ``-AllowStartIfOnBatteries`` and ``-DontStopIfGoingOnBatteries`` are
        not optional and their defaults are the wrong way round for this
        fleet: two of the three Windows nodes are laptops, so a dispatch to an
        unplugged sedona would register a task that never runs, or would have
        a running suite killed the moment somebody unplugged it. ``-Priority
        4`` because 7, the default, sets LOW I/O and a run that inherits it
        crawls in a way that reads as a slow node.

        The script then WAITS for the task to leave :data:`TASK_HAS_NOT_RUN`
        before saying so, because ``Start-ScheduledTask`` reports a refusal as
        a non-terminating error that would otherwise exit 0 and be recorded as
        a launch.

        Args:
            target: Absolute remote directory holding the staged tree.
            run_id: The dispatch, which names its own task.

        Returns:
            The script's text.
        """
        task = names.task_name(run_id)
        build = f"{target}/{names.BUILD_STEM}.ps1"
        return (
            f"$ErrorActionPreference = 'Stop'\n"
            f"$action = New-ScheduledTaskAction -Execute 'powershell.exe' "
            f"-Argument '-NoProfile -ExecutionPolicy Bypass -File \"{build}\"'\n"
            f"$settings = New-ScheduledTaskSettingsSet -Priority 4 "
            f"-ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew "
            f"-AllowStartIfOnBatteries -DontStopIfGoingOnBatteries\n"
            f"$principal = New-ScheduledTaskPrincipal "
            f"-UserId ([Security.Principal.WindowsIdentity]::GetCurrent().User.Value) "
            f"-LogonType S4U\n"
            f"Register-ScheduledTask -TaskName '{task}' -Action $action "
            f"-Settings $settings -Principal $principal -Force | Out-Null\n"
            f"Start-ScheduledTask -TaskName '{task}'\n"
            f"$deadline = (Get-Date).AddSeconds({LAUNCH_TIMEOUT_SECONDS})\n"
            f"while ((Get-Date) -lt $deadline) {{\n"
            f"  if ((Get-ScheduledTaskInfo -TaskName '{task}').LastTaskResult "
            f"-ne {TASK_HAS_NOT_RUN}) {{ Write-Output 'launched'; exit 0 }}\n"
            f"  Start-Sleep -Milliseconds 500\n"
            f"}}\n"
            f'throw "{task} registered but has not run after {LAUNCH_TIMEOUT_SECONDS}s"\n'
        )

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
        result = f"{target}/{names.RESULT_NAME}"
        return (
            f"if (Test-Path -LiteralPath '{result}') {{\n"
            f"  $file = Get-Item -LiteralPath '{result}'\n"
            f"  $code = (Get-Content -Raw -LiteralPath '{result}').Trim()\n"
            f"  $epoch = [int]($file.LastWriteTimeUtc - [datetime]'1970-01-01').TotalSeconds\n"
            f'  "$code $epoch"\n'
            f"}}\n"
        )

    def stop_script(self, *, target: str, run_id: str) -> str:
        """End the build's process tree, then stop and unregister the task.

        STOPPING THE TASK IS NOT STOPPING THE BUILD. ``Stop-ScheduledTask``
        ends the process the task started and leaves its children running:
        measured on sedona 2026-09-23, a probe task whose ``build.ps1`` ran a
        native child read parent alive=False, child alive=True afterwards. So
        the tree is ended first, by the process id the build recorded as its
        first act, with ``taskkill /T /F``, which on the same probe ended the
        parent, its child and its grandchild.

        THE ID IS CHECKED BEFORE ANYTHING IS KILLED. A recorded id outlives
        its process, and Windows reuses ids, so the kill happens only when
        the process holding that id right now is running this dispatch's own
        ``build.ps1``, read off its command line. A build that has finished
        or died leaves an id that names nothing or names a stranger, and
        either way nothing is killed. It kills by id and never by name or
        pattern, which is the fleet's one rule about killing.

        ``-Confirm:$false`` because there is nobody at the node to answer, and
        an unanswered prompt would hang the cancel until its ssh timeout
        rather than stopping anything.

        Args:
            target: Absolute remote directory holding the staged tree and the
                build's recorded process id.
            run_id: The dispatch.

        Returns:
            The script's text. ``taskkill`` exiting non-zero on a verified
            process fails the script, so a stop that ended nothing is never
            reported as one that did.
        """
        task = names.task_name(run_id)
        pid_file = f"{target}/{names.PID_NAME}"
        build = f"{target}/{names.BUILD_STEM}.ps1"
        return (
            f"if (Test-Path -LiteralPath '{pid_file}') {{\n"
            f"  $buildPid = [int](Get-Content -Raw -LiteralPath '{pid_file}').Trim()\n"
            f'  $process = Get-CimInstance Win32_Process -Filter "ProcessId=$buildPid"\n'
            f"  if ($null -ne $process -and $process.CommandLine -like '*{build}*') {{\n"
            f"    & taskkill.exe /PID $buildPid /T /F\n"
            f"    if ($LASTEXITCODE -gt 0) {{\n"
            f'      throw "taskkill of $buildPid exited $LASTEXITCODE"\n'
            f"    }}\n"
            f"  }}\n"
            f"}}\n"
            f"Stop-ScheduledTask -TaskName '{task}' -ErrorAction SilentlyContinue\n"
            f"Unregister-ScheduledTask -TaskName '{task}' -Confirm:$false "
            f"-ErrorAction SilentlyContinue\n"
            f"Write-Output 'stopped {task}'\n"
        )

    def capacity_probe_script(self) -> str:
        """The constant capacity probe.

        Returns:
            :data:`CAPACITY_PROBE_SCRIPT`.
        """
        return CAPACITY_PROBE_SCRIPT

    def toolchain_probe_script(self) -> str:
        """The constant toolchain probe.

        Returns:
            :data:`TOOLCHAIN_PROBE_SCRIPT`.
        """
        return TOOLCHAIN_PROBE_SCRIPT


__all__ = [
    "CAPACITY_PROBE_SCRIPT",
    "LAUNCH_TIMEOUT_SECONDS",
    "OBSERVE_SESSIONS_SCRIPT",
    "POWERSHELL_INVOCATION",
    "TASK_HAS_NOT_RUN",
    "TOOLCHAIN_PROBE_SCRIPT",
    "WRITE_COMMAND",
    "WindowsDialect",
]
