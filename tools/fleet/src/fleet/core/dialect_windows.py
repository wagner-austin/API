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
#: of the interpreter, so it is asked as ``python -m pip --version`` behind a
#: ``Get-Command python`` guard: without the guard a node with no Python
#: would print a CommandNotFound error where a line was expected. Its output
#: is collected whole and the first line taken afterwards, NOT piped through
#: ``Select-Object -First 1`` like the others: that stops the pipeline early,
#: PowerShell 5.1 then reports the native process as exit -1, and a present
#: pip read as absent (measured on the hub 2026-09-20).
TOOLCHAIN_PROBE_SCRIPT = """\
foreach ($tool in @('python','poetry','git','make','tar','winget','choco')) {
  $found = Get-Command $tool -ErrorAction SilentlyContinue
  if ($found) {
    $raw = (& $tool --version 2>&1 | Select-Object -First 1)
    $text = ($raw | Out-String).Trim() -replace '[\\r\\n]', ' '
    "$tool=yes=$text"
  } else {
    "$tool=no="
  }
}
if (Get-Command python -ErrorAction SilentlyContinue) {
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

    def make_directory_script(self, target: str) -> str:
        """The script that creates a dispatch's directory.

        Args:
            target: Absolute remote directory for this dispatch.

        Returns:
            ``New-Item -Force`` with ``-LiteralPath``, so brackets in a path
            are not read as wildcards.
        """
        return f"New-Item -ItemType Directory -Force -LiteralPath '{target}' | Out-Null\n"

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
            The script's text. Its last act writes the recipe's exit status
            to the result file.
        """
        log = names.log_path(target)
        result = f"{target}/{names.RESULT_NAME}"
        lines = [
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

    def stop_script(self, run_id: str) -> str:
        """Stop and unregister the task, never prompting.

        ``-Confirm:$false`` because there is nobody at the node to answer, and
        an unanswered prompt would hang the cancel until its ssh timeout
        rather than stopping anything.

        Args:
            run_id: The dispatch.

        Returns:
            The script's text.
        """
        task = names.task_name(run_id)
        return (
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
