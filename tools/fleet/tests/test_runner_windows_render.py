"""What the Windows runner host's two renders carry, decided before anything runs.

What the scripts DO is executed by tests/pester/rendered-runner-windows.Tests.ps1
over their committed copies; here are the values each copy is rendered with.
"""

from __future__ import annotations

import pytest

from fleet.contracts.runner_windows import WindowsRunnerHost, WindowsRunnerInstall
from fleet.core import (
    runner_windows_base_render,
    runner_windows_job_render,
    runner_windows_python_render,
    runner_windows_render,
)
from fleet.core.runner_audit_types import ExpectedCheck
from tests._runner_fixtures import a_windows_host


def _install(repo: str) -> WindowsRunnerInstall:
    """One of loki's runners.

    Args:
        repo: Its repository.

    Returns:
        The install.
    """
    name = repo.split("/")[1]
    return WindowsRunnerInstall(
        repo=repo,
        runner_name="loki",
        service=f"actions.runner.{repo.replace('/', '-')}.loki",
        root=f"C:/actions-runner/{name}",
        labels=["loki", "windows-ci"],
        python_toolcache=["3.11.9"],
    )


def _host(*installs: WindowsRunnerInstall) -> WindowsRunnerHost:
    """loki with the given runners and the fixture's base.

    Args:
        installs: Its runners.

    Returns:
        The host.
    """
    host = a_windows_host()
    host["installs"] = list(installs)
    return host


def _parameter_lines(script: str) -> list[str]:
    """The lines of a render's param block.

    Args:
        script: The render.

    Returns:
        Every line from ``param(`` to its closing parenthesis.
    """
    lines = script.splitlines()
    return lines[: lines.index(")") + 1]


class TestTheInstallScript:
    def test_the_installs_are_one_json_string_the_script_decodes_in_its_body(self) -> None:
        mcps = _install("wagner-austin/MCPs")
        mcps["python_toolcache"] = []
        script = runner_windows_render.render_install_script(
            _host(_install("wagner-austin/chat"), mcps), {}
        )
        assert _parameter_lines(script)[-2:] == [
            "    [string]$Installs = '["
            '{"Repo":"wagner-austin/chat","Name":"loki","Labels":"loki,windows-ci",'
            '"Root":"C:/actions-runner/chat","Service":"actions.runner.wagner-austin-chat.loki",'
            '"Python":["3.11.9"]},'
            '{"Repo":"wagner-austin/MCPs","Name":"loki","Labels":"loki,windows-ci",'
            '"Root":"C:/actions-runner/MCPs","Service":"actions.runner.wagner-austin-MCPs.loki",'
            '"Python":[]}'
            "]'",
            ")",
        ]
        assert "foreach ($install in ($Installs | ConvertFrom-Json)) {" in script.splitlines()

    def test_the_base_parameters_and_lines_come_before_the_first_runner(self) -> None:
        script = runner_windows_render.committed_install_script()
        lines = script.splitlines()
        assert _parameter_lines(script)[4:16] == [
            *runner_windows_base_render.parameter_lines(a_windows_host()["base"]),
            *runner_windows_job_render.parameter_lines(),
            *runner_windows_python_render.PARAMETER_LINES,
            f"    [int]$StopWaitSeconds = {runner_windows_render.STOP_WAIT_SECONDS},",
        ]
        first = lines.index(runner_windows_base_render.INSTALL_LINES[0])
        laid = [*runner_windows_base_render.INSTALL_LINES, *runner_windows_job_render.INSTALL_LINES]
        assert lines[first : first + len(laid)] == laid
        assert first < lines.index("foreach ($install in ($Installs | ConvertFrom-Json)) {")

    def test_each_runner_s_tool_cache_is_seeded_after_its_registration_before_its_service(
        self,
    ) -> None:
        lines = runner_windows_render.committed_install_script().splitlines()
        seed = lines.index(runner_windows_python_render.INSTALL_LINES[0])
        assert lines[seed : seed + len(runner_windows_python_render.INSTALL_LINES)] == list(
            runner_windows_python_render.INSTALL_LINES
        )
        registered = '        "runner: registered $($install.Repo) runner $($install.Name)"'
        assert lines[seed - 2 : seed] == [registered, "    }"]
        assert lines[seed + len(runner_windows_python_render.INSTALL_LINES)].startswith(
            "    $null = & $Sc failure $install.Service"
        )

    def test_the_tokens_are_one_pair_per_repository_in_order(self) -> None:
        script = runner_windows_render.render_install_script(
            _host(_install("wagner-austin/chat")),
            {"wagner-austin/tree-bot": "BBB222", "wagner-austin/chat": "AAA111"},
        )
        assert (
            "    [hashtable]$Tokens = @{ 'wagner-austin/chat' = 'AAA111'; "
            "'wagner-austin/tree-bot' = 'BBB222' },"
        ) in _parameter_lines(script)

    def test_the_committed_copy_carries_no_token(self) -> None:
        assert "    [hashtable]$Tokens = @{}," in _parameter_lines(
            runner_windows_render.committed_install_script()
        )

    def test_a_token_that_could_break_its_quoting_is_refused(self) -> None:
        with pytest.raises(ValueError) as refused:
            runner_windows_render.render_install_script(
                _host(_install("wagner-austin/chat")), {"wagner-austin/chat": "A'; rm"}
            )
        assert str(refused.value) == (
            "the registration token for wagner-austin/chat is not alphanumeric"
        )

    def test_an_install_value_a_script_could_not_carry_is_refused(self) -> None:
        install = _install("wagner-austin/chat")
        install["root"] = "C:/actions-runner/it's"
        with pytest.raises(ValueError) as refused:
            runner_windows_render.render_install_script(_host(install), {})
        assert str(refused.value) == (
            'root "C:/actions-runner/it\'s" contains "\'", which cannot be embedded in a '
            "rendered script verbatim; rename the item rather than escaping it"
        )

    def test_the_release_is_the_pinned_one_and_sc_is_named_by_its_system32_path(self) -> None:
        lines = _parameter_lines(runner_windows_render.committed_install_script())
        assert lines[1:3] == [
            "    [string]$RunnerZipUrl = 'https://github.com/actions/runner/releases/download/"
            "v2.337.0/actions-runner-win-x64-2.337.0.zip',",
            '    [string]$Sc = "$env:SystemRoot\\System32\\sc.exe",',
        ]


class TestTheAudit:
    def test_the_audit_names_each_service_after_the_base(self) -> None:
        script = runner_windows_render.render_audit_script(
            _host(_install("wagner-austin/chat"), _install("wagner-austin/MCPs"))
        )
        assert _parameter_lines(script) == [
            "param(",
            '    [string]$Sc = "$env:SystemRoot\\System32\\sc.exe",',
            *runner_windows_base_render.parameter_lines(a_windows_host()["base"]),
            *runner_windows_job_render.parameter_lines(),
            "    [string]$Installs = '["
            '{"Service":"actions.runner.wagner-austin-chat.loki",'
            '"Root":"C:/actions-runner/chat","Python":["3.11.9"]},'
            '{"Service":"actions.runner.wagner-austin-MCPs.loki",'
            '"Root":"C:/actions-runner/MCPs","Python":["3.11.9"]}'
            "]'",
            ")",
        ]
        lines = script.splitlines()
        base = lines.index(runner_windows_base_render.AUDIT_LINES[0])
        read = [*runner_windows_base_render.AUDIT_LINES, *runner_windows_job_render.AUDIT_LINES]
        assert lines[base : base + len(read)] == read
        rows = lines.index(runner_windows_python_render.AUDIT_LINES[0])
        assert lines[rows : rows + len(runner_windows_python_render.AUDIT_LINES)] == list(
            runner_windows_python_render.AUDIT_LINES
        )

    def test_a_service_name_a_script_could_not_carry_is_refused(self) -> None:
        install = _install("wagner-austin/chat")
        install["service"] = "actions.runner.$x"
        with pytest.raises(ValueError, match=r"^service 'actions\.runner\.\$x' contains '\$'"):
            runner_windows_render.render_audit_script(_host(install))

    def test_the_account_row_matches_the_name_the_install_registers_as_sc_reads_it(
        self,
    ) -> None:
        assert (
            "    if ($config.Code -eq 0 -and $config.Text -match 'START_TYPE : 2 AUTO_START' "
            "-and $config.Text -match 'SERVICE_START_NAME : NT AUTHORITY\\\\SYSTEM') {"
        ) in runner_windows_render.committed_audit_script().splitlines()

    def test_the_base_rows_then_three_rows_per_install_in_the_order_the_script_prints_them(
        self,
    ) -> None:
        host = a_windows_host()
        service = "actions.runner.wagner-austin-chat.loki"
        assert runner_windows_render.expected_checks(host) == [
            *runner_windows_base_render.expected_checks(host["base"]),
            *runner_windows_job_render.expected_checks(),
            ExpectedCheck(
                check_id=f"service:{service}",
                reason=f"{runner_windows_render.SERVICE_REASON}: wagner-austin/chat's runner loki",
            ),
            ExpectedCheck(
                check_id=f"account:{service}", reason=runner_windows_render.ACCOUNT_REASON
            ),
            ExpectedCheck(
                check_id=f"recovery:{service}", reason=runner_windows_render.RECOVERY_REASON
            ),
            ExpectedCheck(
                check_id=f"python:{service}:3.11.9",
                reason=runner_windows_python_render.SEED_REASON,
            ),
        ]

    def test_a_runner_with_no_tool_cache_version_has_no_tool_cache_row(self) -> None:
        install = _install("wagner-austin/chat")
        install["python_toolcache"] = []
        checks = runner_windows_render.expected_checks(_host(install))
        assert [check["check_id"] for check in checks][-1] == (
            "recovery:actions.runner.wagner-austin-chat.loki"
        )

    def test_a_tool_cache_version_a_script_could_not_carry_is_refused(self) -> None:
        install = _install("wagner-austin/chat")
        install["python_toolcache"] = ["3.11.9'"]
        with pytest.raises(ValueError, match=r"^python_toolcache \"3\.11\.9'\" contains"):
            runner_windows_render.render_audit_script(_host(install))

    def test_the_committed_copy_is_the_example_host_s(self) -> None:
        assert runner_windows_render.committed_audit_script() == (
            runner_windows_render.render_audit_script(a_windows_host())
        )
