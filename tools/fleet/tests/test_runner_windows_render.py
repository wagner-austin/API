"""What the Windows runner host's two renders carry, decided before anything runs.

What the scripts DO is executed by tests/pester/rendered-runner-windows.Tests.ps1
over their committed copies; here are the values each copy is rendered with.
"""

from __future__ import annotations

import pytest

from fleet.contracts.runner_windows import WindowsRunnerInstall
from fleet.core import runner_windows_render
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
    )


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
        script = runner_windows_render.render_install_script(
            [_install("wagner-austin/chat"), _install("wagner-austin/MCPs")], {}
        )
        assert _parameter_lines(script)[-2:] == [
            "    [string]$Installs = '["
            '{"Repo":"wagner-austin/chat","Name":"loki","Labels":"loki,windows-ci",'
            '"Root":"C:/actions-runner/chat","Service":"actions.runner.wagner-austin-chat.loki"},'
            '{"Repo":"wagner-austin/MCPs","Name":"loki","Labels":"loki,windows-ci",'
            '"Root":"C:/actions-runner/MCPs","Service":"actions.runner.wagner-austin-MCPs.loki"}'
            "]'",
            ")",
        ]
        assert "foreach ($install in ($Installs | ConvertFrom-Json)) {" in script.splitlines()

    def test_the_tokens_are_one_pair_per_repository_in_order(self) -> None:
        script = runner_windows_render.render_install_script(
            [_install("wagner-austin/chat")],
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
                [_install("wagner-austin/chat")], {"wagner-austin/chat": "A'; rm"}
            )
        assert str(refused.value) == (
            "the registration token for wagner-austin/chat is not alphanumeric"
        )

    def test_the_release_is_the_pinned_one_and_sc_is_named_by_its_system32_path(self) -> None:
        lines = _parameter_lines(runner_windows_render.committed_install_script())
        assert lines[1:3] == [
            "    [string]$RunnerZipUrl = 'https://github.com/actions/runner/releases/download/"
            "v2.337.0/actions-runner-win-x64-2.337.0.zip',",
            '    [string]$Sc = "$env:SystemRoot\\System32\\sc.exe",',
        ]


class TestTheAudit:
    def test_the_audit_names_each_service_and_nothing_else(self) -> None:
        script = runner_windows_render.render_audit_script(
            [_install("wagner-austin/chat"), _install("wagner-austin/MCPs")]
        )
        assert _parameter_lines(script) == [
            "param(",
            '    [string]$Sc = "$env:SystemRoot\\System32\\sc.exe",',
            "    [string]$Installs = '["
            '{"Service":"actions.runner.wagner-austin-chat.loki"},'
            '{"Service":"actions.runner.wagner-austin-MCPs.loki"}'
            "]'",
            ")",
        ]

    def test_the_account_row_matches_the_name_the_install_registers_as_sc_reads_it(
        self,
    ) -> None:
        assert (
            "    if ($config.Code -eq 0 -and $config.Text -match 'START_TYPE : 2 AUTO_START' "
            "-and $config.Text -match 'SERVICE_START_NAME : NT AUTHORITY\\\\SYSTEM') {"
        ) in runner_windows_render.committed_audit_script().splitlines()

    def test_three_rows_per_install_in_the_order_the_script_prints_them(self) -> None:
        service = "actions.runner.wagner-austin-chat.loki"
        assert runner_windows_render.expected_checks(a_windows_host()) == [
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
        ]

    def test_the_committed_copy_is_the_example_install_s(self) -> None:
        assert runner_windows_render.committed_audit_script() == (
            runner_windows_render.render_audit_script(a_windows_host()["installs"])
        )
