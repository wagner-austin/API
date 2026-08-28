"""An imaged command must be one a shell-less exec can actually run.

The rule exists because seven training jobs went out on 2026-08-28 with a
``cd ... && ...`` payload and died in under three seconds, exit 127, before a
GPU was touched. Preflight had passed and was right to: ``sbatch --test-only``
answers whether the scheduler would admit the job, never whether the command
can execute.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.contracts.image import ImageReference
from hpc3.contracts.payload import check_imaged_command_can_run

_IMAGE: ImageReference = {"path": "/pub/images/v1/abl.sif", "sha256": "a" * 64, "binds": ["/pub"]}

_BROKEN = "cd /pub/wagnera3/LSTM && python -m char_lstm.train --lang tr"


class TestTheOperatorThatSplitTheLine:
    def test_the_command_that_killed_seven_jobs_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            check_imaged_command_can_run(_IMAGE, _BROKEN)
        assert excinfo.value.code is Hpc3ErrorCode.IMAGED_COMMAND_NEEDS_A_SHELL

    def test_the_refusal_shows_the_wrapped_form(self) -> None:
        """A refusal that does not show the fix gets worked around."""
        with pytest.raises(AppError) as excinfo:
            check_imaged_command_can_run(_IMAGE, _BROKEN)
        assert f'bash -c "{_BROKEN}"' in str(excinfo.value)

    def test_each_splitting_operator_is_caught(self) -> None:
        for operator in ("&&", "||", ";", "|", ">", ">>", "<", "&"):
            with pytest.raises(AppError) as excinfo:
                check_imaged_command_can_run(_IMAGE, f"python a.py {operator} b")
            assert excinfo.value.code is Hpc3ErrorCode.IMAGED_COMMAND_NEEDS_A_SHELL

    def test_the_wrapped_form_is_admitted(self) -> None:
        check_imaged_command_can_run(_IMAGE, f'bash -c "{_BROKEN}"')

    def test_sh_is_a_shell_too(self) -> None:
        check_imaged_command_can_run(_IMAGE, "sh -c 'a && b'")


class TestQuotingIsRespected:
    """The first version of this check was a substring scan and refused six
    committed probe documents that had really run."""

    def test_a_semicolon_inside_a_python_argument_is_not_an_operator(self) -> None:
        check_imaged_command_can_run(
            _IMAGE, "python -c 'import torch, json; print(json.dumps({}))'"
        )

    def test_a_redirect_character_inside_a_quoted_argument_is_left_alone(self) -> None:
        check_imaged_command_can_run(_IMAGE, "python -c 'assert 1 < 2'")

    def test_env_assignments_need_no_shell_and_are_admitted(self) -> None:
        """`env` takes VAR=value before the command, which is how `floor` runs."""
        check_imaged_command_can_run(
            _IMAGE, "HF_HOME=/pub/hf TRANSFORMERS_OFFLINE=1 modeltrainer-score-baseline --x 1"
        )

    def test_a_dollar_variable_is_left_to_the_outer_shell(self) -> None:
        """It is expanded before apptainer sees it, which is useful and works."""
        check_imaged_command_can_run(_IMAGE, "python train.py --out /tmp/$SLURM_JOB_ID.json")

    def test_quoting_that_never_closes_is_refused(self) -> None:
        with pytest.raises(AppError) as excinfo:
            check_imaged_command_can_run(_IMAGE, "python -c 'import torch")
        assert excinfo.value.code is Hpc3ErrorCode.IMAGED_COMMAND_NEEDS_A_SHELL
        assert "does not close" in str(excinfo.value)


class TestAHostRunIsALineInTheScript:
    def test_a_host_run_may_use_any_shell_construct(self) -> None:
        """Which is why `cleargbm`'s `cd ... && python -m scripts.optimize`
        has always worked: no image, so bash runs it."""
        check_imaged_command_can_run(None, _BROKEN)
