"""Tests for the preflight question "can this environment start the module the run names".

Board task 2c8161a3. The registered ``mi`` image, v23, was built from
``fab1d991``, which predates ``model_trainer.cli.cartridge_composition_sweep``;
a run naming that command under it would die on its first line. The answers
the fake cluster gives here are the ones the real one gave on 2026-10-16,
when the probe was run on login-i17 against v23 itself:

    ENTRYPOINT-COMMIT|fab1d991f4d6f39e72c3b21ca5757a072c01f277
    ENTRYPOINT|ABSENT|model_trainer.cli.cartridge_composition_sweep

and, for the command the v23 known-answer probes ran, PRESENT. The probe's own
program is also run for real, by a real bash and a real interpreter, against
a tree built in the test, so its resolution rule is checked against what
Python does rather than against this file's idea of it.
"""

from __future__ import annotations

import pathlib
import shlex
import subprocess
import sys

import pytest
from platform_core.errors import AppError, Hpc3ErrorCode
from platform_core.json_utils import JSONValue

from hpc3.contracts.image import ImageReference
from hpc3.contracts.job import JobSpec
from hpc3.core.array_submit import submit_array
from hpc3.core.entrypoint import (
    EntrypointReading,
    check_entrypoints,
    invocations,
    parse_probe,
    probe_command,
)
from hpc3.core.preflight import preflight
from tests.against_hpc3 import decode_job_spec, decode_sweep_spec
from tests.bash_discovery import posix_bash
from tests.conftest import (
    FakeRun,
    cluster,
    gpus,
    image_reference,
    script_healthy_cluster,
    script_healthy_environment,
)

_WALL_SECONDS = 60
"""Bound on each real probe; one interpreter start per directory."""

_V23 = ImageReference(
    path="/pub/wagnera3/images/v23/abl.sif",
    sha256="55651342e15d2a8652ea00e7c90c3b0d6bb6f9d3351350dc6a7afc415d76c437",
    binds=["/pub/wagnera3"],
)
"""The registered ``mi`` image, as ``runs/hpc3-mi.json`` declares it."""

_V23_COMMIT = "fab1d991f4d6f39e72c3b21ca5757a072c01f277"
_SWEEP = "model_trainer.cli.cartridge_composition_sweep"
_KNOWN_ANSWER = "model_trainer.cli.known_answer_probe"
_SCRIPTS = "/pub/wagnera3/mi/scripts"


def _answer(commit: str, *verdicts: tuple[str, str]) -> str:
    """Build a probe answer in the shape the cluster printed.

    Args:
        commit: The stamp line's value.
        *verdicts: ``(word, module)`` pairs.

    Returns:
        The probe's standard output.
    """
    lines = [f"ENTRYPOINT-COMMIT|{commit}"]
    lines.extend(f"ENTRYPOINT|{word}|{module}" for word, module in verdicts)
    return "\n".join(lines) + "\n"


def _check(commands: list[str], image: ImageReference | None = _V23) -> EntrypointReading:
    """Run the check against the fake cluster from the mi script directory.

    Args:
        commands: The commands the submission runs.
        image: The image, or None for a host environment.

    Returns:
        The reading.
    """
    return check_entrypoints(
        "hpc3", commands, env_path="/opt/env", image=image, working_directory=_SCRIPTS
    )


class TestFindingTheModules:
    def test_a_plain_invocation_starts_in_the_jobs_directory(self) -> None:
        assert invocations(f"python -m {_SWEEP} --plan x", working_directory="/w") == [
            ("/w", _SWEEP)
        ]

    def test_prefixes_and_interpreter_spellings_are_all_read(self) -> None:
        """The three shapes the committed run documents carry."""
        command = (
            "HF_HOME=/pub/hf /opt/env/bin/python -m a.b --x; "
            "CUBLASLT_WORKSPACE_SIZE=0 python3.11 -m c; python3 -m d.e"
        )
        assert [module for _, module in invocations(command, working_directory="/w")] == [
            "a.b",
            "c",
            "d.e",
        ]

    def test_a_cd_sets_where_the_next_module_is_looked_for(self) -> None:
        """cleargbm runs scripts.optimize from a cluster checkout, in no wheel."""
        command = (
            'bash -c "cd /pub/wagnera3/api/services/covenant-radar-api && '
            'APP__DATA_ROOT=data python -m scripts.optimize -b cleargbm"'
        )
        assert invocations(command, working_directory="/w") == [
            ("/pub/wagnera3/api/services/covenant-radar-api", "scripts.optimize")
        ]

    def test_a_repeat_is_asked_once(self) -> None:
        assert invocations("python -m a; python -m a", working_directory="/w") == [("/w", "a")]

    def test_a_script_or_a_lookalike_names_no_module(self) -> None:
        assert invocations("python train.py --out x; pythonic -m y", working_directory="/w") == []


class TestTheProbeRunsForReal:
    """The probe line, executed by a real bash and a real interpreter."""

    def _environment(self, root: pathlib.Path, *, stamped: bool) -> str:
        """Make an environment whose bin/python is this interpreter.

        Args:
            root: Directory to build in.
            stamped: Whether to write a GIT_COMMIT stamp.

        Returns:
            The environment's path, in the form bash reads.
        """
        env = root / "env"
        (env / "bin").mkdir(parents=True)
        python = env / "bin" / "python"
        interpreter = shlex.quote(pathlib.Path(sys.executable).as_posix())
        python.write_text(f'#!/bin/sh\nexec {interpreter} "$@"\n', encoding="utf-8")
        python.chmod(0o755)
        if stamped:
            (env / "GIT_COMMIT").write_text(f"{_V23_COMMIT}\n", encoding="utf-8")
        return env.as_posix()

    def _run(self, line: str) -> subprocess.CompletedProcess[str]:
        """Run a probe line in a real bash.

        Args:
            line: The probe.

        Returns:
            The finished process.
        """
        return subprocess.run(
            [posix_bash(), "-c", line],
            capture_output=True,
            text=True,
            check=False,
            timeout=_WALL_SECONDS,
        )

    def test_each_module_resolves_the_way_python_m_does(self, tmp_path: pathlib.Path) -> None:
        env = self._environment(tmp_path, stamped=True)
        checkout = tmp_path / "checkout"
        (checkout / "scripts").mkdir(parents=True)
        (checkout / "scripts" / "__init__.py").write_text("", encoding="utf-8")
        (checkout / "scripts" / "optimize.py").write_text("", encoding="utf-8")
        work = tmp_path / "work"
        work.mkdir()
        found = [
            (checkout.as_posix(), "scripts.optimize"),
            (work.as_posix(), "scripts.optimize"),
            (work.as_posix(), "json.tool"),
            (work.as_posix(), "venv"),
            (work.as_posix(), "json"),
            (work.as_posix(), "nope.mod"),
        ]
        result = self._run(probe_command(env, found))
        assert result.returncode == 0, result.stderr
        commit, present = parse_probe(result.stdout, found)
        assert commit == _V23_COMMIT
        # scripts.optimize is only there from the checkout it cds into; json
        # is a package with no __main__.py, which `python -m json` refuses.
        assert present == {
            "scripts.optimize": False,
            "json.tool": True,
            "venv": True,
            "json": False,
            "nope.mod": False,
        }

    def test_an_unstamped_environment_reports_an_empty_commit(self, tmp_path: pathlib.Path) -> None:
        env = self._environment(tmp_path, stamped=False)
        found = [(tmp_path.as_posix(), "json.tool")]
        result = self._run(probe_command(env, found))
        assert parse_probe(result.stdout, found) == ("", {"json.tool": True})

    def test_a_directory_that_is_not_there_fails_the_line(self, tmp_path: pathlib.Path) -> None:
        """The failure the cluster gave for an absent script directory: the
        cd stops the chain, the exit is non-zero, and the answer is short."""
        env = self._environment(tmp_path, stamped=True)
        found = [((tmp_path / "absent").as_posix(), "json.tool")]
        result = self._run(probe_command(env, found))
        assert result.returncode != 0
        with pytest.raises(AppError) as excinfo:
            parse_probe(result.stdout, found)
        assert excinfo.value.code is Hpc3ErrorCode.ENTRYPOINT_PROBE_UNREADABLE


_ONE_MODULE = (("/w", "a"),)
"""One module asked about, for the answers that cannot be read."""


class TestReadingTheAnswer:
    def test_an_answer_without_the_stamp_line_is_unreadable(self) -> None:
        with pytest.raises(AppError) as excinfo:
            parse_probe("ENTRYPOINT|PRESENT|a\n", _ONE_MODULE)
        assert excinfo.value.code is Hpc3ErrorCode.ENTRYPOINT_PROBE_UNREADABLE

    def test_a_verdict_word_it_does_not_know_is_unreadable(self) -> None:
        with pytest.raises(AppError) as excinfo:
            parse_probe(_answer("c", ("MAYBE", "a")), _ONE_MODULE)
        assert excinfo.value.code is Hpc3ErrorCode.ENTRYPOINT_PROBE_UNREADABLE
        assert "'ENTRYPOINT|MAYBE|a'" in excinfo.value.message

    def test_a_verdict_with_an_extra_field_is_unreadable(self) -> None:
        with pytest.raises(AppError) as excinfo:
            parse_probe("ENTRYPOINT-COMMIT|c\nENTRYPOINT|PRESENT|x|a\n", _ONE_MODULE)
        assert excinfo.value.code is Hpc3ErrorCode.ENTRYPOINT_PROBE_UNREADABLE

    def test_a_module_absent_from_any_directory_it_starts_in_is_absent(self) -> None:
        found = [("/one", "a"), ("/two", "a")]
        answer = _answer("c", ("PRESENT", "a"), ("ABSENT", "a"))
        assert parse_probe(answer, found) == ("c", {"a": False})


class TestTheRegisteredMiImage:
    """The real current case, with the answers v23 gave on 2026-10-16."""

    def test_a_command_v23_predates_is_refused(self, fake_run: FakeRun) -> None:
        fake_run.add("ENTRYPOINT-COMMIT", stdout=_answer(_V23_COMMIT, ("ABSENT", _SWEEP)))
        with pytest.raises(AppError) as excinfo:
            _check([f"python -m {_SWEEP} --plan gpt2-small --device cuda"])
        assert excinfo.value.code is Hpc3ErrorCode.ENTRYPOINT_NOT_INSTALLED
        assert f"was built from {_V23_COMMIT}" in excinfo.value.message
        assert "inside /pub/wagnera3/images/v23/abl.sif" in excinfo.value.message
        (command,) = fake_run.commands()
        assert "apptainer exec" in command
        assert _SWEEP in command

    def test_the_old_command_it_was_built_for_still_runs(self, fake_run: FakeRun) -> None:
        """Reproducing an old command under the old image stays legal."""
        fake_run.add("ENTRYPOINT-COMMIT", stdout=_answer(_V23_COMMIT, ("PRESENT", _KNOWN_ANSWER)))
        assert _check([f"python -m {_KNOWN_ANSWER} --device cuda"]) == {
            "image_commit": _V23_COMMIT,
            "modules": [_KNOWN_ANSWER],
        }

    def test_members_starting_one_module_ask_about_it_once(self, fake_run: FakeRun) -> None:
        """Every member of an array usually starts the same entry point."""
        fake_run.add("ENTRYPOINT-COMMIT", stdout=_answer(_V23_COMMIT, ("PRESENT", _KNOWN_ANSWER)))
        reading = _check([f"python -m {_KNOWN_ANSWER} --seed {seed}" for seed in (1, 2, 3)])
        assert reading["modules"] == [_KNOWN_ANSWER]
        (command,) = fake_run.commands()
        assert command.count(_KNOWN_ANSWER) == 1

    def test_an_unstamped_environment_says_so(self, fake_run: FakeRun) -> None:
        fake_run.add("ENTRYPOINT-COMMIT", stdout=_answer("", ("ABSENT", _SWEEP)))
        with pytest.raises(AppError) as excinfo:
            _check([f"python -m {_SWEEP}"], image=None)
        assert "carries no commit stamp" in excinfo.value.message
        assert "on hpc3" in excinfo.value.message
        assert "apptainer" not in fake_run.commands()[0]

    def test_a_command_starting_no_module_costs_no_round_trip(self, fake_run: FakeRun) -> None:
        """And the reading says it examined none, rather than passing."""
        assert _check(["python train.py --out /pub/x.pt"]) == {"image_commit": "", "modules": []}
        assert fake_run.calls == []


class TestBothSubmissionPathsAskBeforeTheScheduler:
    """Preflight for a job, array preflight for a sweep: refused before sbatch."""

    def _template(self, command: str) -> dict[str, JSONValue]:
        """Build an ``mi`` job document under v23 running a command.

        Args:
            command: The command line.

        Returns:
            The document, before decoding.
        """
        return {
            "project": "mi",
            "name": "composition",
            "partition": "free-gpu",
            "gpu": gpus("A100"),
            "cpus": 8,
            "mem_gb": 64,
            "minutes": 60,
            "requeue": False,
            "resumes_from_checkpoint": False,
            "env_path": "/opt/env",
            "image": image_reference(path=_V23["path"], digest=_V23["sha256"]),
            "pinned_packages": {},
            "deterministic": False,
            "experiment": {"n": "8"},
            "command": command,
            "artifact": None,
        }

    def _job(self, command: str) -> JobSpec:
        """Build an ``mi`` job under v23 running a command.

        Args:
            command: The command line.

        Returns:
            A validated spec.
        """
        return decode_job_spec(self._template(command))

    def test_preflight_refuses_a_command_its_image_predates(self, fake_run: FakeRun) -> None:
        fake_run.add("ENTRYPOINT-COMMIT", stdout=_answer(_V23_COMMIT, ("ABSENT", _SWEEP)))
        script_healthy_environment(fake_run)
        with pytest.raises(AppError) as excinfo:
            preflight(
                self._job(f"python -m {_SWEEP} --plan gpt2-small"),
                host="hpc3",
                script_dir=_SCRIPTS,
                log_dir="/pub/wagnera3/mi/logs",
                cluster=cluster(),
                charge_account="",
            )
        assert excinfo.value.code is Hpc3ErrorCode.ENTRYPOINT_NOT_INSTALLED
        commands = fake_run.commands()
        assert any(f"mkdir -p '{_SCRIPTS}'" in command for command in commands)
        assert f"cd {_SCRIPTS} &&" in commands[-1]
        assert not any("--test-only" in command for command in commands)

    def test_preflight_admits_the_command_its_image_carries(self, fake_run: FakeRun) -> None:
        fake_run.add("ENTRYPOINT-COMMIT", stdout=_answer(_V23_COMMIT, ("PRESENT", _KNOWN_ANSWER)))
        script_healthy_cluster(fake_run)
        result = preflight(
            self._job(f"python -m {_KNOWN_ANSWER} --device cuda"),
            host="hpc3",
            script_dir=_SCRIPTS,
            log_dir="/pub/wagnera3/mi/logs",
            cluster=cluster(),
            charge_account="",
        )
        assert result["partition"] == "free-gpu"

    def test_an_array_is_refused_for_one_members_module(
        self, tmp_path: pathlib.Path, fake_run: FakeRun
    ) -> None:
        """Members share the environment but each may start its own module."""
        members: list[JSONValue] = [
            {"suffix": "n2", "command": f"python -m {_KNOWN_ANSWER} --n 2", "artifact": None},
            {"suffix": "n8", "command": f"python -m {_SWEEP} --n 8", "artifact": None},
        ]
        spec = decode_sweep_spec({"base": self._template("python train.py"), "members": members})
        fake_run.add(
            "ENTRYPOINT-COMMIT",
            stdout=_answer(_V23_COMMIT, ("PRESENT", _KNOWN_ANSWER), ("ABSENT", _SWEEP)),
        )
        script_healthy_cluster(fake_run)
        with pytest.raises(AppError) as excinfo:
            submit_array(
                spec,
                (0, 1),
                host="hpc3",
                script_dir=_SCRIPTS,
                log_dir="/pub/wagnera3/mi/logs",
                ledger_path=tmp_path / "ledger.jsonl",
                submitted_at="2026-10-16T09:00:00+00:00",
                submitter="",
                cluster=cluster(),
                charge_account="",
            )
        assert excinfo.value.code is Hpc3ErrorCode.ENTRYPOINT_NOT_INSTALLED
        assert _SWEEP in excinfo.value.message
        assert _KNOWN_ANSWER not in excinfo.value.message
        assert not any("sbatch" in command for command in fake_run.commands())
