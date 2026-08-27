"""Gating and registering, exercised as real code against real files.

Nothing is faked. The registry is a JSON file, so the tests write one in
tmp_path and run the production functions over it.
"""

from __future__ import annotations

import pathlib
import runpy
import sys

import pytest
from platform_core.comparability import RunFingerprint
from platform_core.determinism_record import DeterminismRecord
from platform_core.json_utils import dump_json_str
from platform_core.known_answer import KnownAnswer
from platform_core.known_answer_registry import encode_registry, read_registry
from platform_core.run_record import Observation, RunRecord, encode_run_record, run_record
from platform_core.testing import sample_run_fingerprint

from model_trainer.cli import known_answer_registry as registry_cli

_PINNED = DeterminismRecord(stack="torch", settings=(("matmul_tf32", "false"),))

_A100 = sample_run_fingerprint(
    image_digest="a" * 64,
    gpu_model="NVIDIA A100 80GB PCIe",
    driver_version="580.82.07",
    determinism=_PINNED,
)

_V100 = sample_run_fingerprint(
    image_digest="a" * 64,
    gpu_model="Tesla V100-FHHL-16GB",
    driver_version="580.82.07",
    determinism=_PINNED,
)

_LABEL = "probe-label"
_VALUE = 6.250983715057373


def _record(fingerprint: RunFingerprint, value: float = _VALUE) -> RunRecord:
    """Build a single-observation record."""
    return run_record(
        experiment="environment-known-answer",
        label=_LABEL,
        fingerprint=fingerprint,
        observations=(Observation(name="probe_loss", value=value),),
        payload_digest="",
    )


def _entry(
    fingerprint: RunFingerprint, expected: float = _VALUE, tolerance: float = 0.0
) -> KnownAnswer:
    """Build an entry."""
    return KnownAnswer(
        label=_LABEL, fingerprint=fingerprint, expected=expected, tolerance=tolerance
    )


def _write_registry(path: pathlib.Path, *answers: KnownAnswer) -> pathlib.Path:
    """Write a registry file and return its path."""
    path.write_text(encode_registry(tuple(answers)), encoding="utf-8")
    return path


def _write_record(path: pathlib.Path, record: RunRecord) -> pathlib.Path:
    """Write a run record file and return its path."""
    path.write_text(dump_json_str(encode_run_record(record)), encoding="utf-8")
    return path


class TestLoadRecord:
    """Reading a record off disk."""

    def test_it_decodes_what_a_probe_wrote(self, tmp_path: pathlib.Path) -> None:
        path = _write_record(tmp_path / "r.json", _record(_A100))

        assert registry_cli.load_record(path)["observations"][0]["value"] == _VALUE


class TestDiscrimination:
    """An entry that cannot fail is not a gate."""

    def test_a_sound_entry_reports_no_failures(self) -> None:
        assert registry_cli.discrimination_failures(_entry(_A100)) == ()

    def test_an_entry_that_cannot_match_itself_is_reported(self) -> None:
        # A negative tolerance admits no value at all, so the entry can only
        # ever report a deviation -- it would read as a broken image forever.
        failures = registry_cli.discrimination_failures(_entry(_A100, tolerance=-1.0))

        assert len(failures) == 1
        assert "does not match its own measurement" in failures[0]

    def test_an_entry_too_slack_to_notice_drift_is_reported(self) -> None:
        failures = registry_cli.discrimination_failures(_entry(_A100, tolerance=1e-6))

        assert len(failures) == 1
        assert "does not fire on a drift" in failures[0]

    def test_an_entry_that_cannot_see_a_card_change_is_reported(self) -> None:
        # An entry whose own card IS the control card cannot distinguish one.
        blind = sample_run_fingerprint(
            image_digest="a" * 64,
            gpu_model=registry_cli._CONTROL_CARD,
            driver_version="580.82.07",
            determinism=_PINNED,
        )

        failures = registry_cli.discrimination_failures(_entry(blind))

        assert len(failures) == 1
        assert "does not treat a card change as a move" in failures[0]


class TestParseTolerance:
    """The argument is validated before conversion, so conversion cannot fail."""

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("0", 0.0), ("0.0", 0.0), ("1e-9", 1e-9), ("1E-9", 1e-9), ("2.5e+3", 2500.0)],
    )
    def test_it_accepts_a_non_negative_number(self, raw: str, expected: float) -> None:
        assert registry_cli.parse_tolerance(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        ["loose", "", "1e", ".5", "5.", "1,0", "nan", "inf", " 1", "1 "],
        ids=[
            "word",
            "empty",
            "bare-exp",
            "no-int",
            "no-frac",
            "comma",
            "nan",
            "inf",
            "lead",
            "trail",
        ],
    )
    def test_it_refuses_anything_else(self, raw: str) -> None:
        with pytest.raises(ValueError, match="non-negative number"):
            registry_cli.parse_tolerance(raw)

    @pytest.mark.parametrize("raw", ["-1", "-0.5", "-1e-9"])
    def test_it_refuses_a_negative_tolerance_at_the_boundary(self, raw: str) -> None:
        # A negative tolerance admits no value at all, so an answer carrying
        # one could only ever report a deviation -- it would read as a broken
        # image rather than a broken answer.
        with pytest.raises(ValueError, match="non-negative number"):
            registry_cli.parse_tolerance(raw)


class TestGate:
    """Checking a record against what is registered."""

    def test_a_matching_run_returns_zero(self, tmp_path: pathlib.Path) -> None:
        path = _write_registry(tmp_path / "k.json", _entry(_A100))

        assert registry_cli.run_gate(path, _record(_A100)) == 0

    def test_a_drifted_run_returns_one(self, tmp_path: pathlib.Path) -> None:
        path = _write_registry(tmp_path / "k.json", _entry(_A100))

        assert registry_cli.run_gate(path, _record(_A100, _VALUE + 1e-9)) == 1

    def test_an_uncovered_configuration_returns_one(self, tmp_path: pathlib.Path) -> None:
        # "Nothing to compare against" is NOT a pass. Returning 0 here would
        # let an unregistered card look verified.
        path = _write_registry(tmp_path / "k.json", _entry(_A100))

        assert registry_cli.run_gate(path, _record(_V100)) == 1

    def test_an_empty_registry_returns_one(self, tmp_path: pathlib.Path) -> None:
        path = _write_registry(tmp_path / "k.json")

        assert registry_cli.run_gate(path, _record(_A100)) == 1

    def test_it_matches_against_the_right_entry_among_several(self, tmp_path: pathlib.Path) -> None:
        path = _write_registry(tmp_path / "k.json", _entry(_V100), _entry(_A100))

        assert registry_cli.run_gate(path, _record(_A100)) == 0

    def test_gating_changes_nothing(self, tmp_path: pathlib.Path) -> None:
        path = _write_registry(tmp_path / "k.json", _entry(_A100))
        before = path.read_text(encoding="utf-8")

        registry_cli.run_gate(path, _record(_A100))

        assert path.read_text(encoding="utf-8") == before


class TestRegister:
    """Establishing an entry, and the refusals that keep the registry honest."""

    def test_it_appends_a_verified_entry(self, tmp_path: pathlib.Path) -> None:
        path = _write_registry(tmp_path / "k.json")

        assert registry_cli.run_register(path, _record(_A100), 0.0) == 0

        assert read_registry(path) == (_entry(_A100),)

    def test_it_preserves_existing_entries(self, tmp_path: pathlib.Path) -> None:
        path = _write_registry(tmp_path / "k.json", _entry(_V100))

        registry_cli.run_register(path, _record(_A100), 0.0)

        assert read_registry(path) == (_entry(_V100), _entry(_A100))

    def test_re_registering_the_identical_entry_is_a_no_op(self, tmp_path: pathlib.Path) -> None:
        path = _write_registry(tmp_path / "k.json", _entry(_A100))

        assert registry_cli.run_register(path, _record(_A100), 0.0) == 0

        assert read_registry(path) == (_entry(_A100),)

    def test_a_conflicting_value_for_one_configuration_is_refused(
        self, tmp_path: pathlib.Path
    ) -> None:
        # Two answers for one configuration is worse than none: a later gate
        # would match or deviate depending on which it read first.
        path = _write_registry(tmp_path / "k.json", _entry(_A100, expected=1.0))

        with pytest.raises(ValueError, match="already registered"):
            registry_cli.run_register(path, _record(_A100), 0.0)

    def test_an_entry_that_cannot_discriminate_is_refused(self, tmp_path: pathlib.Path) -> None:
        path = _write_registry(tmp_path / "k.json")

        with pytest.raises(ValueError, match="does not discriminate"):
            registry_cli.run_register(path, _record(_A100), 1e-6)

    def test_an_incomplete_fingerprint_is_refused(self, tmp_path: pathlib.Path) -> None:
        path = _write_registry(tmp_path / "k.json")
        blank_driver = sample_run_fingerprint(
            image_digest="a" * 64,
            gpu_model="NVIDIA A100 80GB PCIe",
            driver_version="",
            determinism=_PINNED,
        )

        with pytest.raises(ValueError, match="driver_version"):
            registry_cli.run_register(path, _record(blank_driver), 0.0)

    def test_a_refused_registration_leaves_the_file_untouched(self, tmp_path: pathlib.Path) -> None:
        path = _write_registry(tmp_path / "k.json", _entry(_V100))
        before = path.read_text(encoding="utf-8")

        with pytest.raises(ValueError):
            registry_cli.run_register(path, _record(_A100), 1e-6)

        assert path.read_text(encoding="utf-8") == before


class TestTheCommandLine:
    """Modes, and the refusals that stop a mistyped command from writing."""

    def _cmd(self, tmp_path: pathlib.Path, mode: str, *extra: str) -> list[str]:
        """Build a command line against files in tmp_path."""
        registry = _write_registry(tmp_path / "k.json", _entry(_A100))
        record = _write_record(tmp_path / "r.json", _record(_A100))
        return [
            "--registry",
            str(registry),
            "--record",
            str(record),
            "--mode",
            mode,
            *extra,
        ]

    def test_gate_mode_returns_the_gate_result(self, tmp_path: pathlib.Path) -> None:
        assert registry_cli.main(self._cmd(tmp_path, "gate")) == 0

    def test_register_mode_registers(self, tmp_path: pathlib.Path) -> None:
        registry = _write_registry(tmp_path / "k.json")
        record = _write_record(tmp_path / "r.json", _record(_A100))

        code = registry_cli.main(
            [
                "--registry",
                str(registry),
                "--record",
                str(record),
                "--mode",
                "register",
                "--tolerance",
                "0",
            ]
        )

        assert code == 0
        assert read_registry(registry) == (_entry(_A100),)

    def test_an_unknown_mode_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="must be one of"):
            registry_cli.main(self._cmd(tmp_path, "clobber"))

    def test_a_non_numeric_tolerance_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="must be a non-negative number such as"):
            registry_cli.main(self._cmd(tmp_path, "register", "--tolerance", "loose"))

    def test_register_without_a_tolerance_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="--tolerance"):
            registry_cli.main(self._cmd(tmp_path, "register"))

    def test_an_absent_registry_flag_is_refused(self, tmp_path: pathlib.Path) -> None:
        record = _write_record(tmp_path / "r.json", _record(_A100))

        with pytest.raises(ValueError, match="--registry"):
            registry_cli.main(["--record", str(record), "--mode", "gate"])

    def test_the_console_entry_point_carries_the_exit_code(self, tmp_path: pathlib.Path) -> None:
        saved = sys.argv
        sys.argv = ["modeltrainer-known-answer-registry", *self._cmd(tmp_path, "gate")]
        try:
            with pytest.raises(SystemExit) as excinfo:
                registry_cli.entrypoint()
        finally:
            sys.argv = saved

        assert excinfo.value.code == 0

    def test_running_the_module_as_main_actually_gates(self, tmp_path: pathlib.Path) -> None:
        # Without the __main__ guard, `python -m` imports the module, runs
        # nothing and exits 0. That shipped once already in the sibling probe
        # CLI, and two Slurm jobs reported success having done nothing -- so
        # the guard is EXECUTED here rather than asserted to exist.
        module_name = "model_trainer.cli.known_answer_registry"
        argv = self._cmd(tmp_path, "gate")
        saved_argv = sys.argv
        saved_module = sys.modules.pop(module_name, None)
        sys.argv = ["modeltrainer-known-answer-registry", *argv]
        try:
            with pytest.raises(SystemExit) as excinfo:
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            sys.argv = saved_argv
            if saved_module is not None:
                sys.modules[module_name] = saved_module

        assert excinfo.value.code == 0
