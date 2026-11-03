"""The toolchain probe's ``nixos-test`` question, run for real (MCPs board task 3268afdf).

:data:`fleet.contracts.tagged_tools.NIXOS_TEST_PROGRAM` is run under ``sh -c``
as the probe runs it inside a check's unit, with a stand-in ``nix`` first on
the PATH and a temporary file in place of ``/dev/kvm``, so every way a node
can fail it is shown answering nonzero and only lavender's shape answers
nix's version. The probe body's own line is pinned beside it.
"""

from __future__ import annotations

import pathlib
import subprocess

import pytest

from fleet.contracts.tagged_tools import KVM_DEVICE, NIXOS_TEST_FEATURES, NIXOS_TEST_PROGRAM
from fleet.core.dialect import SH_INVOCATION
from fleet.core.linux_toolchain_probe import TOOLCHAIN_PROBE_BODY

#: lavender's nix system features on 2026-11-03.
LAVENDER_FEATURES = "benchmark big-parallel kvm nixos-test"

#: What the stand-in nix answers to ``--version``.
NIX_VERSION = "nix (Nix) 2.31.2"


def _nix(bin_dir: pathlib.Path, features: str) -> None:
    """Write a stand-in ``nix`` that answers the two questions the program asks.

    Args:
        bin_dir: The directory put first on the PATH.
        features: What it prints for ``config show system-features``.
    """
    bin_dir.mkdir(exist_ok=True)
    nix = bin_dir / "nix"
    nix.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = --version ]; then echo "' + NIX_VERSION + '"; exit 0; fi\n'
        'if [ "$*" = "--extra-experimental-features nix-command config show system-features" ];'
        ' then echo "' + features + '"; exit 0; fi\n'
        "exit 64\n",
        encoding="utf-8",
    )
    nix.chmod(0o755)


def _device(tmp_path: pathlib.Path, mode: int | None) -> pathlib.Path:
    """A stand-in for ``/dev/kvm``.

    Args:
        tmp_path: Where it goes.
        mode: Its permission bits, or None for a node with no such device.

    Returns:
        Its path.
    """
    device = tmp_path / "kvm"
    if mode is not None:
        device.write_bytes(b"")
        device.chmod(mode)
    return device


def _ask(
    tmp_path: pathlib.Path, device: pathlib.Path, *, with_nix: bool
) -> subprocess.CompletedProcess[str]:
    """Run the program as the probe does, against the stand-ins.

    Args:
        tmp_path: Where the stand-in nix's directory goes.
        device: The stand-in device.
        with_nix: Whether a stand-in nix was written into that directory;
            without one the directory is made empty.

    Returns:
        The finished program.
    """
    # The program runs only shell builtins and nix, so its PATH is exactly the
    # stand-in's directory: with the stand-in it is the nix asked, and without
    # it no nix is found, whatever this host carries.
    bin_dir = tmp_path / "bin"
    if not with_nix:
        bin_dir.mkdir()
    program = NIXOS_TEST_PROGRAM.replace(KVM_DEVICE, device.as_posix())
    return subprocess.run(
        [*SH_INVOCATION, "-c", f"PATH='{bin_dir.as_posix()}'\n{program}"],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


@pytest.mark.host_linux
class TestTheProgramUnderSh:
    def test_lavender_s_shape_answers_nix_s_version(self, tmp_path: pathlib.Path) -> None:
        _nix(tmp_path / "bin", LAVENDER_FEATURES)

        completed = _ask(tmp_path, _device(tmp_path, 0o666), with_nix=True)

        assert completed.returncode == 0, completed.stderr
        assert completed.stdout == f"{NIX_VERSION}\n"

    @pytest.mark.parametrize(
        "features",
        [
            "benchmark big-parallel nixos-test",
            "benchmark big-parallel kvm",
            "kvmx nixos-testing",
            "",
        ],
    )
    def test_a_feature_missing_or_only_a_prefix_answers_no(
        self, tmp_path: pathlib.Path, features: str
    ) -> None:
        """Each of kvm and nixos-test must be a whole word of the list."""
        _nix(tmp_path / "bin", features)

        completed = _ask(tmp_path, _device(tmp_path, 0o666), with_nix=True)

        assert completed.returncode != 0
        assert completed.stdout == ""

    def test_no_device_answers_no(self, tmp_path: pathlib.Path) -> None:
        """colossus on 2026-11-03: NixOS without /dev/kvm."""
        _nix(tmp_path / "bin", LAVENDER_FEATURES)

        completed = _ask(tmp_path, _device(tmp_path, None), with_nix=True)

        assert completed.returncode != 0
        assert completed.stdout == ""

    def test_a_device_the_build_account_cannot_write_answers_no(
        self, tmp_path: pathlib.Path
    ) -> None:
        _nix(tmp_path / "bin", LAVENDER_FEATURES)

        completed = _ask(tmp_path, _device(tmp_path, 0o444), with_nix=True)

        assert completed.returncode != 0
        assert completed.stdout == ""

    def test_no_nix_answers_no(self, tmp_path: pathlib.Path) -> None:
        """An Ubuntu node with /dev/kvm and no nix."""
        completed = _ask(tmp_path, _device(tmp_path, 0o666), with_nix=False)

        assert completed.returncode != 0
        assert completed.stdout == ""


def test_the_program_asks_both_features_and_the_device() -> None:
    assert KVM_DEVICE == "/dev/kvm"
    assert NIXOS_TEST_FEATURES == ("kvm", "nixos-test")
    assert NIXOS_TEST_PROGRAM.startswith("test -r /dev/kvm && test -w /dev/kvm && ")
    assert NIXOS_TEST_PROGRAM.endswith("nix --version")
    assert "'" not in NIXOS_TEST_PROGRAM


def test_the_probe_asks_it_inside_a_check_s_unit() -> None:
    """Where builds run, so /dev/kvm is read with the build's account and groups."""
    assert (
        f"if n=\"$(unit sh -c '{NIXOS_TEST_PROGRAM}' 2>/dev/null)\"; then\n"
        "  printf 'nixos-test=yes=%s\\n' \"$(printf '%s\\n' \"$n\" | head -n 1)\"\n"
        "else\n"
        "  printf 'nixos-test=no=\\n'\n"
        "fi\n"
    ) in TOOLCHAIN_PROBE_BODY
    assert TOOLCHAIN_PROBE_BODY.index("unit() {") < TOOLCHAIN_PROBE_BODY.index("nixos-test=")
