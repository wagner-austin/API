"""Tests for the measurement scripts' dependency-injection hooks.

Two different things are checked here, and the distinction matters.

The *shape* tests assert which hooks exist. They double as a drift guard: a hook
added without a test, or removed while a script still calls it, changes this
list.

The *conformance* tests run the real production implementations against the real
Warp package. They are not testing Warp -- they are testing this repo's claim
that the Protocols in :mod:`scripts._test_hooks` describe the vendor's actual
surface. That claim is otherwise unchecked, because ``__import__`` behind a
Protocol annotation is exactly the construct that stops mypy from verifying it,
and a Protocol that had drifted from Warp would fail only at measurement time.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts._test_hooks import WARP_MODULE, WarpModuleProtocol

from navprobe.adapters.mjx_warp_state import MjWarpStateSimulatorFactory
from scripts import _test_hooks


class TestHookSurface:
    """The set of hooks the scripts may inject."""

    def test_exposes_exactly_the_expected_hooks(self) -> None:
        """A hook added or removed changes this list.

        Asserting the whole sorted set rather than membership means a new hook
        cannot be introduced without a decision recorded here.
        """
        exported = sorted(name for name in _test_hooks.__all__ if not name.endswith("Protocol"))
        assert exported == ["init_warp", "load_state_factory", "monotonic", "write_out"]

    def test_binds_every_hook_at_import(self) -> None:
        """Production wires nothing; the bindings are live on import.

        A hook left unbound would make the ``if hook is not None`` branch this
        design exists to avoid necessary.
        """
        assert [
            callable(_test_hooks.init_warp),
            callable(_test_hooks.load_state_factory),
            callable(_test_hooks.write_out),
            callable(_test_hooks.monotonic),
        ] == [True, True, True, True]


class TestWriteOut:
    """The real standard-output hook."""

    def test_writes_the_text_it_was_given(self, capsys: pytest.CaptureFixture[str]) -> None:
        """The production implementation writes verbatim, with no formatting."""
        _test_hooks.write_out("scene bodies=6\n")
        assert capsys.readouterr().out == "scene bodies=6\n"


class TestMonotonic:
    """The real clock hook."""

    def test_advances(self) -> None:
        """A clock that never advanced would report every wall time as zero."""
        first = _test_hooks.monotonic()
        second = _test_hooks.monotonic()
        assert second >= first


class TestWarpConformance:
    """The Protocols describe the Warp package that is actually installed."""

    def test_initialises_warp_and_resolves_a_device(self, tmp_path: Path) -> None:
        """The real hook brings Warp up and returns something with its surface.

        Uses ``NOT_GUARANTEED``, which is Warp's default: setting a
        deterministic mode is process-global and would leak into every other
        test sharing this worker.
        """
        runtime = _test_hooks.init_warp("NOT_GUARANTEED", str(tmp_path / "cache"), 0)
        assert str(runtime.get_device("cpu")) == "cpu"

    def test_scopes_to_a_device(self, tmp_path: Path) -> None:
        """``ScopedDevice`` is a context manager, as the Protocol declares."""
        runtime = _test_hooks.init_warp("NOT_GUARANTEED", str(tmp_path / "cache"), 0)
        with runtime.ScopedDevice("cpu") as scoped:
            assert str(scoped) == "cpu"

    def test_an_absent_device_raises_rather_than_falling_back(self, tmp_path: Path) -> None:
        """The property the ``--device`` flag rests on.

        If Warp resolved an unknown identifier to the default device, a sweep
        labelled with a second card would carry the first card's numbers.
        """
        runtime = _test_hooks.init_warp("NOT_GUARANTEED", str(tmp_path / "cache"), 0)
        with pytest.raises(ValueError, match="Invalid device identifier"):
            runtime.get_device("cuda:99")

    def test_applies_a_deterministic_mode_and_a_record_bound(self, tmp_path: Path) -> None:
        """The two settings the determinism findings rest on reach Warp.

        Both are process-global, so the previous values are put back before the
        test returns: leaving ``deterministic`` set would silently change how
        every module compiled later in this worker is lowered, which is the
        one thing this package must not do by accident.

        Nothing is compiled here. The settings are asserted on Warp's own
        config object, which is where the script's contract with the vendor
        actually lives -- a hook that read them and dropped them would pass
        every other test in this file.
        """
        warp: WarpModuleProtocol = __import__(WARP_MODULE, fromlist=["config", "DeterministicMode"])
        saved_mode = warp.config.deterministic
        saved_records = warp.config.deterministic_max_records
        try:
            _test_hooks.init_warp("RUN_TO_RUN", str(tmp_path / "cache"), 64)
            applied = (warp.config.deterministic, warp.config.deterministic_max_records)
        finally:
            warp.config.deterministic = saved_mode
            warp.config.deterministic_max_records = saved_records
        assert applied == (warp.DeterministicMode.RUN_TO_RUN, 64)

    def test_leaves_the_default_mode_alone(self, tmp_path: Path) -> None:
        """``NOT_GUARANTEED`` is Warp's default and is not written back.

        The branch that matters: a hook that set the mode unconditionally would
        pin every default-mode run to an explicitly-set value, and the "only
        mode available" measurements would no longer be of the default.
        """
        warp: WarpModuleProtocol = __import__(WARP_MODULE, fromlist=["config"])
        saved_records = warp.config.deterministic_max_records
        try:
            _test_hooks.init_warp("NOT_GUARANTEED", str(tmp_path / "cache"), 0)
            applied = warp.config.deterministic_max_records
        finally:
            warp.config.deterministic_max_records = saved_records
        assert applied == saved_records

    def test_loads_the_state_factory_constructor(self) -> None:
        """The hook returns the adapter the measurements are taken with.

        Asserted by identity against the imported class rather than by name, so
        a hook that pointed at some other constructor with a matching signature
        would fail here.
        """
        assert _test_hooks.load_state_factory() is MjWarpStateSimulatorFactory
