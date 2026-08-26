"""The trainer honours the determinism posture a launcher states.

This closes a real half-open seam. The submitter could declare a run
deterministic or not, export the posture into the batch script, and record it
in a ledger -- and the trainer read none of it, pinning determinism
unconditionally. A run declared OFF would have been deterministic anyway, and
the ledger would have said otherwise.

The record of what happened has to come from this side. A launcher declares;
only the process making the torch calls knows whether they happened.
"""

from __future__ import annotations

from collections.abc import Generator, Mapping

import pytest
from platform_core.config import _test_hooks as config_hooks
from platform_core.determinism_env import (
    CUBLAS_DETERMINISTIC_WORKSPACE,
    DETERMINISM_ENV_VAR,
    DETERMINISM_OFF,
    DETERMINISM_ON,
)
from platform_core.determinism_record import (
    FALSE,
    TRUE,
    UNPINNED_STACK,
    DeterminismRecord,
    determinism_record,
)
from platform_ml import TORCH_STACK
from platform_ml.determinism import TORCH_THREAD_SETTING, with_torch_thread_count

from model_trainer.core import _test_hooks
from model_trainer.core.config.settings import Settings
from model_trainer.worker.job_utils import setup_env
from tests.conftest import SettingsFactory

_REPORT = determinism_record(
    TORCH_STACK,
    {
        "deterministic_algorithms": TRUE,
        "cublas_workspace_config": CUBLAS_DETERMINISTIC_WORKSPACE,
        "matmul_tf32": FALSE,
        "cudnn_tf32": FALSE,
        "cudnn_deterministic": TRUE,
        "cudnn_benchmark": FALSE,
    },
)


class _CountingPin:
    """Records whether the determinism pin was actually invoked."""

    __slots__ = ("calls",)

    def __init__(self) -> None:
        """Start with no calls recorded."""
        self.calls = 0

    def __call__(self) -> DeterminismRecord:
        """Record an invocation and return a fixed report.

        Returns:
            A report standing in for what torch would have applied.
        """
        self.calls += 1
        return _REPORT


def _restore_env_hook() -> Generator[None, None, None]:
    """Put the config package's env reader back after each test.

    The hook is module-global. Left swapped it would answer for every later
    test in the same worker, which is the failure mode a shared hook has and
    a fixture is the whole of the fix.

    Yields:
        None, for the duration of one test.
    """
    original = config_hooks.get_env
    yield
    config_hooks.get_env = original


restore_env_hook = pytest.fixture(_restore_env_hook)


def _stated(environ: Mapping[str, str]) -> _CountingPin:
    """Point the config layer at a stated environment, and count the pin.

    Supplied through the config package's own ``get_env`` hook rather than by
    mutating ``os.environ`` -- that is the seam the monorepo's env guard
    exists to enforce.

    Args:
        environ: The environment the trainer should read.

    Returns:
        The pin, for asserting whether it ran.
    """
    pin = _CountingPin()
    _test_hooks.apply_determinism_hook = pin
    config_hooks.get_env = environ.get
    return pin


def _settings(settings_factory: SettingsFactory) -> Settings:
    """Build settings with a fixed thread count.

    Built BEFORE the environment hook is swapped: loading settings reads the
    environment itself, and an environment stating only a determinism posture
    would answer None to everything else it needs.

    Args:
        settings_factory: The suite's settings builder.

    Returns:
        Settings for setup_env.
    """
    return settings_factory(threads=2)


@pytest.mark.usefixtures("restore_env_hook")
class TestPosture:
    def test_an_absent_variable_still_pins_determinism(
        self, settings_factory: SettingsFactory
    ) -> None:
        """The local worker predates any launcher and pinned unconditionally.
        Nothing about adding a launcher may change what it does."""
        settings = _settings(settings_factory)
        pin = _stated({})
        threads, record = setup_env(settings)
        assert threads == 2
        assert pin.calls == 1
        assert record == with_torch_thread_count(_REPORT, 2)

    def test_an_explicit_on_pins_determinism(self, settings_factory: SettingsFactory) -> None:
        settings = _settings(settings_factory)
        pin = _stated({DETERMINISM_ENV_VAR: DETERMINISM_ON})
        setup_env(settings)
        assert pin.calls == 1

    def test_an_explicit_off_does_not_pin_determinism(
        self, settings_factory: SettingsFactory
    ) -> None:
        """The half that was missing: before this, a run declared OFF was
        pinned anyway and the ledger said it was not."""
        settings = _settings(settings_factory)
        pin = _stated({DETERMINISM_ENV_VAR: DETERMINISM_OFF})
        setup_env(settings)
        assert pin.calls == 0

    def test_declining_still_returns_the_thread_count(
        self, settings_factory: SettingsFactory
    ) -> None:
        """The early return must not skip what the caller asked for."""
        settings = _settings(settings_factory)
        _stated({DETERMINISM_ENV_VAR: DETERMINISM_OFF})
        threads, _ = setup_env(settings)
        assert threads == 2

    def test_declining_records_that_nothing_was_pinned(
        self, settings_factory: SettingsFactory
    ) -> None:
        """The declining path must SAY so, not say nothing.

        A run deliberately left free and a run whose posture is unknown are
        the same thing to a later comparison, and both differ from a pinned
        one. Returning an empty record rather than None is what makes the
        manifest able to state it.

        "Nothing was pinned" is about the STACK. The thread count is pinned on
        this path too -- a job pins it to use the machine well, not to be
        reproducible -- so it belongs in the record even here. It used to be
        dropped, which made this branch return an empty settings map for a run
        that had just pinned something.
        """
        settings = _settings(settings_factory)
        _stated({DETERMINISM_ENV_VAR: DETERMINISM_OFF})

        _, record = setup_env(settings)

        assert record == with_torch_thread_count(determinism_record(UNPINNED_STACK, {}), 2)
        assert record["stack"] == UNPINNED_STACK
        assert record != _REPORT

    def test_the_thread_count_reaches_the_record_on_both_paths(
        self, settings_factory: SettingsFactory
    ) -> None:
        """The count decides the numbers, so it has to be in the provenance.

        Measured on this stack: a 4096x4096 matmul at one thread and at eight
        differs in 865,498 of 16,777,216 elements. Before this, both paths
        returned a record that omitted the count entirely, so two runs at
        different thread counts had byte-identical provenance.
        """
        settings = _settings(settings_factory)

        _stated({DETERMINISM_ENV_VAR: DETERMINISM_ON})
        _, pinned = setup_env(settings)
        _stated({DETERMINISM_ENV_VAR: DETERMINISM_OFF})
        _, declined = setup_env(settings)

        assert dict(pinned["settings"])[TORCH_THREAD_SETTING] == "2"
        assert dict(declined["settings"])[TORCH_THREAD_SETTING] == "2"

    def test_an_unreadable_posture_fails_the_job_before_any_cuda_work(
        self, settings_factory: SettingsFactory
    ) -> None:
        """Raised from setup_env, which runs before the corpus is fetched, so
        the job fails rather than training under a posture nobody can name."""
        settings = _settings(settings_factory)
        pin = _stated({DETERMINISM_ENV_VAR: "true"})
        with pytest.raises(ValueError, match="Refusing to guess"):
            setup_env(settings)
        assert pin.calls == 0
