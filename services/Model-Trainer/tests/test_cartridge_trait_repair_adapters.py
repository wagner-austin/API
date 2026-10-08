"""The trait-repair families behind an adapted base, on a real tiny GPT-2.

The arms, the crowding pool, the PEFT adapter and both of its objectives are
REAL; the faked seams are the hub loaders, the trait and corpus readers and
the plan tables (``tests/_trait_repair_support.py``).

THE PROPERTY THAT MATTERS MOST is asserted against another entry point: the
adapter a trait-repair record is measured on must be the adapter the corpus
arc's own sweep builds from the same pool, or the question "does the recorded
lever transfer" is asked of a different lever. Its epoch rows are compared
bit for bit with the corpus sweep's.

Split from ``test_cartridge_trait_repair_sweep.py`` (MCPs board task
2f90d785) so these runs take their own xdist worker.
"""

from __future__ import annotations

import pathlib
from collections.abc import Generator

import pytest

from model_trainer.cli import cartridge_base_lora_sweep as corpus_sweep
from model_trainer.core.services.model.cartridge_trait_repair_plans import TraitRepairAdapter
from tests._trait_repair_support import (
    TINY_CROWD,
    install_repair_fakes,
    measured_values,
    restore_repair_hooks,
    staged_pool,
)

#: The module-scoped language-modeling run is shared by two tests, so this
#: file's tests run on one xdist worker (tests/test_xdist_grouping.py says why).
pytestmark = pytest.mark.xdist_group("test_cartridge_trait_repair_adapters.py")


@pytest.fixture(name="wired", autouse=True)
def _wired() -> Generator[None, None, None]:
    """Install the fakes, and put the real hooks back afterwards.

    Yields:
        None, once the fakes are installed.
    """
    install_repair_fakes()
    yield None
    restore_repair_hooks()


@pytest.fixture(name="language_modeling", scope="module")
def _language_modeling(tmp_path_factory: pytest.TempPathFactory) -> dict[str, float]:
    """Measure the language-modeling adapter's plan once, for the two tests reading it.

    The two tests each ran this same plan until MCPs board task 2f90d785
    (17.7 s and, beside the corpus sweep, 29.1 s in CI job 113153367247).
    Module-scoped, so it installs the fakes itself, ahead of ``wired``.

    Args:
        tmp_path_factory: Source of the run's own directory.

    Returns:
        Every row's value, keyed by name.
    """
    install_repair_fakes()
    try:
        return measured_values(tmp_path_factory.mktemp("lm"), TraitRepairAdapter.LANGUAGE_MODELING)
    finally:
        restore_repair_hooks()


class TestTheAdaptedBases:
    """Both families, behind the adapter the corpus arc recorded."""

    def test_the_language_modeling_adapter_runs_both_families(
        self, language_modeling: dict[str, float]
    ) -> None:
        """Plain and diverse cells, and the adapter's convergence rows."""
        values = language_modeling
        assert values["lora-train-epoch-0_loss"] > 0.0
        assert values["lora-plain_composed_noise_floor"] >= 0.0
        assert values["lora-diverse_composed_noise_floor"] >= 0.0
        assert values["bullets-lora-plain-n2-alone-expression_spread"] >= 0.0

    def test_the_invariance_adapter_records_its_distillation(self, tmp_path: pathlib.Path) -> None:
        """The KL rows are the distillation's own, under the recorded name.

        Args:
            tmp_path: The test's temporary directory.
        """
        values = measured_values(tmp_path, TraitRepairAdapter.CROWD_INVARIANCE)
        assert values["invariance-train-epoch-0_kl"] > 0.0
        assert not [name for name in values if name.startswith("lora-train-epoch")]

    def test_the_adapter_is_the_one_the_corpus_sweep_builds(
        self, tmp_path: pathlib.Path, language_modeling: dict[str, float]
    ) -> None:
        """Same pool, same plan row: bit-identical epoch rows from both entries.

        The corpus sweep's primary corpus yields eight training windows, the
        count the repair plan declares, so both truncate the pool alike.

        Args:
            tmp_path: The test's temporary directory.
            language_modeling: The repair entry's rows behind the same adapter.
        """
        corpus_root = tmp_path / "corpus"
        corpus_root.mkdir()
        primary, beta, gamma = (corpus_root / name for name in ("alpha", "beta", "gamma"))
        for path in (primary, beta, gamma):
            path.mkdir()
        corpus_rows, _digest = corpus_sweep.measure_grid(
            TINY_CROWD,
            plan_name="tiny",
            corpus=primary,
            other_corpora=[beta, gamma],
            pool_corpora=staged_pool(corpus_root),
            device="cpu",
            checkpoints=corpus_root / "checkpoints",
        )
        corpus_values = {row["name"]: row["value"] for row in corpus_rows}
        assert (
            language_modeling["lora-train-epoch-0_loss"] == corpus_values["lora-train-epoch-0_loss"]
        )
