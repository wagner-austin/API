"""The record a finished training run writes beside its weights."""

from __future__ import annotations

import pathlib

import pytest
from platform_core.determinism_record import UNPINNED_STACK, determinism_record
from platform_core.run_record import NO_PAYLOAD

from model_trainer.core.contracts.model import QuantizationConfig, TrainOutcome
from model_trainer.core.services.training.run_records import (
    BASE_TRAINING_DISTRIBUTIONS,
    PEFT_DISTRIBUTION,
    QUANTIZATION_DISTRIBUTION,
    TRAINING_EXPERIMENT,
    TRANSFORMERS_DISTRIBUTION,
    model_distributions,
    saved_model_digest,
    training_distributions,
    training_observations,
    training_record_path,
    training_run_record,
    write_training_run_record,
)


class _Prepared:
    """Stands in for a prepared model, carrying only what the record reads.

    The real :class:`PreparedLMModel` holds a torch module. Building one would
    load weights to answer two boolean questions, so this exposes exactly the
    two attributes the library set depends on.
    """

    def __init__(self, *, is_peft: bool, quantized: bool) -> None:
        """Record what the model was prepared as.

        Args:
            is_peft: Whether adapters were attached.
            quantized: Whether the weights were quantized.
        """
        self.is_peft = is_peft
        self.quantization: QuantizationConfig | None = (
            QuantizationConfig(
                load_in_4bit=True,
                load_in_8bit=False,
                bnb_4bit_compute_dtype="bfloat16",
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            if quantized
            else None
        )


def _outcome(
    out_dir: str,
    *,
    test_loss: float | None = 0.9,
    test_perplexity: float | None = 2.47,
    best_val_loss: float | None = 0.81,
    early_stopped: bool = False,
    cancelled: bool = False,
) -> TrainOutcome:
    """Build a finished-training outcome.

    Written with explicit typed parameters rather than a kwargs splat, so the
    TypedDict is constructed once and checked. A splat would need a
    suppression to update a TypedDict from an untyped mapping, and this
    package does not carry those.

    Args:
        out_dir: Where the run saved.
        test_loss: Held-out loss, or None when the run had no test split.
        test_perplexity: Held-out perplexity, or None.
        best_val_loss: Best validation loss, or None.
        early_stopped: Whether early stopping fired.
        cancelled: Whether the run was cancelled.

    Returns:
        The outcome.
    """
    return TrainOutcome(
        loss=1.5,
        perplexity=4.5,
        steps=1633,
        out_dir=out_dir,
        cancelled=cancelled,
        test_loss=test_loss,
        test_perplexity=test_perplexity,
        best_val_loss=best_val_loss,
        early_stopped=early_stopped,
    )


class TestWhichLibrariesAreRecorded:
    """The set depends on the run, because the arithmetic does."""

    def test_a_char_lstm_run_records_neither_transformers_nor_peft(self) -> None:
        """Naming a distribution the environment lacks would raise, not record."""
        names = training_distributions(_Prepared(is_peft=False, quantized=False), "char_lstm")

        assert names == BASE_TRAINING_DISTRIBUTIONS

    def test_a_plain_hf_run_records_transformers(self) -> None:
        """It decides which attention path the model takes."""
        names = training_distributions(_Prepared(is_peft=False, quantized=False), "hf_lm")

        assert TRANSFORMERS_DISTRIBUTION in names
        assert PEFT_DISTRIBUTION not in names

    def test_an_adapter_run_records_peft(self) -> None:
        """PEFT decides which tensors exist and how they merge."""
        names = training_distributions(_Prepared(is_peft=True, quantized=False), "hf_lm")

        assert PEFT_DISTRIBUTION in names
        assert QUANTIZATION_DISTRIBUTION not in names

    def test_a_quantized_run_records_bitsandbytes(self) -> None:
        """It chooses the NF4 kernels, so its version moves the loss.

        This is the gap the training manifest had: it records numpy, torch
        and transformers for every run, so the QLoRA adapter trained on
        2026-09-01 carries no record of the two libraries that decided its
        arithmetic.
        """
        names = training_distributions(_Prepared(is_peft=True, quantized=True), "hf_lm")

        assert PEFT_DISTRIBUTION in names
        assert QUANTIZATION_DISTRIBUTION in names

    def test_an_evaluation_reaches_the_same_table_by_the_same_answers(self) -> None:
        """The table lives in one place, and a sweep reads it the other way round.

        Training reads the three answers off a prepared model; a continuation
        sweep reads them out of a saved run's metadata, because its base arm
        deliberately attaches no adapter and would otherwise report a
        narrower set than the arm it controls for. Two copies of the table
        would be two package axes that drift, and a drifted axis answers
        wrongly rather than obviously not at all.
        """
        through_training = training_distributions(_Prepared(is_peft=True, quantized=True), "hf_lm")
        through_metadata = model_distributions(
            uses_transformers=True, uses_peft=True, uses_quantization=True
        )

        assert through_metadata == through_training

    def test_a_run_using_none_of_the_three_records_only_the_base_set(self) -> None:
        """Named separately from the char_lstm case, which reaches it via a family."""
        names = model_distributions(
            uses_transformers=False, uses_peft=False, uses_quantization=False
        )

        assert names == BASE_TRAINING_DISTRIBUTIONS

    def test_no_distribution_is_named_twice(self) -> None:
        """capture_package_versions refuses a repeated name outright."""
        names = training_distributions(_Prepared(is_peft=True, quantized=True), "hf_lm")

        assert len(names) == len(set(names))


class TestTheObservations:
    """The numbers a later contrast reads."""

    def test_steps_travel_with_the_loss(self) -> None:
        """A loss at 1,633 steps and the same loss at 200 differ."""
        values = {o["name"]: o["value"] for o in training_observations(_outcome("out"))}

        assert values["train_loss"] == 1.5
        assert values["steps"] == 1633.0

    def test_the_held_out_figures_are_carried(self) -> None:
        """These are the ones worth comparing across runs."""
        values = {o["name"]: o["value"] for o in training_observations(_outcome("out"))}

        assert values["test_loss"] == 0.9
        assert values["best_val_loss"] == 0.81

    def test_an_absent_held_out_figure_is_omitted_not_zeroed(self) -> None:
        """A best_val_loss of 0.0 would be read as a measurement."""
        outcome = _outcome("out", best_val_loss=None, test_loss=None, test_perplexity=None)

        names = {o["name"] for o in training_observations(outcome)}

        assert "best_val_loss" not in names
        assert "test_loss" not in names

    def test_the_flags_are_recorded_as_numbers(self) -> None:
        """Whether a run early-stopped changes how its loss reads."""
        outcome = _outcome("out", early_stopped=True, cancelled=True)
        values = {o["name"]: o["value"] for o in training_observations(outcome)}

        assert values["early_stopped"] == 1.0
        assert values["cancelled"] == 1.0


class TestDigestingTheWeights:
    """The digest is what makes two runs checkable for bit-identity."""

    def test_the_same_weights_digest_the_same(self, tmp_path: pathlib.Path) -> None:
        """Otherwise no two runs could be shown to agree."""
        (tmp_path / "adapter.safetensors").write_bytes(b"weights")

        assert saved_model_digest(tmp_path) == saved_model_digest(tmp_path)

    def test_changed_weights_change_the_digest(self, tmp_path: pathlib.Path) -> None:
        """The property it exists for."""
        path = tmp_path / "adapter.safetensors"
        path.write_bytes(b"weights")
        before = saved_model_digest(tmp_path)
        path.write_bytes(b"other")

        assert saved_model_digest(tmp_path) != before

    def test_a_renamed_file_changes_the_digest(self, tmp_path: pathlib.Path) -> None:
        """Names are hashed with the bytes, so a rename is a difference."""
        (tmp_path / "adapter.safetensors").write_bytes(b"weights")
        before = saved_model_digest(tmp_path)
        (tmp_path / "adapter.safetensors").rename(tmp_path / "model.safetensors")

        assert saved_model_digest(tmp_path) != before

    def test_nested_files_are_covered(self, tmp_path: pathlib.Path) -> None:
        """A saved model is a directory, not one file."""
        (tmp_path / "adapter.safetensors").write_bytes(b"weights")
        before = saved_model_digest(tmp_path)
        nested = tmp_path / "checkpoint"
        nested.mkdir()
        (nested / "extra.bin").write_bytes(b"more")

        assert saved_model_digest(tmp_path) != before

    def test_an_empty_directory_records_no_payload(self, tmp_path: pathlib.Path) -> None:
        """The digest of nothing is a constant every empty run would share."""
        assert saved_model_digest(tmp_path) == NO_PAYLOAD


class TestTheRecord:
    """The whole record, as it lands beside the weights."""

    def _write(self, tmp_path: pathlib.Path) -> pathlib.Path:
        """Save a model directory.

        Args:
            tmp_path: Directory to save into.

        Returns:
            The directory.
        """
        (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
        return tmp_path

    def test_the_record_names_the_experiment_and_the_run(self, tmp_path: pathlib.Path) -> None:
        """Experiment pairs runs; label distinguishes them within it.

        Args:
            tmp_path: Directory for the saved model.
        """
        out_dir = self._write(tmp_path)

        record = training_run_record(
            _outcome(str(out_dir)),
            label="run-42",
            device="cpu",
            determinism=determinism_record(UNPINNED_STACK, {}),
            prepared=_Prepared(is_peft=False, quantized=False),
            model_family="char_lstm",
        )

        assert record["experiment"] == TRAINING_EXPERIMENT
        assert record["label"] == "run-42"

    def test_the_payload_digest_covers_the_weights(self, tmp_path: pathlib.Path) -> None:
        """The record must move when the weights do.

        Args:
            tmp_path: Directory for the saved model.
        """
        out_dir = self._write(tmp_path)

        record = training_run_record(
            _outcome(str(out_dir)),
            label="run-42",
            device="cpu",
            determinism=determinism_record(UNPINNED_STACK, {}),
            prepared=_Prepared(is_peft=False, quantized=False),
            model_family="char_lstm",
        )

        assert record["payload_digest"] == saved_model_digest(out_dir)

    def test_an_unpinned_run_records_that_honestly(self, tmp_path: pathlib.Path) -> None:
        """A null posture beats a missing key, which reads as forgotten.

        Args:
            tmp_path: Directory for the saved model.
        """
        out_dir = self._write(tmp_path)

        record = training_run_record(
            _outcome(str(out_dir)),
            label="run-42",
            device="cpu",
            determinism=determinism_record(UNPINNED_STACK, {}),
            prepared=_Prepared(is_peft=False, quantized=False),
            model_family="char_lstm",
        )

        assert record["fingerprint"]["determinism"]["stack"] == UNPINNED_STACK

    def test_a_cpu_run_records_no_card(self, tmp_path: pathlib.Path) -> None:
        """Querying one would describe hardware the run never touched.

        Args:
            tmp_path: Directory for the saved model.
        """
        out_dir = self._write(tmp_path)

        record = training_run_record(
            _outcome(str(out_dir)),
            label="run-42",
            device="cpu",
            determinism=determinism_record(UNPINNED_STACK, {}),
            prepared=_Prepared(is_peft=False, quantized=False),
            model_family="char_lstm",
        )

        assert record["fingerprint"]["gpu_model"] == ""

    def test_an_unlabelled_run_is_refused(self, tmp_path: pathlib.Path) -> None:
        """A run with no label cannot be told apart from another.

        Args:
            tmp_path: Directory for the saved model.
        """
        out_dir = self._write(tmp_path)

        with pytest.raises(ValueError, match="label"):
            _ = training_run_record(
                _outcome(str(out_dir)),
                label="",
                device="cpu",
                determinism=determinism_record(UNPINNED_STACK, {}),
                prepared=_Prepared(is_peft=False, quantized=False),
                model_family="char_lstm",
            )


class TestWritingTheSidecar:
    """The record travels with the artifact, not in a log."""

    def test_the_sidecar_is_named_from_the_model_directory(self, tmp_path: pathlib.Path) -> None:
        """Which file holds the weights differs between an adapter and a full model.

        Args:
            tmp_path: Directory for the saved model.
        """
        assert training_record_path(tmp_path).name.endswith(".runrecord.json")

    def test_writing_lands_a_decodable_record_beside_the_weights(
        self, tmp_path: pathlib.Path
    ) -> None:
        """End to end, on a real directory.

        Args:
            tmp_path: Directory for the saved model.
        """
        (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")

        path = write_training_run_record(
            _outcome(str(tmp_path)),
            label="run-42",
            device="cpu",
            determinism=determinism_record(UNPINNED_STACK, {}),
            prepared=_Prepared(is_peft=False, quantized=False),
            model_family="char_lstm",
        )

        assert path.is_file()
        assert '"label": "run-42"' in path.read_text(encoding="utf-8")
