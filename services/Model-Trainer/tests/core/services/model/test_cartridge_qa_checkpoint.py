"""Persisting and reloading a question-set measurement's completed arms.

REAL FILES IN ``tmp_path``, NOT A FAKED FILESYSTEM. The property this module
exists for is that a save is atomic -- ``os.replace`` either publishes a whole
checkpoint or leaves the previous one untouched -- and that property lives in
the operating system, not in this code. A test against a fake would assert
that the fake behaves as the author imagined; these assert what the platform
actually does, which is why the module carries no hook for it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from platform_core.errors import AppError, ModelTrainerErrorCode
from platform_core.json_utils import JSONTypeError, dump_json_str

from model_trainer.core.contracts.cloze import ClozeEvalResult, ClozeItemOutcome
from model_trainer.core.contracts.qa_checkpoint import (
    QA_CHECKPOINT_SCHEMA_VERSION,
    ArmRecord,
    QaCheckpoint,
    completed_arms,
    encode_qa_checkpoint,
)
from model_trainer.core.services.model.cartridge_qa_checkpoint import (
    checkpoint_exists,
    checkpoint_path,
    delete_qa_checkpoint,
    load_qa_checkpoint,
    resume_or_start,
    save_qa_checkpoint,
)

_PLAN = "gpt2-full-wiki-qa"


def _arm(name: str) -> ArmRecord:
    """Build one completed arm record.

    Args:
        name: The arm's name.

    Returns:
        The record.
    """
    outcomes: list[ClozeItemOutcome] = [
        {"item_id": "item-0", "correct": True, "scores": [1.0, 2.0]}
    ]
    return ArmRecord(
        arm=name,
        result=ClozeEvalResult(total=1, correct=1, accuracy=1.0, chance=0.25, outcomes=outcomes),
        select_seconds=1.5,
        score_seconds=20.25,
        index_seconds=3.0,
        evidence_fraction=0.5,
    )


def _checkpoint(*arms: ArmRecord) -> QaCheckpoint:
    """Build a checkpoint holding the given arms.

    Args:
        *arms: Completed arms.

    Returns:
        The checkpoint.
    """
    return QaCheckpoint(
        schema_version=QA_CHECKPOINT_SCHEMA_VERSION,
        plan_name=_PLAN,
        corpus_digest="dccd3375f54d",
        item_count=3740,
        model_id="gpt2",
        arms=list(arms),
        seeds=[],
    )


class TestSavingAndLoading:
    def test_a_saved_checkpoint_loads_back_identically(self, tmp_path: Path) -> None:
        original = _checkpoint(_arm("bm25"), _arm("dense"))

        save_qa_checkpoint(tmp_path, original)

        assert load_qa_checkpoint(tmp_path, _PLAN) == original

    def test_saving_creates_the_directory(self, tmp_path: Path) -> None:
        """The checkpoint directory is on scratch, and a resumed run may be
        the first thing to touch it after the previous one's node went
        away."""
        nested = tmp_path / "runs" / "checkpoints"

        save_qa_checkpoint(nested, _checkpoint(_arm("bm25")))

        assert checkpoint_exists(nested, _PLAN)

    def test_a_second_save_replaces_the_first(self, tmp_path: Path) -> None:
        """One ROLLING file, not an accumulating series. The resumed run
        wants the latest arm boundary, and a directory of numbered
        checkpoints is a second thing to be wrong about."""
        save_qa_checkpoint(tmp_path, _checkpoint(_arm("bm25")))

        save_qa_checkpoint(tmp_path, _checkpoint(_arm("bm25"), _arm("dense")))

        assert len(load_qa_checkpoint(tmp_path, _PLAN)["arms"]) == 2
        assert len(list(tmp_path.glob("qa-checkpoint-*"))) == 1

    def test_no_temporary_file_survives_a_save(self, tmp_path: Path) -> None:
        """A leftover .pending would be read by nothing and would sit on
        scratch forever, but more importantly its presence would mean the
        rename did not happen and the published file is the OLD one."""
        save_qa_checkpoint(tmp_path, _checkpoint(_arm("bm25")))

        assert list(tmp_path.glob("*.pending")) == []


class TestTheTemporaryIsASibling:
    def test_the_pending_file_shares_the_target_directory(self, tmp_path: Path) -> None:
        """NOT A SYSTEM TEMPORARY DIRECTORY, and this is the bug that would
        never show up in a unit test on one machine.

        ``os.replace`` is atomic only WITHIN a filesystem. On the cluster the
        checkpoint directory is on ``/pub`` scratch while ``/tmp`` is
        routinely a different device, and a cross-device rename degrades to a
        copy -- which is exactly the non-atomic publish this module exists to
        prevent. Asserting the sibling relationship pins the property that
        makes the rename safe, rather than the rename itself.
        """
        published = save_qa_checkpoint(tmp_path, _checkpoint(_arm("bm25")))

        assert published.parent == tmp_path
        assert published == checkpoint_path(tmp_path, _PLAN)


class TestRefusals:
    def test_a_file_holding_a_json_array_is_refused(self, tmp_path: Path) -> None:
        checkpoint_path(tmp_path, _PLAN).write_text("[]", encoding="utf-8")

        with pytest.raises(TypeError, match="does not hold a JSON object"):
            load_qa_checkpoint(tmp_path, _PLAN)

    def test_a_checkpoint_from_another_schema_version_is_refused(self, tmp_path: Path) -> None:
        """The refusal lives in the contract's decode, and this asserts the
        service does not route around it."""
        # WRITTEN from an encoded checkpoint rather than saved-then-edited.
        # Two earlier versions of this test were worse: one replaced text and
        # had to guess the serialiser's spacing, matched nothing, and passed
        # against an unmodified file; the next narrowed the reloaded JSON with
        # an isinstance, which the guards correctly refuse as a weak
        # assertion. Encoding directly needs neither.
        payload = encode_qa_checkpoint(_checkpoint(_arm("bm25")))
        payload["schema_version"] = QA_CHECKPOINT_SCHEMA_VERSION + 98
        checkpoint_path(tmp_path, _PLAN).write_text(dump_json_str(payload), encoding="utf-8")

        with pytest.raises(JSONTypeError, match="Re-run rather than resume"):
            load_qa_checkpoint(tmp_path, _PLAN)

    def test_loading_an_absent_checkpoint_raises_rather_than_inventing_one(
        self, tmp_path: Path
    ) -> None:
        """No empty-checkpoint fallback. A caller that has not asked
        `checkpoint_exists` is confused about whether it is resuming, and
        handing it an empty checkpoint would let a first run and a lost
        checkpoint look identical."""
        with pytest.raises(OSError):
            load_qa_checkpoint(tmp_path, _PLAN)


class TestExistenceAndDeletion:
    def test_a_directory_with_no_checkpoint_reports_none(self, tmp_path: Path) -> None:
        assert checkpoint_exists(tmp_path, _PLAN) is False

    def test_a_directory_named_like_the_file_is_not_a_checkpoint(self, tmp_path: Path) -> None:
        """``is_file`` rather than ``exists``: a directory at that path would
        pass an existence check and then fail to read."""
        checkpoint_path(tmp_path, _PLAN).mkdir(parents=True)

        assert checkpoint_exists(tmp_path, _PLAN) is False

    def test_a_completed_run_deletes_its_checkpoint(self, tmp_path: Path) -> None:
        """A leftover file is indistinguishable from an interrupted run, so
        the next submission of this plan would skip arms it should have
        re-measured."""
        save_qa_checkpoint(tmp_path, _checkpoint(_arm("bm25")))

        delete_qa_checkpoint(tmp_path, _PLAN)

        assert checkpoint_exists(tmp_path, _PLAN) is False

    def test_deleting_an_absent_checkpoint_is_not_an_error(self, tmp_path: Path) -> None:
        """A run that completes its first arm-less plan, or one deleting
        twice after a retry, must not fail at the very end for tidying."""
        delete_qa_checkpoint(tmp_path, _PLAN)

        assert checkpoint_exists(tmp_path, _PLAN) is False

    def test_two_plans_do_not_share_a_file(self, tmp_path: Path) -> None:
        """The path is keyed by plan, so a sweep measuring several plans
        into one directory cannot have one resume into another's arms."""
        assert checkpoint_path(tmp_path, "plan-a") != checkpoint_path(tmp_path, "plan-b")


class TestResumeOrStart:
    """Three states, and only the third is why this function exists."""

    def test_no_file_starts_an_empty_run(self, tmp_path: Path) -> None:
        resumed = resume_or_start(
            tmp_path,
            plan_name=_PLAN,
            corpus_digest="dccd3375f54d",
            item_count=3740,
            model_id="gpt2",
        )

        assert resumed["arms"] == []
        assert resumed["plan_name"] == _PLAN

    def test_a_matching_checkpoint_is_resumed(self, tmp_path: Path) -> None:
        save_qa_checkpoint(tmp_path, _checkpoint(_arm("bm25"), _arm("dense")))

        resumed = resume_or_start(
            tmp_path,
            plan_name=_PLAN,
            corpus_digest="dccd3375f54d",
            item_count=3740,
            model_id="gpt2",
        )

        assert completed_arms(resumed) == frozenset({"bm25", "dense"})

    def test_a_foreign_checkpoint_is_refused_not_discarded(self, tmp_path: Path) -> None:
        """THE FAILURE THIS WHOLE MODULE IS FOR.

        Resuming across a checkpoint from another corpus produces a COMPLETE
        arms table in which some arms were scored on a different question
        set. Every number is plausible, the run exits zero, and nothing
        downstream can see it -- which makes it strictly worse than a crash.
        """
        save_qa_checkpoint(tmp_path, _checkpoint(_arm("bm25")))

        with pytest.raises(AppError) as excinfo:
            resume_or_start(
                tmp_path,
                plan_name=_PLAN,
                corpus_digest="9f21ab0c771e",
                item_count=3740,
                model_id="gpt2",
            )

        assert excinfo.value.code is ModelTrainerErrorCode.CARTRIDGE_CHECKPOINT_FOREIGN
        assert "another question set" in excinfo.value.message

    def test_the_refusal_names_the_disagreeing_field_and_the_file(self, tmp_path: Path) -> None:
        """Refusing without naming the field sends the operator to delete the
        checkpoint, which spends exactly the hours it existed to save. The
        disagreement is nearly always a corpus path they can correct.
        """
        save_qa_checkpoint(tmp_path, _checkpoint(_arm("bm25")))

        with pytest.raises(AppError) as excinfo:
            resume_or_start(
                tmp_path,
                plan_name=_PLAN,
                corpus_digest="9f21ab0c771e",
                item_count=3740,
                model_id="gpt2",
            )

        assert "corpus_digest: checkpoint 'dccd3375f54d' != current '9f21ab0c771e'" in (
            excinfo.value.message
        )
        assert str(checkpoint_path(tmp_path, _PLAN)) in excinfo.value.message

    def test_deleting_the_checkpoint_is_offered_as_the_deliberate_way_out(
        self, tmp_path: Path
    ) -> None:
        """No automatic discard, but the operator must be told the escape
        exists or the refusal is an obstacle rather than a next action."""
        save_qa_checkpoint(tmp_path, _checkpoint(_arm("bm25")))

        with pytest.raises(AppError, match="delete the checkpoint"):
            resume_or_start(
                tmp_path,
                plan_name=_PLAN,
                corpus_digest="9f21ab0c771e",
                item_count=3740,
                model_id="gpt2",
            )

    def test_a_checkpoint_file_renamed_onto_another_plan_is_caught(self, tmp_path: Path) -> None:
        """The only route by which the plan_name field can disagree, and it
        is worth a test because it is otherwise unreachable.

        The path encodes the plan, so `resume_or_start` normally cannot see a
        plan mismatch -- looking up a different plan simply finds no file and
        starts fresh. The field earns its place against a file that was COPIED
        or RENAMED, which is a plausible thing to do by hand on scratch when
        two measurements are being juggled, and which would otherwise hand one
        plan's arms to another.
        """
        save_qa_checkpoint(tmp_path, _checkpoint(_arm("bm25")))
        checkpoint_path(tmp_path, _PLAN).rename(checkpoint_path(tmp_path, "other-plan"))

        with pytest.raises(AppError) as excinfo:
            resume_or_start(
                tmp_path,
                plan_name="other-plan",
                corpus_digest="dccd3375f54d",
                item_count=3740,
                model_id="gpt2",
            )

        assert "plan_name" in excinfo.value.message

    def test_a_shifted_item_count_refuses_even_with_the_same_corpus(self, tmp_path: Path) -> None:
        """The subtlest foreign checkpoint: same plan, same corpus digest,
        and a question set that realised a different number of items. The
        power gate ruled on the realised count, so a resume across a shift in
        it is a resume into a differently-powered instrument."""
        save_qa_checkpoint(tmp_path, _checkpoint(_arm("bm25")))

        with pytest.raises(AppError) as excinfo:
            resume_or_start(
                tmp_path,
                plan_name=_PLAN,
                corpus_digest="dccd3375f54d",
                item_count=3739,
                model_id="gpt2",
            )

        assert "item_count" in excinfo.value.message
