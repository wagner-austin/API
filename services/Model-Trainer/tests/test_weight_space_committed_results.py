"""The committed weight-space results, read back and, where it is cheap, re-run.

Board task 7964f9f6 committed two HPC3 RunRecords under
``results/weight-space``. These tests are what makes those files evidence
rather than prose: each number the task reports is either reproduced here
from the committed inputs or recounted from the committed per-item flags.

The MountainCar record (HPC3 job 58906732) is re-run in full. The committed
policy, the committed program and the record's own 1,024 starts on eval seed 7
go through the same ``edit_and_measure`` the job ran, on CPU. Every
observation, and the edited weights' digest that is the record's payload
digest, must come out exactly as the cluster wrote them. The expected values
are the cluster's: 0 of 1,024 before, 1,024 of 1,024 edited, 0 of 1,024
restored.

The GPT-2 record (HPC3 V100 job 58906742) cannot be re-run without the
model's weights. So its loop and cloze counts are recounted from the
committed ``outcomes.json``, one flag pair per prompt and per item, and its
McNemar p-values are recomputed from those counts. The program it measured is
checked against the weights digest and the parameter shapes of the committed
``digest.json``.
"""

from __future__ import annotations

import math
import pathlib

from platform_core.json_utils import (
    JSONObject,
    load_json_str,
    narrow_json_to_bool,
    narrow_json_to_dict,
    narrow_json_to_int,
    narrow_json_to_list,
    narrow_json_to_str,
)
from platform_core.power_distributions import exact_mcnemar_p
from platform_core.run_record import RunRecord, decode_run_record

from model_trainer.cli.weight_space_policy_edit import edit_and_measure, read_program
from model_trainer.core.contracts.weight_program import (
    WeightProgram,
    decode_weight_program,
    encode_weight_program,
)
from tests._qa_benchmark_support import values

_RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results" / "weight-space"
_CAR = _RESULTS / "mountain-car-v57"
_GPT2 = _RESULTS / "gpt2-v57"


def _object(path: pathlib.Path) -> JSONObject:
    """Read a committed JSON object.

    Args:
        path: The file.

    Returns:
        Its top-level object.
    """
    return narrow_json_to_dict(load_json_str(path.read_text("utf-8")))


def _record(path: pathlib.Path) -> RunRecord:
    """Read a committed RunRecord.

    Args:
        path: The record's file.

    Returns:
        The validated record.
    """
    return decode_run_record(load_json_str(path.read_text("utf-8")))


def _without_last_op(program: WeightProgram) -> WeightProgram:
    """The same program with its final operation dropped.

    Args:
        program: A program of two or more operations.

    Returns:
        The shortened program, still valid.
    """
    encoded = encode_weight_program(program)
    ops = narrow_json_to_list(encoded["ops"])
    encoded["ops"] = ops[:-1]
    return decode_weight_program(encoded)


class TestTheMountainCarRecordIsReproduced:
    def test_the_committed_edit_reproduces_every_observation_and_the_digest(self) -> None:
        committed = _record(_CAR / "edit-record.json")
        reproduced = edit_and_measure(
            policy_path=_CAR / "policy.json",
            program=read_program(_CAR / "velocity-sign-pump.json"),
            episodes=1024,
            eval_seed=7,
        )
        assert reproduced["observations"] == committed["observations"]
        assert reproduced["payload_digest"] == committed["payload_digest"]
        assert reproduced["label"] == "mountain-car-edit-velocity-sign-pump"

    def test_the_record_says_the_edit_took_none_to_all_and_back(self) -> None:
        observed = values(_record(_CAR / "edit-record.json")["observations"])
        assert observed["episodes"] == 1024.0
        assert observed["base_successes"] == 0.0
        assert observed["edited_successes"] == 1024.0
        assert observed["restored_successes"] == 0.0
        assert observed["elements_changed"] == 135.0
        assert observed["elements_total"] == 1251.0

    def test_the_payload_digest_binds_the_whole_program(self) -> None:
        committed = _record(_CAR / "edit-record.json")
        shortened = edit_and_measure(
            policy_path=_CAR / "policy.json",
            program=_without_last_op(read_program(_CAR / "velocity-sign-pump.json")),
            episodes=16,
            eval_seed=7,
        )
        assert shortened["payload_digest"] != committed["payload_digest"]

    def test_the_program_was_written_against_the_trained_policy(self) -> None:
        program = read_program(_CAR / "velocity-sign-pump.json")
        assert program["written_against"] == (
            "sha256:b800b62057e499627d5fae8698814ef443b29db1ba498a50a8df343dd4f1dd62"
        )
        trained = values(_record(_CAR / "train-record.json")["observations"])
        assert trained["base_successes"] == 0.0
        assert trained["policy_parameters"] == 1251.0


def _flag_pairs(outcomes: JSONObject) -> list[tuple[bool, bool]]:
    """Each prompt's (control looped, edited looped) pair.

    Args:
        outcomes: The committed ``outcomes.json``.

    Returns:
        One pair per prompt, in prompt order.
    """
    pairs: list[tuple[bool, bool]] = []
    for entry in narrow_json_to_list(outcomes["continuations"]):
        row = narrow_json_to_dict(entry)
        pairs.append(
            (narrow_json_to_bool(row["control_looped"]), narrow_json_to_bool(row["edited_looped"]))
        )
    return pairs


def _cloze_pairs(outcomes: JSONObject) -> list[tuple[bool, bool]]:
    """Each cloze item's (control correct, edited correct) pair.

    Args:
        outcomes: The committed ``outcomes.json``.

    Returns:
        One pair per item, in item order.
    """
    pairs: list[tuple[bool, bool]] = []
    for entry in narrow_json_to_list(outcomes["cloze"]):
        triple = narrow_json_to_list(entry)
        pairs.append((narrow_json_to_bool(triple[1]), narrow_json_to_bool(triple[2])))
    return pairs


def _shapes(digest: JSONObject) -> dict[str, tuple[int, ...]]:
    """The committed inventory's parameter shapes by name.

    Args:
        digest: The committed ``digest.json``.

    Returns:
        Each parameter's shape.
    """
    shapes: dict[str, tuple[int, ...]] = {}
    for entry in narrow_json_to_list(digest["parameters"]):
        pair = narrow_json_to_list(entry)
        dims = tuple(narrow_json_to_int(dim) for dim in narrow_json_to_list(pair[1]))
        shapes[narrow_json_to_str(pair[0])] = dims
    return shapes


class TestTheGpt2RecordIsRecounted:
    def test_the_loop_counts_recount_from_the_per_prompt_flags(self) -> None:
        observed = values(_record(_GPT2 / "record.json")["observations"])
        pairs = _flag_pairs(_object(_GPT2 / "outcomes.json"))
        introduced = sum(1 for control, edited in pairs if edited and not control)
        removed = sum(1 for control, edited in pairs if control and not edited)
        assert len(pairs) == 256
        assert observed["prompts"] == 256.0
        assert sum(1 for control, _ in pairs if control) == 44
        assert sum(1 for _, edited in pairs if edited) == 125
        assert observed["control_looped"] == 44.0
        assert observed["edited_looped"] == 125.0
        assert (introduced, removed) == (100, 19)
        assert observed["loops_introduced"] == 100.0
        assert observed["loops_removed"] == 19.0
        assert observed["loop_mcnemar_p"] == exact_mcnemar_p(19, 119)
        assert observed["loop_mcnemar_p"] < 1e-13

    def test_the_cloze_counts_recount_from_the_per_item_flags(self) -> None:
        observed = values(_record(_GPT2 / "record.json")["observations"])
        pairs = _cloze_pairs(_object(_GPT2 / "outcomes.json"))
        lost = sum(1 for control, edited in pairs if control and not edited)
        gained = sum(1 for control, edited in pairs if edited and not control)
        assert len(pairs) == 2627
        assert sum(1 for control, _ in pairs if control) == 1374
        assert sum(1 for _, edited in pairs if edited) == 1329
        assert observed["control_cloze_correct"] == 1374.0
        assert observed["edited_cloze_correct"] == 1329.0
        assert (lost, gained) == (189, 144)
        assert observed["cloze_mcnemar_p"] == exact_mcnemar_p(144, 333)

    def test_the_program_matches_the_weights_it_was_measured_on(self) -> None:
        observed = values(_record(_GPT2 / "record.json")["observations"])
        digest = _object(_GPT2 / "digest.json")
        program = read_program(_GPT2 / "induction-value-ablation-7heads.json")
        shapes = _shapes(digest)
        assert program["written_against"] == narrow_json_to_str(digest["weights_digest"])
        written = sum(len(op["indices"]) * shapes[op["parameter"]][0] for op in program["ops"])
        assert [op["axis"] for op in program["ops"]] == [1, 1, 1]
        assert written == 344064
        assert observed["elements_changed"] == 344064.0
        assert sum(math.prod(shape) for shape in shapes.values()) == 124439808
        assert observed["elements_total"] == 124439808.0

    def test_the_record_is_the_v100_run_of_that_program(self) -> None:
        record = _record(_GPT2 / "record.json")
        assert record["experiment"] == "weight-space-lm-edit"
        assert record["label"] == "gpt2-induction-value-ablation-7heads"
        assert record["fingerprint"]["gpu_model"] == "Tesla V100-FHHL-16GB"
        assert record["payload_digest"] == (
            "sha256:0f64d4f03790a9781b03112fc15bbdbab445b98e8b0f48e3eab1aaefed0ccb35"
        )
