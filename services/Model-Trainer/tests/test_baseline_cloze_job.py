"""Tests for scoring an untrained model on a cloze item set.

These cover the capability that did not exist until now: every cloze result
this service produces is read as lift over an unexposed-model floor, and there
was no way to measure that floor because the scorer resolved weights from a
completed run's artifacts. The wiki's published 52.3% gpt2 floor is
consequently the one figure in its ablation with no run id behind it.

The hub load is faked so no test downloads weights. Everything else is real:
the item file is parsed by the production parser and scored by the production
scorer, and the result is read back out of Redis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import pytest
from platform_core.errors import AppError, ModelTrainerErrorCode
from platform_core.json_utils import dump_json_str, load_json_str, narrow_json_to_dict
from platform_core.trainer_keys import baseline_cloze_key, cloze_key
from platform_workers.testing import FakeRedis

from model_trainer.core import _test_hooks
from model_trainer.core._hook_protocols_ml import CorpusFetcherProto
from model_trainer.core.config.settings import Settings
from model_trainer.core.contracts.cloze import BLANK_MARKER, ClozeItem, encode_cloze_item
from model_trainer.core.contracts.queue import BaselineClozeJobPayload
from model_trainer.core.services.model.backends.hf_lm._test_hooks import Hooks as HfLmHooks
from model_trainer.core.services.model.backends.hf_lm.io import load_prepared_hf_lm_from_hub
from model_trainer.core.types import LMModelProto
from model_trainer.worker.baseline_cloze_job import process_baseline_cloze_job

from .core.services.model.backends.hf_lm.testing import FakeHFModel, FakeHFTokenizer

MODEL_ID = "gpt2-medium"
ITEMS_FILE_ID = "items-80f9732a"


class _SettingsFactory(Protocol):
    def __call__(
        self,
        *,
        artifacts_root: str | None = ...,
        runs_root: str | None = ...,
        logs_root: str | None = ...,
        data_root: str | None = ...,
        data_bank_api_url: str | None = ...,
        data_bank_api_key: str | None = ...,
    ) -> Settings: ...


def _items_jsonl(items: list[ClozeItem]) -> str:
    return "\n".join(dump_json_str(encode_cloze_item(item)) for item in items) + "\n"


def _item(item_id: str, answer: str, distractors: list[str]) -> ClozeItem:
    return ClozeItem(
        item_id=item_id,
        template=f"this is {BLANK_MARKER} here",
        answer=answer,
        distractors=distractors,
    )


def _install_hub_fakes(loaded: list[str]) -> None:
    """Bind the hub loaders to fakes and record what was asked for.

    Args:
        loaded: List the requested model ids are appended to, so a test can
            assert the job asked the hub for the model it was given.
    """

    def _load_model(model_id_or_path: str) -> LMModelProto:
        loaded.append(model_id_or_path)
        return FakeHFModel(model_id_or_path)

    def _load_tokenizer(model_id_or_path: str) -> FakeHFTokenizer:
        return FakeHFTokenizer()

    HfLmHooks.load_hf_model = _load_model
    HfLmHooks.load_hf_tokenizer = _load_tokenizer


def _install_fetcher(items_path: Path) -> None:
    """Bind the corpus fetcher to one returning a fixed item file.

    Args:
        items_path: File the fetcher returns for any file id.
    """

    class _FakeFetcher:
        def fetch(self, file_id: str) -> Path:
            return items_path

    def _factory(api_url: str, api_key: str, cache_dir: Path) -> CorpusFetcherProto:
        return _FakeFetcher()

    _test_hooks.corpus_fetcher_factory = _factory


def _make_settings(tmp_path: Path, settings_factory: _SettingsFactory) -> Settings:
    return settings_factory(
        artifacts_root=str(tmp_path / "artifacts"),
        runs_root=str(tmp_path / "runs"),
        logs_root=str(tmp_path / "logs"),
        data_root=str(tmp_path / "data"),
        data_bank_api_url="http://data-bank-api.local",
        data_bank_api_key="secret-key",
    )


def _payload(*, max_seq_len: int = 128) -> BaselineClozeJobPayload:
    return {
        "hub_model_id": MODEL_ID,
        "items_file_id": ITEMS_FILE_ID,
        "max_seq_len": max_seq_len,
        "device": "cpu",
    }


class TestLoadPreparedFromHub:
    def test_loads_the_named_model_with_nothing_applied_to_it(self) -> None:
        """A baseline is defined by having no finetuning applied.

        Asserted as strategy_name being absent rather than as some "none"
        string, because a sentinel would read as a strategy that ran.
        """
        loaded: list[str] = []
        _install_hub_fakes(loaded)

        prepared = load_prepared_hf_lm_from_hub(MODEL_ID)

        assert loaded == [MODEL_ID]
        assert prepared.hub_model_id == MODEL_ID
        assert prepared.strategy_name is None
        assert prepared.is_peft is False
        # No trained tokenizer of its own; the hub model's is the only one that
        # matches these weights.
        assert prepared.tokenizer_id is None

    def test_takes_token_ids_and_sequence_length_from_the_model_it_loaded(self) -> None:
        """These come from the hub artefacts, not from a manifest a run wrote."""
        _install_hub_fakes([])

        prepared = load_prepared_hf_lm_from_hub(MODEL_ID)

        assert prepared.eos_id == FakeHFTokenizer().eos_token_id
        assert prepared.pad_id == FakeHFTokenizer().pad_token_id
        assert prepared.max_seq_len == 512


class TestProcessBaselineClozeJob:
    def test_scores_the_item_set_and_records_it_under_a_baseline_key(
        self, tmp_path: Path, settings_factory: _SettingsFactory
    ) -> None:
        """The result must be addressable as a baseline and never as a run.

        Asserted on both keys: present under the baseline key, and absent from
        the run-cloze namespace, because a baseline that reads as a run is the
        provenance defect this whole capability exists to remove.
        """
        fake = FakeRedis()
        _test_hooks.kv_store_factory = lambda url: fake
        _test_hooks.load_settings = lambda: _make_settings(tmp_path, settings_factory)
        loaded: list[str] = []
        _install_hub_fakes(loaded)
        items_path = tmp_path / "items.jsonl"
        items_path.write_text(
            _items_jsonl([_item("a", "one", ["two"]), _item("b", "two", ["one"])]),
            encoding="utf-8",
        )
        _install_fetcher(items_path)

        process_baseline_cloze_job(_payload())

        raw = fake.get(baseline_cloze_key(MODEL_ID, ITEMS_FILE_ID))
        if not isinstance(raw, str):
            raise AssertionError(f"expected cached str, got {type(raw)}")
        record = narrow_json_to_dict(load_json_str(raw))
        assert record["status"] == "completed"
        assert record["total"] == 2
        assert record["chance"] == 0.5
        assert fake.get(cloze_key(MODEL_ID, ITEMS_FILE_ID)) is None
        assert loaded == [MODEL_ID]
        fake.assert_only_called({"set", "get"})

    def test_needs_no_run_artifacts_at_all(
        self, tmp_path: Path, settings_factory: _SettingsFactory
    ) -> None:
        """The whole point: an untrained model has no run to resolve.

        The artifact store hook is left unbound, so any attempt to materialize
        run artifacts would fail the test rather than pass silently.
        """
        fake = FakeRedis()
        _test_hooks.kv_store_factory = lambda url: fake
        _test_hooks.load_settings = lambda: _make_settings(tmp_path, settings_factory)
        _install_hub_fakes([])
        items_path = tmp_path / "items.jsonl"
        items_path.write_text(_items_jsonl([_item("a", "one", ["two"])]), encoding="utf-8")
        _install_fetcher(items_path)

        process_baseline_cloze_job(_payload())

        raw = fake.get(baseline_cloze_key(MODEL_ID, ITEMS_FILE_ID))
        if not isinstance(raw, str):
            raise AssertionError(f"expected cached str, got {type(raw)}")
        assert narrow_json_to_dict(load_json_str(raw))["status"] == "completed"
        fake.assert_only_called({"set", "get"})

    def test_records_per_item_outcomes_so_two_models_can_be_paired(
        self, tmp_path: Path, settings_factory: _SettingsFactory
    ) -> None:
        """A floor is only useful item-by-item.

        An aggregate cannot support a paired test against an arm scored on the
        same items, which is how every contrast on the ablation page is taken.
        """
        fake = FakeRedis()
        _test_hooks.kv_store_factory = lambda url: fake
        _test_hooks.load_settings = lambda: _make_settings(tmp_path, settings_factory)
        _install_hub_fakes([])
        items_path = tmp_path / "items.jsonl"
        items_path.write_text(
            _items_jsonl([_item("a", "one", ["two"]), _item("b", "two", ["one"])]),
            encoding="utf-8",
        )
        _install_fetcher(items_path)

        process_baseline_cloze_job(_payload())

        raw = fake.get(baseline_cloze_key(MODEL_ID, ITEMS_FILE_ID))
        if not isinstance(raw, str):
            raise AssertionError(f"expected cached str, got {type(raw)}")
        outcomes = narrow_json_to_dict(load_json_str(raw))["outcomes"]
        if not isinstance(outcomes, list):
            raise AssertionError(f"expected outcomes list, got {type(outcomes)}")
        assert len(outcomes) == 2
        fake.assert_only_called({"set", "get"})

    def test_an_empty_item_set_fails_the_job_and_records_the_failure(
        self, tmp_path: Path, settings_factory: _SettingsFactory
    ) -> None:
        """A failed baseline must be visible, not an absent record.

        An absent record is indistinguishable from "never scored", which is
        exactly the state that made the published floor unverifiable.
        """
        fake = FakeRedis()
        _test_hooks.kv_store_factory = lambda url: fake
        _test_hooks.load_settings = lambda: _make_settings(tmp_path, settings_factory)
        _install_hub_fakes([])
        items_path = tmp_path / "items.jsonl"
        items_path.write_text("\n\n", encoding="utf-8")
        _install_fetcher(items_path)

        with pytest.raises(AppError) as excinfo:
            process_baseline_cloze_job(_payload())

        exc: AppError[ModelTrainerErrorCode] = excinfo.value
        assert exc.code == ModelTrainerErrorCode.CLOZE_ITEMS_EMPTY
        raw = fake.get(baseline_cloze_key(MODEL_ID, ITEMS_FILE_ID))
        if not isinstance(raw, str):
            raise AssertionError(f"expected cached str, got {type(raw)}")
        assert narrow_json_to_dict(load_json_str(raw))["status"] == "failed"
        fake.assert_only_called({"set", "get"})
