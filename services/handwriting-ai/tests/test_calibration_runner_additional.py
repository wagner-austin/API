from __future__ import annotations

import logging
from pathlib import Path

from platform_core.logging import get_logger

from handwriting_ai.training.calibration.candidates import Candidate
from handwriting_ai.training.calibration.ds_spec import AugmentSpec, InlineSpec, PreprocessSpec
from handwriting_ai.training.calibration.runner import _build_dataset_from_spec, _child_entry


def test_build_dataset_from_spec_inline_ok() -> None:
    inline: InlineSpec = {"n": 1, "sleep_s": 0.0, "fail": False}
    aug: AugmentSpec = {
        "augment": False,
        "aug_rotate": 0.0,
        "aug_translate": 0.0,
        "noise_prob": 0.0,
        "noise_salt_vs_pepper": 0.5,
        "dots_prob": 0.0,
        "dots_count": 0,
        "dots_size_px": 1,
        "blur_sigma": 0.0,
        "morph": "none",
    }
    spec: PreprocessSpec = {"base_kind": "inline", "mnist": None, "inline": inline, "augment": aug}
    ds = _build_dataset_from_spec(spec)
    assert len(ds) == 1


def test_child_entry_removes_stream_handlers(tmp_path: Path) -> None:
    # Ensure the application logger has a StreamHandler to exercise removal
    log = get_logger("handwriting_ai")
    h = logging.StreamHandler()
    log.addHandler(h)

    inline: InlineSpec = {"n": 1, "sleep_s": 0.0, "fail": False}
    aug: AugmentSpec = {
        "augment": False,
        "aug_rotate": 0.0,
        "aug_translate": 0.0,
        "noise_prob": 0.0,
        "noise_salt_vs_pepper": 0.5,
        "dots_prob": 0.0,
        "dots_count": 0,
        "dots_size_px": 1,
        "blur_sigma": 0.0,
        "morph": "none",
    }
    spec: PreprocessSpec = {
        "base_kind": "inline",
        "mnist": None,
        "inline": inline,
        "augment": aug,
    }
    cand: Candidate = {
        "intra_threads": 1,
        "interop_threads": None,
        "num_workers": 0,
        "batch_size": 1,
    }
    out_file = str(tmp_path / "child_log_remove.txt")

    # Use a real multiprocessing queue to satisfy types but keep test lightweight
    import multiprocessing as mp
    from multiprocessing.queues import Queue as MPQueue

    q: MPQueue[logging.LogRecord] = mp.get_context("spawn").Queue()
    _child_entry(out_file, spec, cand, samples=1, abort_pct=99.0, log_q=q)
    assert (tmp_path / "child_log_remove.txt").exists()
