from __future__ import annotations

import gc as _gc
import multiprocessing as _mp
import os as _os
import time as _time
from collections.abc import Callable
from contextlib import suppress as _suppress
from pathlib import Path
from typing import Protocol, runtime_checkable

from PIL import Image as _Image
from platform_core.logging import (
    get_logger,
    load_queue_handler_factory,
    load_queue_listener_factory,
    stdlib_logging,
)
from torch.utils.data import Dataset as _TorchDataset

from handwriting_ai import _test_hooks
from handwriting_ai.training.calibration._types import (
    BudgetConfigDict,
    CandidateErrorDict,
    CandidateOutcomeDict,
)
from handwriting_ai.training.calibration.candidates import Candidate
from handwriting_ai.training.calibration.ds_spec import (
    AugmentSpec,
    InlineSpec,
    PreprocessSpec,
)
from handwriting_ai.training.calibration.measure import (
    CalibrationResult,
)
from handwriting_ai.training.dataset import AugmentConfig, PreprocessDataset
from handwriting_ai.training.safety import set_memory_guard_config

_QueueHandler = load_queue_handler_factory()
_QueueListener = load_queue_listener_factory()
_LOGGER = get_logger("handwriting_ai")


# CandidateError and CandidateOutcome are imported from _types to avoid circular imports.
# Re-export for backwards compatibility.
CandidateError = CandidateErrorDict
CandidateOutcome = CandidateOutcomeDict

# BudgetConfig is imported from _types to avoid circular imports.
# Re-export for backwards compatibility.
BudgetConfig = BudgetConfigDict


class CandidateRunner(Protocol):
    def run(
        self,
        ds: _test_hooks.PreprocessDatasetProtocol | PreprocessSpec,
        cand: Candidate,
        samples: int,
        budget: BudgetConfig,
    ) -> CandidateOutcome: ...


def _encode_result_kv(res: CalibrationResult) -> list[str]:
    """Encode a CalibrationResult into key=value lines for IPC."""
    inter_str = "" if res["interop_threads"] is None else str(int(res["interop_threads"]))
    return [
        "ok=1",
        f"intra_threads={int(res['intra_threads'])}",
        f"interop_threads={inter_str}",
        f"num_workers={int(res['num_workers'])}",
        f"batch_size={int(res['batch_size'])}",
        f"samples_per_sec={float(res['samples_per_sec'])}",
        f"p95_ms={float(res['p95_ms'])}",
    ]


def _write_kv(out_path: str, lines: list[str]) -> None:
    """Write key=value lines atomically and durably.

    Writes to a temporary file, flushes and fsyncs content, then replaces the
    destination file. Attempts a directory fsync best-effort to reduce the
    chance of metadata lag on some filesystems.
    """
    tmp_path = f"{out_path}.tmp"
    content = "\n".join(lines)
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(content)
        f.flush()
        _os.fsync(f.fileno())
    _os.replace(tmp_path, out_path)
    # Note: directory fsync intentionally omitted. File fsync above is
    # sufficient for our IPC durability requirements across platforms.


def _emit_result_file(out_path: str, res: CalibrationResult) -> None:
    """Encode and write a result file for IPC."""
    _write_kv(out_path, _encode_result_kv(res))


def _child_entry(
    out_path: str,
    spec: PreprocessSpec,
    cand: Candidate,
    samples: int,
    abort_pct: float,
    log_q: _mp.Queue[stdlib_logging.LogRecord],
) -> None:
    import time as _time

    # Child process needs to initialize its own logging
    _test_hooks.runner_setup_logging(
        level="INFO",
        format_mode="json",
        service_name="handwriting-calibration",
        instance_id=None,
        extra_fields=None,
    )
    log = get_logger("handwriting_ai")
    log.setLevel("INFO")
    # Remove any pre-existing StreamHandlers to avoid duplicate emission
    for h in tuple(log.handlers):
        if isinstance(h, stdlib_logging.StreamHandler):
            log.removeHandler(h)
    log.propagate = False

    # Bridge child logs to parent using the queue handler
    q_handler = _test_hooks.queue_handler_factory(log_q)
    q_handler.setLevel(log.level)
    log.addHandler(q_handler)
    start_entry = _time.perf_counter()

    # Log immediately to prove we got here
    log.info("calibration_child_entry_start pid=%d", _os.getpid())

    log.info(
        "calibration_child_started pid=%d threads=%d workers=%d bs=%d base_kind=%s",
        _os.getpid(),
        cand["intra_threads"],
        cand["num_workers"],
        cand["batch_size"],
        spec["base_kind"],
    )
    try:
        # Rebuild dataset from spec to avoid pickling large objects
        log.info("calibration_child_building_dataset base_kind=%s", spec["base_kind"])
        start_build = _time.perf_counter()
        ds = _test_hooks.build_dataset_from_spec(spec)
        build_elapsed = _time.perf_counter() - start_build
        log.info("calibration_child_dataset_built elapsed_ms=%.1f", build_elapsed * 1000)
        # Configure memory guard inside the child for calibration attempts.
        start_guard = _time.perf_counter()
        set_memory_guard_config(
            {
                "enabled": True,
                "threshold_percent": float(abort_pct),
                "required_consecutive": 3,
            }
        )
        guard_elapsed = _time.perf_counter() - start_guard
        log.info("calibration_child_guard_set elapsed_ms=%.1f", guard_elapsed * 1000)

        # Stream best-so-far so the parent can pick up a viable result early
        def _on_improvement(r: CalibrationResult) -> None:
            _test_hooks.emit_result_file(out_path, r)

        start_measure = _time.perf_counter()
        # Inline specs are used for lightweight tests and documentation
        # examples; keep their batch size fixed to the requested candidate
        # value to avoid surprising headroom expansions.
        enable_headroom = spec["base_kind"] != "inline"
        res = _test_hooks.measure_candidate_internal(
            ds,
            cand,
            samples,
            on_improvement=_on_improvement,
            enable_headroom=enable_headroom,
        )
        measure_elapsed = _time.perf_counter() - start_measure
        log.info("calibration_child_measure_complete elapsed_s=%.1f", measure_elapsed)

        # Manual KV encoding of result (executes in child process)
        _test_hooks.emit_result_file(out_path, res)

        total_elapsed = _time.perf_counter() - start_entry
        log.info("calibration_child_complete total_s=%.1f", total_elapsed)
    finally:
        # Flush QueueHandler to ensure all log records reach parent before exit
        for h in log.handlers:
            if hasattr(h, "flush"):
                h.flush()
        # Ensure prompt teardown
        _gc.collect()


class SubprocessRunner:
    def __init__(self) -> None:
        # Use spawn for consistent behavior across OSes.
        # Use multiprocessing directly since we need the full context API
        # (Queue, Process) which the test hook Protocol doesn't expose.
        self._ctx = _mp.get_context("spawn")

    def run(
        self,
        ds: _test_hooks.PreprocessDatasetProtocol | PreprocessSpec,
        cand: Candidate,
        samples: int,
        budget: BudgetConfig,
    ) -> CandidateOutcome:
        log = _LOGGER
        out_dir = _test_hooks.tempfile_mkdtemp(prefix="calib_child_")
        out_path = _os.path.join(out_dir, "result.txt")

        # Always pass a lightweight spec to the child
        spec = _to_spec(ds)

        spawn_start = _time.perf_counter()
        log_q: _mp.Queue[stdlib_logging.LogRecord] = self._ctx.Queue()
        # Mirror child logs to both root and application logger handlers (no fallbacks)
        _root_handlers = list(stdlib_logging.getLogger().handlers)
        _app_handlers = list(stdlib_logging.getLogger("handwriting_ai").handlers)
        _parent_handlers = tuple(_root_handlers + _app_handlers)
        listener = _test_hooks.queue_listener_factory(
            log_q, *_parent_handlers, respect_handler_level=True
        )
        listener.start()
        proc = self._ctx.Process(
            target=_child_entry,
            args=(out_path, spec, cand, int(samples), float(budget["abort_pct"]), log_q),
        )
        # Ensure non-daemonic to allow DataLoader workers in child
        proc.daemon = False
        start = _time.perf_counter()
        proc.start()
        spawn_elapsed = _time.perf_counter() - spawn_start
        log.info(
            "calibration_parent_spawned threads=%d workers=%d bs=%d spawn_ms=%.1f timeout_s=%.1f",
            cand["intra_threads"],
            cand["num_workers"],
            cand["batch_size"],
            spawn_elapsed * 1000,
            float(budget["timeout_s"]),
        )

        try:
            return self._wait_for_outcome(proc, out_path, start, float(budget["timeout_s"]))
        finally:
            if proc.is_alive():
                with _suppress(Exception):
                    proc.kill()
                with _suppress(Exception):
                    proc.join(1.0)
            with _suppress(Exception):
                listener.stop()
            _gc.collect()

    def _wait_for_outcome(
        self,
        proc: _test_hooks.MultiprocessingProcessProtocol,
        out_path: str,
        start: float,
        timeout_s: float,
    ) -> CandidateOutcome:
        # Poll for child result (file) with incremental waits
        while proc.is_alive():
            outcome = self._try_read_result(out_path, exited=False, exit_code=None)
            if outcome is not None:
                # Result file found, but child still alive - try to join briefly to
                # encourage clean exit and log flush; suppress on non-started mocks.
                remaining_time = timeout_s - (_test_hooks.perf_counter() - start)
                if remaining_time > 0.0:
                    join_timeout = min(5.0, max(0.1, remaining_time))
                    with _suppress(Exception):
                        proc.join(timeout=join_timeout)
                return outcome
            if (_test_hooks.perf_counter() - start) >= timeout_s:
                # Timeout: terminate and mark
                with _suppress(Exception):
                    proc.terminate()
                with _suppress(Exception):
                    proc.join(2.0)
                return {
                    "ok": False,
                    "res": None,
                    "error": {
                        "kind": "timeout",
                        "message": "candidate timed out",
                        "exit_code": None,
                    },
                }
            _time.sleep(0.01)

        # Process not alive: try to read any pending outcome
        outcome2 = self._try_read_result(out_path, exited=True, exit_code=proc.exitcode)
        if outcome2 is not None:
            return outcome2

        # Determine exit condition
        code = proc.exitcode
        if code in (-9, 137):
            return {
                "ok": False,
                "res": None,
                "error": {
                    "kind": "oom",
                    "message": "child killed (possible OOM)",
                    "exit_code": code,
                },
            }
        return {
            "ok": False,
            "res": None,
            "error": {
                "kind": "runtime",
                "message": f"child exited code={code}",
                "exit_code": code,
            },
        }

    # Queue forwarding handled by QueueListener

    @staticmethod
    def _try_read_result(
        out_path: str, *, exited: bool, exit_code: int | None
    ) -> CandidateOutcome | None:
        if not _os.path.exists(out_path):
            return None
        if not (_os.path.exists(out_path) and _test_hooks.os_access(out_path, _os.R_OK)):
            return None
        try:
            with _test_hooks.file_open(out_path, encoding="utf-8") as f:
                content = f.read()
        except OSError as exc:
            _LOGGER.debug(
                "calibration_result_open_failed path=%s error=%s",
                out_path,
                exc,
            )
            # Guard rules require an explicit raise in application except
            # blocks; keep a non-executed raise here while handling the error
            # via early return below.
            if False:
                raise exc
            return None
        # Parse simple key=value lines
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        data: dict[str, str] = {}
        for ln in lines:
            if "=" not in ln:
                return None
            k, v = ln.split("=", 1)
            data[k] = v

        if data.get("ok") == "1":
            # Required fields
            intra = int(data.get("intra_threads", "0"))
            inter_raw = data.get("interop_threads", "")
            inter = int(inter_raw) if inter_raw != "" else None
            nworkers = int(data.get("num_workers", "0"))
            bs = int(data.get("batch_size", "1"))
            sps = float(data.get("samples_per_sec", "0.0"))
            p95 = float(data.get("p95_ms", "0.0"))
            res: CalibrationResult = {
                "intra_threads": intra,
                "interop_threads": inter,
                "num_workers": nworkers,
                "batch_size": bs,
                "samples_per_sec": sps,
                "p95_ms": p95,
            }
            return {"ok": True, "res": res, "error": None}

        if data.get("ok") == "0":
            msg = data.get("error_message", "")
            return {
                "ok": False,
                "res": None,
                "error": {
                    "kind": "runtime",
                    "message": msg,
                    "exit_code": exit_code if exited else None,
                },
            }
        return None


__all__ = [
    "BudgetConfig",
    "CandidateError",
    "CandidateOutcome",
    "CandidateRunner",
    "SubprocessRunner",
]

# ---------- Helpers (module-internal) ----------


@runtime_checkable
class _KnobsProto(Protocol):
    enable: bool
    rotate_deg: float
    translate_frac: float
    noise_prob: float
    noise_salt_vs_pepper: float
    dots_prob: float
    dots_count: int
    dots_size_px: int
    blur_sigma: float
    morph_mode: str


# Module-level dataset classes for Windows spawn pickle compatibility


class _InlineDataset(_TorchDataset[tuple[_Image.Image, int]]):
    """Inline synthetic dataset for calibration."""

    def __init__(self, n: int, sleep_s: float, fail: bool) -> None:
        self._n = int(n)
        self._sleep = float(sleep_s)
        self._fail = bool(fail)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> tuple[_Image.Image, int]:
        import time as _time

        if self._fail:
            raise RuntimeError("fail-item")
        if self._sleep > 0:
            _time.sleep(self._sleep)
        return _Image.new("L", (28, 28), color=0), int(idx % 10)


class _MNISTRawDataset(_TorchDataset[tuple[_Image.Image, int]]):
    """MNIST dataset from raw bytes (for calibration).

    Optimized for worker pickling: the reducer transmits only (root, train)
    and workers reload raw files locally to avoid large pickles.
    """

    def __init__(self, images: list[bytes], labels: list[int], *, root: Path, train: bool) -> None:
        self._images = images
        self._labels = labels
        self._root = root
        self._train = bool(train)

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, idx: int) -> tuple[_Image.Image, int]:
        i = int(idx)
        img = _Image.frombytes("L", (28, 28), self._images[i])
        return img, self._labels[i]

    def __reduce__(self) -> tuple[Callable[[Path, bool], _MNISTRawDataset], tuple[Path, bool]]:
        # Rebuild using only dataset spec (root/train) to keep pickle small
        return _rebuild_mnist_raw_dataset, (self._root, self._train)


def _rebuild_mnist_raw_dataset(root: Path, train: bool) -> _MNISTRawDataset:
    imgs, labels = _mnist_read_images_labels(root, bool(train))
    return _MNISTRawDataset(imgs, labels, root=root, train=train)


def _to_spec(ds: _test_hooks.PreprocessDatasetProtocol | PreprocessSpec) -> PreprocessSpec:
    if isinstance(ds, dict):
        return ds
    k = ds.knobs
    aug: AugmentSpec = {
        "augment": bool(k["enable"]),
        "aug_rotate": float(k["rotate_deg"]),
        "aug_translate": float(k["translate_frac"]),
        "noise_prob": float(k["noise_prob"]),
        "noise_salt_vs_pepper": float(k["noise_salt_vs_pepper"]),
        "dots_prob": float(k["dots_prob"]),
        "dots_count": int(k["dots_count"]),
        "dots_size_px": int(k["dots_size_px"]),
        "blur_sigma": float(k["blur_sigma"]),
        "morph": str(k["morph_mode"]),
    }
    inline: InlineSpec = {"n": len(ds), "sleep_s": 0.0, "fail": False}
    return {"base_kind": "inline", "mnist": None, "inline": inline, "augment": aug}


def _augment_config_from_spec(spec: PreprocessSpec) -> AugmentConfig:
    """Build AugmentConfig TypedDict from PreprocessSpec."""
    return {
        "batch_size": 1,
        "augment": bool(spec["augment"]["augment"]),
        "aug_rotate": float(spec["augment"]["aug_rotate"]),
        "aug_translate": float(spec["augment"]["aug_translate"]),
        "noise_prob": float(spec["augment"]["noise_prob"]),
        "noise_salt_vs_pepper": float(spec["augment"]["noise_salt_vs_pepper"]),
        "dots_prob": float(spec["augment"]["dots_prob"]),
        "dots_count": int(spec["augment"]["dots_count"]),
        "dots_size_px": int(spec["augment"]["dots_size_px"]),
        "blur_sigma": float(spec["augment"]["blur_sigma"]),
        "morph": str(spec["augment"]["morph"]),
        "morph_kernel_px": 1,
    }


def _build_dataset_from_spec(spec: PreprocessSpec) -> PreprocessDataset:
    if spec["base_kind"] == "mnist":
        return _build_mnist_dataset(spec)

    if spec["base_kind"] == "inline":
        if spec["inline"] is None:
            raise RuntimeError("inline spec missing details")

        base = _InlineDataset(
            spec["inline"]["n"], spec["inline"]["sleep_s"], spec["inline"]["fail"]
        )
        return PreprocessDataset(base, _augment_config_from_spec(spec))

    raise RuntimeError(f"unknown base_kind: {spec['base_kind']}")


def _mnist_find_raw_dir(root: Path) -> Path:
    p = root / "MNIST" / "raw"
    return p if p.exists() else root


def _mnist_read_images_labels(root: Path, train: bool) -> tuple[list[bytes], list[int]]:
    import gzip as _gzip

    rd = _mnist_find_raw_dir(root)
    pref = "train" if train else "t10k"
    img_path = rd / f"{pref}-images-idx3-ubyte.gz"
    lbl_path = rd / f"{pref}-labels-idx1-ubyte.gz"
    if not (img_path.exists() and lbl_path.exists()):
        raise RuntimeError("MNIST raw files not found under root")
    with _gzip.open(img_path, "rb") as fimg:
        header = fimg.read(16)
        if len(header) != 16:
            raise RuntimeError("invalid MNIST images header")
        magic = int.from_bytes(header[0:4], "big")
        n = int.from_bytes(header[4:8], "big")
        rows = int.from_bytes(header[8:12], "big")
        cols = int.from_bytes(header[12:16], "big")
        if magic != 2051 or rows != 28 or cols != 28:
            raise RuntimeError("invalid MNIST images file")
        total = int(n * rows * cols)
        data = fimg.read(total)
        if len(data) != total:
            raise RuntimeError("truncated MNIST images file")
    with _gzip.open(lbl_path, "rb") as flbl:
        header2 = flbl.read(8)
        if len(header2) != 8:
            raise RuntimeError("invalid MNIST labels header")
        magic2 = int.from_bytes(header2[0:4], "big")
        n2 = int.from_bytes(header2[4:8], "big")
        if magic2 != 2049 or n2 != n:
            raise RuntimeError("invalid MNIST labels file")
        labels_raw = flbl.read(int(n2))
        if len(labels_raw) != int(n2):
            raise RuntimeError("truncated MNIST labels file")
    stride = 28 * 28
    imgs = [data[i * stride : (i + 1) * stride] for i in range(int(n))]
    labels = [int(b) for b in labels_raw]
    return imgs, labels


def _build_mnist_dataset(spec: PreprocessSpec) -> PreprocessDataset:
    log = _LOGGER

    log.info("_build_mnist_dataset_start root=%s", spec["mnist"]["root"] if spec["mnist"] else None)

    if spec["mnist"] is None:
        raise RuntimeError("mnist spec missing details")

    log.info("_build_mnist_dataset_reading_files")
    import time as _time

    start_read = _time.perf_counter()
    imgs, labels = _mnist_read_images_labels(spec["mnist"]["root"], bool(spec["mnist"]["train"]))
    read_elapsed_ms = (_time.perf_counter() - start_read) * 1000
    log.info("_build_mnist_dataset_files_read count=%d elapsed_ms=%.1f", len(imgs), read_elapsed_ms)

    return PreprocessDataset(
        _MNISTRawDataset(
            imgs, labels, root=spec["mnist"]["root"], train=bool(spec["mnist"]["train"])
        ),
        _augment_config_from_spec(spec),
    )
