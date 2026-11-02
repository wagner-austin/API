"""Reading what a node's model scripts print into what the queue is told (MCPs board task b9235198).

Each case answers the ssh seam with what the script printed on the node
(:mod:`tests.test_model_scripts` runs the scripts themselves), so what is
asserted is each word's verdict and every line no script prints refused.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, FleetErrorCode

from fleet.contracts.dispatch import ClosingStatus
from fleet.contracts.model_host import CatalogModel, ModelBackend
from fleet.core import _test_hooks
from fleet.core.model_scripts import scan_script, unload_script
from fleet.core.model_serve import (
    LOAD_DEADLINE_SECONDS,
    EndedModel,
    ModelScan,
    ServedModel,
    forget_models,
    read_load,
    scan_models,
    serve_model,
    stop_model,
)
from tests._contract_fixtures import COLOSSUS_MODELS, node
from tests.conftest import FakeRun, failed, ok

COLOSSUS = node(host="colossus", gpu=None, models=COLOSSUS_MODELS)
JOB = "e33cfb0d-66bf-4ee2-9b7c-5110cc7328ee"
ORNITH = CatalogModel(
    model_id="ornith-1.5-9b",
    file="Ornith-1.5-9B-Q4_K_M.gguf",
    sha256="70c112196e0b7023803c9762752e46d29e612a92c83f995bc3ba1ceb07e8fab6",
    size_bytes=5780090816,
    gpu_bytes=6499381248,
    context_tokens=32768,
    backends=(ModelBackend.VULKAN,),
)
FILE = "/home/corvis/models/Ornith-1.5-9B-Q4_K_M.gguf"


def _read(output: str) -> tuple[ClosingStatus, str | None, str]:
    """Read a load script's line on colossus.

    Args:
        output: What it printed.

    Returns:
        The status, the endpoint and the detail.
    """
    read = read_load(output, host="colossus", models=COLOSSUS_MODELS, model=ORNITH)
    return read["status"], read["endpoint"], read["detail"]


class TestReadLoad:
    """Each line a load prints, and its verdict."""

    def test_every_layer_on_the_gpu_passes_with_the_url(self) -> None:
        assert _read("loaded 100.111.149.33 8091 34/34\n") == (
            ClosingStatus.PASSED,
            "http://100.111.149.33:8091",
            "llama-server serves ornith-1.5-9b (Ornith-1.5-9B-Q4_K_M.gguf, sha256 70c112196e0b) on "
            "colossus at 100.111.149.33:8091 with 34/34 layers on the GPU and a 32768-token "
            "context, and answered /health",
        )

    def test_a_file_that_is_not_the_catalog_s_is_refused_by_name(self) -> None:
        digest = "0" * 64
        assert _read(f"absent {FILE}") == (
            ClosingStatus.REFUSED,
            None,
            f"MODEL_FILE_ABSENT: ornith-1.5-9b's file {FILE} is not on colossus",
        )
        assert _read("size 12") == (
            ClosingStatus.REFUSED,
            None,
            f"MODEL_FILE_MISMATCH: {FILE} on colossus is 12 bytes, not the 5780090816 the "
            "catalog's ornith-1.5-9b is; nothing was served",
        )
        assert _read(f"sha256 {digest}") == (
            ClosingStatus.REFUSED,
            None,
            f"MODEL_FILE_MISMATCH: {FILE} on colossus hashes to sha256 {digest}, not the "
            f"catalog's {ORNITH['sha256']} for ornith-1.5-9b; nothing was served",
        )

    def test_no_free_port_is_refused(self) -> None:
        assert _read("ports 8091-8094") == (
            ClosingStatus.REFUSED,
            None,
            "MODEL_PORTS_BUSY: every model port of colossus, 8091-8094, is taken",
        )

    def test_a_server_that_did_not_serve_fails(self) -> None:
        assert _read("exited 1 ggml_vulkan: device lost") == (
            ClosingStatus.FAILED,
            None,
            "llama-server of ornith-1.5-9b on colossus exited while loading; its status and "
            "its log's last line: 1 ggml_vulkan: device lost",
        )
        assert _read("unhealthy 300") == (
            ClosingStatus.FAILED,
            None,
            "llama-server of ornith-1.5-9b on colossus did not answer /health in 300 s; stopped",
        )

    def test_a_partial_offload_or_none_fails(self) -> None:
        assert _read("loaded 100.111.149.33 8091 30/34") == (
            ClosingStatus.FAILED,
            None,
            "MODEL_PARTIAL_OFFLOAD: ornith-1.5-9b answered on colossus with 30/34 layers on the "
            "GPU, not every layer on the GPU; llama-server stopped",
        )
        assert _read("loaded 100.111.149.33 8091 none")[2] == (
            "MODEL_PARTIAL_OFFLOAD: ornith-1.5-9b answered on colossus with no offload line in "
            "its log, not every layer on the GPU; llama-server stopped"
        )

    @pytest.mark.parametrize(
        "output",
        [
            "loaded colossus 8091 34/34",
            "loaded 100.111.149.33 port 34/34",
            "loaded 100.111.149.33 8091 all",
            "loaded 100.111.149.33 8091",
            "served",
            "",
        ],
    )
    def test_a_line_no_load_prints_is_a_fault(self, output: str) -> None:
        with pytest.raises(AppError, match="the model load script on colossus printed") as fault:
            _read(output)
        assert fault.value.code is FleetErrorCode.DISPATCH_FAILED


class TestServeModel:
    """Sending the load, and stopping a server that did not serve."""

    def test_a_pass_sends_the_load_under_its_deadline_and_stops_nothing(self) -> None:
        runner = FakeRun([ok(""), ok("loaded 100.111.149.33 8091 34/34\n")])
        _test_hooks.run = runner
        served = serve_model(COLOSSUS, COLOSSUS_MODELS, ORNITH, job_id=JOB)
        assert served["endpoint"] == "http://100.111.149.33:8091"
        assert runner.calls[1][-2:] == (
            "/bin/sh",
            "/home/corvis/fleet/stage/model-load-ornith-1.5-9b.sh",
        )
        assert runner.timeouts[1] == LOAD_DEADLINE_SECONDS
        assert len(runner.calls) == 2

    def test_a_failed_load_stops_its_server_before_it_returns(self) -> None:
        runner = FakeRun([ok(""), ok("unhealthy 300\n"), ok(""), ok(f"stopped {JOB}\n")])
        _test_hooks.run = runner
        assert (
            serve_model(COLOSSUS, COLOSSUS_MODELS, ORNITH, job_id=JOB)["status"]
            is ClosingStatus.FAILED
        )
        assert runner.stdin[2] == unload_script("ornith-1.5-9b").encode()

    def test_a_refused_load_stops_nothing(self) -> None:
        runner = FakeRun([ok(""), ok(f"absent {FILE}\n")])
        _test_hooks.run = runner
        assert (
            serve_model(COLOSSUS, COLOSSUS_MODELS, ORNITH, job_id=JOB)["status"]
            is ClosingStatus.REFUSED
        )
        assert len(runner.calls) == 2

    def test_a_node_that_does_not_answer_raises_its_fault(self) -> None:
        _test_hooks.run = FakeRun([failed(255, "ssh: connect to host colossus: No route to host")])
        with pytest.raises(AppError, match="No route to host") as fault:
            serve_model(COLOSSUS, COLOSSUS_MODELS, ORNITH, job_id=JOB)
        assert fault.value.code is FleetErrorCode.NODE_UNREACHABLE


class TestStopModel:
    """An unload's two answers, and one it cannot give."""

    def test_a_stopped_server_names_its_load(self) -> None:
        _test_hooks.run = FakeRun([ok(""), ok(f"stopped {JOB}\n")])
        assert stop_model(COLOSSUS, "ornith-1.5-9b") == JOB

    def test_no_active_server_is_none(self) -> None:
        _test_hooks.run = FakeRun([ok(""), ok("absent\n")])
        assert stop_model(COLOSSUS, "ornith-1.5-9b") is None

    def test_any_other_line_is_a_fault(self) -> None:
        _test_hooks.run = FakeRun([ok(""), ok("stopped\n")])
        with pytest.raises(AppError, match="the model unload script on colossus printed 'stopped'"):
            stop_model(COLOSSUS, "ornith-1.5-9b")


class TestScanAndForget:
    """A scan read line by line, and the state forgotten after it."""

    def test_a_scan_reads_each_served_and_ended_model(self) -> None:
        runner = FakeRun(
            [
                ok(""),
                ok(
                    f"serving ornith-1.5-9b {JOB} 8091 42\n"
                    f"ended glimmer-30b {JOB} idle 1803 s with no request, past its 1800 s; "
                    "llama-server stopped\n"
                ),
            ]
        )
        _test_hooks.run = runner
        assert scan_models(COLOSSUS, COLOSSUS_MODELS) == ModelScan(
            serving=(
                ServedModel(model_id="ornith-1.5-9b", job_id=JOB, port=8091, idle_seconds=42),
            ),
            ended=(
                EndedModel(
                    model_id="glimmer-30b",
                    job_id=JOB,
                    why="idle 1803 s with no request, past its 1800 s; llama-server stopped",
                ),
            ),
        )
        assert runner.stdin[0] == scan_script(COLOSSUS_MODELS).encode()

    def test_an_empty_scan_reads_nothing(self) -> None:
        _test_hooks.run = FakeRun([ok(""), ok("")])
        assert scan_models(COLOSSUS, COLOSSUS_MODELS) == ModelScan(serving=(), ended=())

    def test_a_line_no_scan_prints_is_a_fault(self) -> None:
        _test_hooks.run = FakeRun([ok(""), ok("serving ornith-1.5-9b\n")])
        with pytest.raises(AppError, match="the model scan script on colossus printed"):
            scan_models(COLOSSUS, COLOSSUS_MODELS)

    def test_forget_names_what_it_removed(self) -> None:
        ended = EndedModel(model_id="glimmer-30b", job_id=JOB, why="idle")
        _test_hooks.run = FakeRun([ok(""), ok("forgot glimmer-30b\n")])
        assert forget_models(COLOSSUS, (ended,)) == ("glimmer-30b",)
        _test_hooks.run = FakeRun([ok(""), ok("forgot Glimmer\n")])
        with pytest.raises(AppError, match="the model forget script on colossus printed"):
            forget_models(COLOSSUS, (ended,))
