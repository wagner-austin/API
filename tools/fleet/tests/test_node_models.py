"""A node runner's models pass and idle pass, against the queue and the node.

MCPs board task b9235198.

The queue is :class:`tests._queue_fakes.FakeQueue`, which records every
report, and the node answers through the ssh seam with what its model
scripts print (:mod:`tests.test_model_scripts` runs those), so each case
asserts what the queue is told for what the node said.
"""

from __future__ import annotations

import pytest
from platform_core.errors import AppError, FleetErrorCode
from platform_core.json_utils import JSONObject, dump_json_str

from fleet.cli.node_models import MODEL_LEASE_SECONDS, idle_pass, models_pass
from fleet.contracts.model_host import ModelServer
from fleet.contracts.node import NodeConfig
from fleet.core import _test_hooks
from fleet.core.model_scripts import unload_script
from fleet.core.queue_transport import answering
from tests._contract_fixtures import COLOSSUS_MODELS, colossus_gpu_lines, node
from tests._queue_fakes import (
    QUEUE_CREDENTIALS,
    RUNNER_IDENTITY,
    FakeQueue,
    ToolRefusal,
    Unanswered,
    queue_job,
)
from tests.conftest import FakeRun, failed, ok

COLOSSUS = node(host="colossus", gpu=None, models=COLOSSUS_MODELS)
JOB = "e33cfb0d-66bf-4ee2-9b7c-5110cc7328ee"
ORNITH: JSONObject = {
    "id": "ornith-1.5-9b",
    "server": "llama-server",
    "file": "Ornith-1.5-9B-Q4_K_M.gguf",
    "sha256": "70c112196e0b7023803c9762752e46d29e612a92c83f995bc3ba1ceb07e8fab6",
    "sizeBytes": 5780090816,
    "gpuBytes": 6499381248,
    "contextTokens": 32768,
    "backends": ["vulkan"],
}
#: The catalog's whisper model, as fleet-mcp renders it (MCPs board task 228aa097).
WHISPER: JSONObject = {
    "id": "whisper-large-v3-turbo",
    "server": "whisper-server",
    "file": "ggml-large-v3-turbo.bin",
    "sha256": "1fc70f774d38eb169993ac391eea357ef47c88757ef72ee5943879b7e8e2bc69",
    "sizeBytes": 1624555275,
    "gpuBytes": 1932406784,
    "contextTokens": None,
    "backends": ["vulkan"],
}
FILE = "/home/corvis/models/Ornith-1.5-9B-Q4_K_M.gguf"
EMPTY = dump_json_str({"claimed": None})


def _model_job(command: str, status: str, **overrides: str | JSONObject | None) -> JSONObject:
    """A model job's wire row on colossus.

    Args:
        command: ``model-load`` or ``model-unload``.
        status: Its status.
        **overrides: Fields to vary.

    Returns:
        The row.
    """
    row = queue_job(
        id=JOB,
        command=command,
        project="models/ornith-1.5-9b",
        status=status,
        requestedNode="colossus",
        sha=None,
        model=ORNITH,
        modelEndpoint=None,
    )
    row.update(overrides)
    return row


def _claimed(command: str) -> list[str]:
    """The queue's answers to a claim of one model job and its start.

    Args:
        command: Its command.

    Returns:
        The claim's and the start's answers.
    """
    return [
        dump_json_str({"claimed": _model_job(command, "claimed")}),
        dump_json_str({"job": _model_job(command, "running", node="colossus")}),
    ]


def _closed(command: str, status: str) -> str:
    """The queue's answer to a close.

    Args:
        command: The job's command.
        status: How it closed.

    Returns:
        The answer.
    """
    return dump_json_str({"job": _model_job(command, status, node="colossus")})


def _report(queue: FakeQueue, index: int) -> JSONObject:
    """One recorded report without its identity arguments.

    Args:
        queue: The fake.
        index: Which call.

    Returns:
        Its arguments.
    """
    return {
        key: value for key, value in queue.arguments[index].items() if key not in RUNNER_IDENTITY
    }


def _pass(queue: FakeQueue, runner: FakeRun, on: NodeConfig = COLOSSUS) -> tuple[str, ...]:
    """Run colossus's models pass against the fakes.

    Args:
        queue: The queue.
        runner: The node.
        on: Its declaration.

    Returns:
        The jobs it claimed.
    """
    _test_hooks.http_post = answering(queue)
    _test_hooks.run = runner
    return models_pass(QUEUE_CREDENTIALS, RUNNER_IDENTITY, alias="colossus", node=on)


def _whisper_claimed() -> list[str]:
    """The queue's answers to a claim of a whisper model's load and its start.

    Returns:
        The claim's and the start's answers.
    """
    whisper: dict[str, str | JSONObject] = {
        "project": "models/whisper-large-v3-turbo",
        "model": WHISPER,
    }
    return [
        dump_json_str({"claimed": _model_job("model-load", "claimed", **whisper)}),
        dump_json_str({"job": _model_job("model-load", "running", node="colossus", **whisper)}),
    ]


class TestModelsPass:
    """Claiming the models lane and closing each job claimed."""

    def test_a_node_that_serves_no_model_never_asks_the_lane(self) -> None:
        queue = FakeQueue([])
        _test_hooks.http_post = answering(queue)
        assert models_pass(QUEUE_CREDENTIALS, RUNNER_IDENTITY, alias="lavender", node=node()) == ()
        assert queue.tools == []

    def test_a_load_that_answers_passes_with_its_url(self) -> None:
        queue = FakeQueue([*_claimed("model-load"), _closed("model-load", "passed"), EMPTY])
        runner = FakeRun([ok(""), ok("loaded 100.111.149.33 8091 34/34\n")])
        assert _pass(queue, runner) == (JOB,)
        assert queue.tools == [
            "dispatch_claim",
            "dispatch_report",
            "dispatch_report",
            "dispatch_claim",
        ]
        assert _report(queue, 0) == {
            "lane": "models",
            "tags": [],
            "leaseSeconds": MODEL_LEASE_SECONDS,
            "node": "colossus",
        }
        assert _report(queue, 1) == {
            "action": "start",
            "jobId": JOB,
            "node": "colossus",
            "runId": "fleet-model-ornith-1.5-9b.service",
            "leaseSeconds": MODEL_LEASE_SECONDS,
        }
        loaded = _report(queue, 2)
        assert (loaded["action"], loaded["endpoint"]) == ("loaded", "http://100.111.149.33:8091")
        assert str(loaded["detail"]).startswith("llama-server serves ornith-1.5-9b")

    def test_a_file_that_is_not_the_catalog_s_is_refused_by_name(self) -> None:
        digest = "0" * 64
        queue = FakeQueue([*_claimed("model-load"), _closed("model-load", "refused"), EMPTY])
        runner = FakeRun([ok(""), ok(f"sha256 {digest}\n")])
        _pass(queue, runner)
        assert _report(queue, 2) == {
            "action": "close",
            "jobId": JOB,
            "status": "refused",
            "detail": (
                f"MODEL_FILE_MISMATCH: {FILE} on colossus hashes to sha256 {digest}, not the "
                f"catalog's {ORNITH['sha256']} for ornith-1.5-9b; nothing was served"
            ),
        }

    def test_a_server_that_did_not_serve_fails_with_exit_1(self) -> None:
        queue = FakeQueue([*_claimed("model-load"), _closed("model-load", "failed"), EMPTY])
        runner = FakeRun([ok(""), ok("unhealthy 300\n"), ok(""), ok(f"stopped {JOB}\n")])
        _pass(queue, runner)
        closed = _report(queue, 2)
        assert (closed["status"], closed["exitCode"]) == ("failed", 1)

    def test_a_backend_the_model_is_not_measured_on_is_refused_before_the_node_is_asked(
        self,
    ) -> None:
        cuda_only: JSONObject = {**ORNITH, "backends": ["cuda"]}
        queue = FakeQueue(
            [
                dump_json_str({"claimed": _model_job("model-load", "claimed", model=cuda_only)}),
                _claimed("model-load")[1],
                _closed("model-load", "refused"),
                EMPTY,
            ]
        )
        runner = FakeRun([])
        _pass(queue, runner)
        assert _report(queue, 2)["detail"] == (
            "MODEL_BACKEND: colossus serves models on vulkan, and ornith-1.5-9b is measured on "
            "cuda only"
        )
        assert runner.calls == []

    def test_a_whisper_model_is_served_by_the_node_s_whisper_server(self) -> None:
        queue = FakeQueue([*_whisper_claimed(), _closed("model-load", "passed"), EMPTY])
        runner = FakeRun([ok(""), ok("loaded 100.111.149.33 8092 Vulkan0\n")])
        assert _pass(queue, runner) == (JOB,)
        assert _report(queue, 1)["runId"] == "fleet-model-whisper-large-v3-turbo.service"
        loaded = _report(queue, 2)
        assert (loaded["action"], loaded["endpoint"]) == ("loaded", "http://100.111.149.33:8092")
        assert str(loaded["detail"]).startswith(
            "whisper-server serves whisper-large-v3-turbo (ggml-large-v3-turbo.bin"
        )
        whisper_binary = COLOSSUS_MODELS["servers"][ModelServer.WHISPER]
        assert f"{whisper_binary} -m " in (runner.stdin[0] or b"").decode()

    def test_a_server_the_node_has_not_built_is_refused_before_the_node_is_asked(self) -> None:
        llama_only = COLOSSUS_MODELS.copy()
        llama_only["servers"] = {ModelServer.LLAMA: COLOSSUS_MODELS["servers"][ModelServer.LLAMA]}
        queue = FakeQueue([*_whisper_claimed(), _closed("model-load", "refused"), EMPTY])
        runner = FakeRun([])
        _pass(queue, runner, node(host="colossus", gpu=None, models=llama_only))
        assert _report(queue, 2) == {
            "action": "close",
            "jobId": JOB,
            "status": "refused",
            "detail": (
                "MODEL_SERVER_ABSENT: whisper-large-v3-turbo runs in whisper-server, which "
                "colossus has not built; its machine file declares llama-server"
            ),
        }
        assert runner.calls == []

    def test_a_node_that_does_not_answer_refuses_the_load_with_its_fault(self) -> None:
        queue = FakeQueue([*_claimed("model-load"), _closed("model-load", "refused"), EMPTY])
        runner = FakeRun([failed(255, "ssh: connect to host colossus: No route to host")])
        _pass(queue, runner)
        closed = _report(queue, 2)
        assert closed["status"] == "refused"
        assert str(closed["detail"]).startswith("NODE_UNREACHABLE: ssh to colossus failed")

    def test_an_unload_stops_the_server_and_passes_with_the_memory_it_freed(self) -> None:
        queue = FakeQueue([*_claimed("model-unload"), _closed("model-unload", "passed"), EMPTY])
        runner = FakeRun(
            [
                ok(""),
                ok(colossus_gpu_lines(2053775360)),
                ok(""),
                ok(f"stopped {JOB}\n"),
                ok(""),
                ok(colossus_gpu_lines(8595890176)),
            ]
        )
        _pass(queue, runner)
        assert _report(queue, 2) == {
            "action": "close",
            "jobId": JOB,
            "status": "passed",
            "exitCode": 0,
            "detail": (
                f"stopped llama-server of ornith-1.5-9b on colossus, which load {JOB} started; "
                "the tick's GPU probe read 2053775360 free bytes before the stop "
                "and 8595890176 after it"
            ),
        }
        assert runner.stdin[2] == unload_script("ornith-1.5-9b").encode()

    def test_an_unload_of_nothing_served_passes_saying_so(self) -> None:
        queue = FakeQueue([*_claimed("model-unload"), _closed("model-unload", "passed"), EMPTY])
        gpus = ok(colossus_gpu_lines(8595890176))
        _pass(queue, FakeRun([ok(""), gpus, ok(""), ok("absent\n"), ok(""), gpus]))
        assert _report(queue, 2)["detail"] == (
            "ornith-1.5-9b was not being served on colossus, so there was nothing to stop; "
            "the tick's GPU probe read 8595890176 free bytes before the stop "
            "and 8595890176 after it"
        )

    def test_an_unload_whose_gpu_cannot_be_read_is_refused_by_that_fault(self) -> None:
        queue = FakeQueue([*_claimed("model-unload"), _closed("model-unload", "refused"), EMPTY])
        _pass(queue, FakeRun([ok(""), ok("gpu_count=two\n")]))
        closed = _report(queue, 2)
        assert (closed["status"], closed["detail"]) == (
            "refused",
            "DISPATCH_FAILED: colossus reported gpu_count='two', "
            "which is not a whole number of GPUs",
        )

    def test_an_unload_the_node_did_not_answer_is_refused(self) -> None:
        queue = FakeQueue([*_claimed("model-unload"), _closed("model-unload", "refused"), EMPTY])
        _pass(queue, FakeRun([failed(255, "ssh: connect to host colossus: No route to host")]))
        assert _report(queue, 2)["status"] == "refused"

    def test_a_job_the_lane_handed_out_without_a_model_is_malformed(self) -> None:
        queue = FakeQueue([dump_json_str({"claimed": queue_job(status="claimed", model=None)})])
        with pytest.raises(AppError, match="names no catalog model") as fault:
            _pass(queue, FakeRun([]))
        assert fault.value.code is FleetErrorCode.QUEUE_ANSWER_MALFORMED


def _idle(queue: FakeQueue, runner: FakeRun) -> tuple[str, ...]:
    """Run colossus's idle pass against the fakes.

    Args:
        queue: The queue.
        runner: The node.

    Returns:
        The models forgotten.
    """
    _test_hooks.http_post = answering(queue)
    _test_hooks.run = runner
    return idle_pass(QUEUE_CREDENTIALS, RUNNER_IDENTITY, alias="colossus", node=COLOSSUS)


IDLE_LINE = (
    f"ended ornith-1.5-9b {JOB} idle 603 s with no request, "
    "past its 600 s; llama-server stopped\n"
)


class TestIdlePass:
    """Stopping what nobody uses, and telling the queue."""

    def test_a_node_that_serves_no_model_asks_nothing(self) -> None:
        _test_hooks.run = FakeRun([])
        assert idle_pass(QUEUE_CREDENTIALS, RUNNER_IDENTITY, alias="lavender", node=node()) == ()

    def test_a_model_still_in_use_is_left_serving(self) -> None:
        queue = FakeQueue([])
        assert _idle(queue, FakeRun([ok(""), ok(f"serving ornith-1.5-9b {JOB} 8091 42\n")])) == ()
        assert queue.tools == []

    def test_an_idle_model_is_ended_on_its_load_then_forgotten(self) -> None:
        queue = FakeQueue([_closed("model-load", "passed")])
        runner = FakeRun([ok(""), ok(IDLE_LINE), ok(""), ok("forgot ornith-1.5-9b\n")])
        assert _idle(queue, runner) == ("ornith-1.5-9b",)
        assert _report(queue, 0) == {
            "action": "ended",
            "jobId": JOB,
            "detail": "idle 603 s with no request, past its 600 s; llama-server stopped",
        }

    def test_an_end_the_queue_refused_is_forgotten_all_the_same(self) -> None:
        queue = FakeQueue(
            [ToolRefusal(f"DISPATCH_BAD_TRANSITION: job {JOB}'s model ended already")]
        )
        runner = FakeRun([ok(""), ok(IDLE_LINE), ok(""), ok("forgot ornith-1.5-9b\n")])
        assert _idle(queue, runner) == ("ornith-1.5-9b",)

    def test_an_end_the_queue_did_not_answer_is_kept_for_the_next_scan(self) -> None:
        queue = FakeQueue([Unanswered()])
        runner = FakeRun([ok(""), ok(IDLE_LINE)])
        with pytest.raises(AppError) as fault:
            _idle(queue, runner)
        assert fault.value.code is FleetErrorCode.QUEUE_UNANSWERED
        assert len(runner.calls) == 2

    def test_a_node_that_does_not_answer_is_read_at_the_next_pass(self) -> None:
        runner = FakeRun([failed(255, "ssh: connect to host colossus: No route to host")])
        assert _idle(FakeQueue([]), runner) == ()

    def test_a_scan_that_failed_on_the_node_is_a_fault(self) -> None:
        runner = FakeRun([ok(""), failed(2, "cat: /home/corvis/.fleet/models/x/job: No such file")])
        with pytest.raises(AppError) as fault:
            _idle(FakeQueue([]), runner)
        assert fault.value.code is FleetErrorCode.DISPATCH_FAILED
