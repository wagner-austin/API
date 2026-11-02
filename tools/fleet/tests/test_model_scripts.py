"""The model scripts, run as written under a real bash (MCPs board task b9235198).

Each case runs a rendered script with ``systemctl``, ``systemd-run``,
``curl``, ``ss``, ``tailscale`` and ``sleep`` answered by stand-ins that keep
a unit's state in files under the case's directory, and with the real
``stat``, ``sha256sum``, ``awk`` and ``/proc/uptime``, so what is asserted is
what the script does with each answer: the file it refuses, the port it
picks, the line it prints, and the use it counts.
"""

from __future__ import annotations

import hashlib
import pathlib
import subprocess

import pytest

from fleet.contracts.model_host import CatalogModel, ModelBackend, ModelServer, NodeModels
from fleet.core.model_scripts import (
    HEALTH_SECONDS,
    forget_script,
    load_script,
    model_unit,
    scan_script,
    unload_script,
)
from tests._stand_ins import run_with_stand_ins

JOB = "e33cfb0d-66bf-4ee2-9b7c-5110cc7328ee"
OTHER_JOB = "aaaaaaaa-1111-4111-8111-aaaaaaaaaaaa"
ADDRESS = "100.111.149.33"
UNIT = model_unit("ornith-1.5-9b")
WHISPER = "whisper-large-v3-turbo"
WHISPER_UNIT = model_unit(WHISPER)
LLAMA_BINARY = "/home/corvis/build/llama.cpp-b11457/build/bin/llama-server"
WHISPER_BINARY = "/home/corvis/build/whisper.cpp-v1.9.5/build/bin/whisper-server"

#: whisper-server's log on colossus as v1.9.5 wrote it, 2026-11-02, cut to
#: the GPU lines and one transcription's.
WHISPER_LOG = (
    "whisper_init_with_params_no_state: flash attn = 1\n"
    "whisper_backend_init_gpu: device 0: Vulkan0 (type: 1)\n"
    "whisper_backend_init_gpu: found GPU device 0: Vulkan0 (type: 1, cnt: 0)\n"
    "whisper_backend_init_gpu: using Vulkan0 backend\n"
)
TRANSCRIPTION = (
    "operator(): processing 'utterance.wav' (34560 samples, 2.2 sec), 4 threads, 1 processors, "
    "lang = en, task = transcribe, timestamps = 1 ...\n"
)

#: llama-server's ``/metrics`` with 13 prompt and 8 generated tokens counted.
METRICS = (
    "# TYPE llamacpp:prompt_tokens_total counter\n"
    "llamacpp:prompt_tokens_total 13\n"
    "llamacpp:prompt_tokens_cached_total 0\n"
    "llamacpp:tokens_predicted_total 8\n"
    "llamacpp:requests_processing {busy}\n"
)

STAND_INS = {
    "tailscale": f"echo {ADDRESS}",
    "sleep": ":",
    "ss": 'cat "$(dirname "$0")/../state/ss"',
    "systemctl": (
        'state="$(dirname "$0")/../state"\n'
        'echo "$*" >> "$state/systemctl.log"\n'
        "shift\n"
        'case "$1" in\n'
        '  is-active) [ -f "$state/active" ] ;;\n'
        '  is-failed) [ -f "$state/failed" ] ;;\n'
        '  reset-failed) rm -f "$state/failed" ;;\n'
        '  stop) rm -f "$state/active" ;;\n'
        '  show) if [ "$4" = ExecMainStatus ]; then echo 1; else echo exit-code; fi ;;\n'
        "esac"
    ),
    "systemd-run": (
        'state="$(dirname "$0")/../state"\n'
        'echo "$*" >> "$state/systemd-run.log"\n'
        'for arg in "$@"; do\n'
        '  case "$arg" in\n'
        '    StandardOutput=truncate:*) log="${arg#StandardOutput=truncate:}" ;;\n'
        "  esac\n"
        "done\n"
        'cat "$state/server-log" > "$log"\n'
        'if [ -f "$state/starts" ]; then touch "$state/active"; fi'
    ),
    "curl": (
        'state="$(dirname "$0")/../state"\n'
        'for url in "$@"; do :; done\n'
        'case "$url" in\n'
        '  */health) [ -f "$state/healthy" ] && [ -f "$state/active" ] ;;\n'
        '  */metrics) [ -f "$state/metrics" ] && cat "$state/metrics" ;;\n'
        "esac"
    ),
}


class Node:
    """One case's node: its home, its model cache and its stand-ins' state."""

    def __init__(self, tmp_path: pathlib.Path) -> None:
        """Lay out an empty node.

        Args:
            tmp_path: The case's directory.
        """
        self.tmp_path = tmp_path
        self.runs = 0
        self.home = tmp_path / "home"
        self.cache = tmp_path / "models"
        self.state = tmp_path / "state"
        for directory in (self.home, self.cache, self.state):
            directory.mkdir()
        (self.state / "ss").write_text("", encoding="utf-8")
        (self.state / "server-log").write_text(
            "load_tensors: offloading 32 repeating layers to GPU\n"
            "load_tensors: offloaded 34/34 layers to GPU\n",
            encoding="utf-8",
        )
        self.models = NodeModels(
            servers={ModelServer.LLAMA: LLAMA_BINARY, ModelServer.WHISPER: WHISPER_BINARY},
            backend=ModelBackend.VULKAN,
            cache=str(self.cache),
            first_port=8091,
            last_port=8094,
            idle_seconds=1800,
        )

    def model(self, *, content: bytes = b"GGUF ornith 9b") -> CatalogModel:
        """Write the model's file into the cache and give its catalog row.

        Args:
            content: The file's bytes.

        Returns:
            The row, whose size and digest are the file's.
        """
        (self.cache / "Ornith-1.5-9B-Q4_K_M.gguf").write_bytes(content)
        return CatalogModel(
            model_id="ornith-1.5-9b",
            server=ModelServer.LLAMA,
            file="Ornith-1.5-9B-Q4_K_M.gguf",
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
            gpu_bytes=6499381248,
            context_tokens=32768,
            backends=(ModelBackend.VULKAN,),
        )

    def whisper(self) -> CatalogModel:
        """Write the whisper model's file into the cache, with its log, and give its row.

        Returns:
            The row, whose size and digest are the file's.
        """
        content = b"ggml whisper large v3 turbo"
        (self.cache / "ggml-large-v3-turbo.bin").write_bytes(content)
        (self.state / "server-log").write_text(WHISPER_LOG, encoding="utf-8")
        return CatalogModel(
            model_id=WHISPER,
            server=ModelServer.WHISPER,
            file="ggml-large-v3-turbo.bin",
            sha256=hashlib.sha256(content).hexdigest(),
            size_bytes=len(content),
            gpu_bytes=1932406784,
            context_tokens=None,
            backends=(ModelBackend.VULKAN,),
        )

    @property
    def served(self) -> pathlib.Path:
        """The Ornith model's state directory on the node."""
        return self.home / ".fleet" / "models" / "ornith-1.5-9b"

    @property
    def whisper_served(self) -> pathlib.Path:
        """The whisper model's state directory on the node."""
        return self.home / ".fleet" / "models" / WHISPER

    def serve(
        self, *, port: int, job: str, use: str, server: ModelServer = ModelServer.LLAMA
    ) -> pathlib.Path:
        """Lay out a model already served, as a load leaves it.

        Args:
            port: Its port.
            job: Its load.
            use: Its ``use`` file's line.
            server: Its server: Ornith's llama-server, or the whisper model's.

        Returns:
            Its state directory.
        """
        served = self.served if server is ModelServer.LLAMA else self.whisper_served
        served.mkdir(parents=True)
        (served / "port").write_text(f"{port}\n", encoding="utf-8")
        (served / "server").write_text(f"{server}\n", encoding="utf-8")
        (served / "job").write_text(f"{job}\n", encoding="utf-8")
        (served / "use").write_text(f"{use}\n", encoding="utf-8")
        (self.state / "active").touch()
        return served

    def flag(self, *names: str) -> None:
        """Set stand-in flags: ``starts``, ``healthy``, ``active`` or ``failed``.

        Args:
            *names: The flags.
        """
        for name in names:
            (self.state / name).touch()

    def run(self, script: str) -> subprocess.CompletedProcess[str]:
        """Run a script on this node.

        Each run gets a directory of its own, as the stand-in harness
        writes its stand-ins afresh, and a link there to the node's one
        stand-in state, so a second run sees what the first left.

        Args:
            script: The rendered script.

        Returns:
            The finished run.
        """
        self.runs += 1
        directory = self.tmp_path / f"run-{self.runs}"
        directory.mkdir()
        (directory / "state").symlink_to(self.state)
        return run_with_stand_ins(
            directory,
            [f"HOME='{self.home}'", *script.splitlines()],
            STAND_INS,
            options=[],
        )

    def calls(self, command: str) -> list[str]:
        """Every call a stand-in recorded.

        Args:
            command: ``systemctl`` or ``systemd-run``.

        Returns:
            Its argument lines, in order.
        """
        log = self.state / f"{command}.log"
        return log.read_text(encoding="utf-8").splitlines() if log.exists() else []

    def load(self, model: CatalogModel) -> subprocess.CompletedProcess[str]:
        """Run a load of a model, with the node's binary of its server.

        Args:
            model: The catalog row.

        Returns:
            The finished run.
        """
        binary = self.models["servers"][model["server"]]
        return self.run(load_script(models=self.models, model=model, job_id=JOB, binary=binary))


def _uptime() -> int:
    """This host's ``/proc/uptime`` in whole seconds, as the scripts read it."""
    return int(pathlib.Path("/proc/uptime").read_text(encoding="utf-8").split(".")[0])


@pytest.fixture(name="node")
def _node(tmp_path: pathlib.Path) -> Node:
    """An empty node for one case."""
    return Node(tmp_path)


class TestLoad:
    """What a load does with the file, the ports and the server's answers."""

    def test_an_absent_file_is_named_and_nothing_starts(self, node: Node) -> None:
        model = node.model()
        (node.cache / model["file"]).unlink()
        ran = node.load(model)
        assert (ran.returncode, ran.stdout) == (0, f"absent {node.cache}/{model['file']}\n")
        assert node.calls("systemd-run") == []

    def test_a_file_of_another_size_is_refused_before_it_is_hashed(self, node: Node) -> None:
        model = node.model()
        expected = model.copy()
        expected["size_bytes"] = 99
        ran = node.load(expected)
        assert ran.stdout == f"size {model['size_bytes']}\n"
        assert node.calls("systemd-run") == []

    def test_a_file_of_another_digest_is_refused_with_its_digest(self, node: Node) -> None:
        model = node.model(content=b"not the catalog's bytes")
        expected = model.copy()
        expected["sha256"] = hashlib.sha256(b"GGUF ornith 9b").hexdigest()
        ran = node.load(expected)
        assert ran.stdout == f"sha256 {model['sha256']}\n"
        assert node.calls("systemd-run") == []

    def test_a_fresh_load_takes_the_lowest_free_port_and_puts_every_layer_on_the_gpu(
        self, node: Node
    ) -> None:
        model = node.model()
        (node.state / "ss").write_text(
            f"LISTEN 0 512 {ADDRESS}:8091 0.0.0.0:*\nLISTEN 0 512 127.0.0.1:8080 0.0.0.0:*\n",
            encoding="utf-8",
        )
        node.flag("starts", "healthy", "failed")
        before = _uptime()
        ran = node.load(model)
        assert (ran.returncode, ran.stdout) == (0, f"loaded {ADDRESS} 8092 34/34\n"), ran.stderr
        [started] = node.calls("systemd-run")
        assert started.startswith(f"--user --quiet --unit={UNIT} --description=catalog model ")
        assert started.endswith(
            f"-- {LLAMA_BINARY} -m {node.cache}/{model['file']} --host {ADDRESS} "
            "--port 8092 -c 32768 -ngl 999 -fit off -fa on --jinja --metrics -lv 4"
        )
        assert f"StandardOutput=truncate:{node.served}/server.log" in started
        assert f"--user reset-failed {UNIT}" in node.calls("systemctl")
        assert (node.served / "port").read_text(encoding="utf-8") == "8092\n"
        assert (node.served / "server").read_text(encoding="utf-8") == "llama-server\n"
        assert (node.served / "job").read_text(encoding="utf-8") == f"{JOB}\n"
        last, seen = (node.served / "use").read_text(encoding="utf-8").split()
        assert seen == "0"
        assert before <= int(last) <= _uptime()

    def test_a_whisper_model_runs_in_whisper_server_and_names_its_gpu(self, node: Node) -> None:
        model = node.whisper()
        node.flag("starts", "healthy")
        ran = node.load(model)
        assert (ran.returncode, ran.stdout) == (0, f"loaded {ADDRESS} 8091 Vulkan0\n"), ran.stderr
        [started] = node.calls("systemd-run")
        assert started.startswith(f"--user --quiet --unit={WHISPER_UNIT} --description=catalog ")
        assert started.endswith(
            f"-- {WHISPER_BINARY} -m {node.cache}/ggml-large-v3-turbo.bin --host {ADDRESS} "
            "--port 8091"
        )
        assert (node.whisper_served / "server").read_text(encoding="utf-8") == "whisper-server\n"

    def test_a_whisper_server_that_found_no_gpu_reads_none(self, node: Node) -> None:
        model = node.whisper()
        (node.state / "server-log").write_text(
            "whisper_backend_init_gpu: no GPU found\n", encoding="utf-8"
        )
        node.flag("starts", "healthy")
        assert node.load(model).stdout == f"loaded {ADDRESS} 8091 none\n"

    def test_an_active_server_is_adopted_on_its_own_port(self, node: Node) -> None:
        model = node.model()
        node.serve(port=8093, job=OTHER_JOB, use="5 21")
        (node.served / "server.log").write_text(
            "load_tensors: offloaded 34/34 layers to GPU\n", encoding="utf-8"
        )
        node.flag("healthy")
        ran = node.load(model)
        assert ran.stdout == f"loaded {ADDRESS} 8093 34/34\n"
        assert node.calls("systemd-run") == []
        assert (node.served / "job").read_text(encoding="utf-8") == f"{JOB}\n"
        assert (node.served / "use").read_text(encoding="utf-8") == "5 21\n"

    def test_every_port_taken_is_said_and_nothing_starts(self, node: Node) -> None:
        model = node.model()
        (node.state / "ss").write_text(
            "".join(f"LISTEN 0 512 *:{port} *:*\n" for port in range(8091, 8095)),
            encoding="utf-8",
        )
        ran = node.load(model)
        assert ran.stdout == "ports 8091-8094\n"
        assert node.calls("systemd-run") == []

    def test_a_server_that_exits_while_loading_says_its_status_and_last_line(
        self, node: Node
    ) -> None:
        model = node.model()
        (node.state / "server-log").write_text(
            "load_model: loading\nggml_vulkan: device lost\n", encoding="utf-8"
        )
        ran = node.load(model)
        assert ran.stdout == "exited 1 ggml_vulkan: device lost\n"

    def test_a_server_that_never_answers_is_stopped(self, node: Node) -> None:
        model = node.model()
        node.flag("starts")
        ran = node.load(model)
        assert ran.stdout == f"unhealthy {HEALTH_SECONDS}\n"
        assert node.calls("systemctl")[-1] == f"--user stop {UNIT}"
        assert not (node.state / "active").exists()

    def test_a_log_without_its_offload_line_reads_none(self, node: Node) -> None:
        model = node.model()
        (node.state / "server-log").write_text("load_model: loading\n", encoding="utf-8")
        node.flag("starts", "healthy")
        ran = node.load(model)
        assert ran.stdout == f"loaded {ADDRESS} 8091 none\n"


class TestUnload:
    """Stopping a served model, or finding none."""

    def test_an_active_server_is_stopped_and_its_state_forgotten(self, node: Node) -> None:
        node.serve(port=8091, job=JOB, use="5 21")
        ran = node.run(unload_script("ornith-1.5-9b"))
        assert (ran.returncode, ran.stdout) == (0, f"stopped {JOB}\n")
        assert node.calls("systemctl")[-1] == f"--user stop {UNIT}"
        assert not node.served.exists()

    def test_no_active_server_is_absent_and_a_failed_record_cleared(self, node: Node) -> None:
        node.flag("failed")
        ran = node.run(unload_script("ornith-1.5-9b"))
        assert ran.stdout == "absent\n"
        assert f"--user reset-failed {UNIT}" in node.calls("systemctl")


class TestScan:
    """Which use counts, and when a served model ends."""

    def _scan(self, node: Node, *, idle_seconds: int) -> subprocess.CompletedProcess[str]:
        """Scan the node under an idle span.

        Args:
            node: The node.
            idle_seconds: Its declared span.

        Returns:
            The finished run.
        """
        models = node.models.copy()
        models["idle_seconds"] = idle_seconds
        return node.run(scan_script(models))

    def test_a_node_with_no_served_model_prints_nothing(self, node: Node) -> None:
        assert self._scan(node, idle_seconds=1800).stdout == ""

    def test_counters_that_moved_are_a_use(self, node: Node) -> None:
        node.serve(port=8091, job=JOB, use="0 0")
        (node.state / "metrics").write_text(METRICS.format(busy=0), encoding="utf-8")
        ran = self._scan(node, idle_seconds=1800)
        assert ran.stdout == f"serving ornith-1.5-9b {JOB} 8091 0\n", ran.stderr
        last, seen = (node.served / "use").read_text(encoding="utf-8").split()
        assert seen == "21"
        assert int(last) <= _uptime()

    def test_a_request_in_flight_is_a_use(self, node: Node) -> None:
        node.serve(port=8091, job=JOB, use="0 21")
        (node.state / "metrics").write_text(METRICS.format(busy=1), encoding="utf-8")
        assert self._scan(node, idle_seconds=1800).stdout == f"serving ornith-1.5-9b {JOB} 8091 0\n"

    def test_unmoved_counters_past_the_span_stop_the_server(self, node: Node) -> None:
        node.serve(port=8091, job=JOB, use="0 21")
        (node.state / "metrics").write_text(METRICS.format(busy=0), encoding="utf-8")
        ran = self._scan(node, idle_seconds=1)
        prefix = f"ended ornith-1.5-9b {JOB} idle "
        assert ran.stdout.startswith(prefix)
        assert ran.stdout.endswith(" s with no request, past its 1 s; llama-server stopped\n")
        assert int(ran.stdout.removeprefix(prefix).split(" ")[0]) >= 1
        assert node.calls("systemctl")[-1] == f"--user stop {UNIT}"
        assert (node.served / "ended").exists()
        again = self._scan(node, idle_seconds=1)
        assert again.stdout == ran.stdout
        assert node.calls("systemctl").count(f"--user stop {UNIT}") == 1

    def test_a_new_transcription_in_whisper_server_s_log_is_a_use(self, node: Node) -> None:
        served = node.serve(port=8092, job=JOB, use="0 1", server=ModelServer.WHISPER)
        (served / "server.log").write_text(WHISPER_LOG + TRANSCRIPTION * 2, encoding="utf-8")
        ran = self._scan(node, idle_seconds=1800)
        assert ran.stdout == f"serving {WHISPER} {JOB} 8092 0\n", ran.stderr
        last, seen = (served / "use").read_text(encoding="utf-8").split()
        assert seen == "2"
        assert int(last) <= _uptime()

    def test_whisper_server_with_no_new_transcription_past_the_span_is_stopped(
        self, node: Node
    ) -> None:
        served = node.serve(port=8092, job=JOB, use="0 0", server=ModelServer.WHISPER)
        (served / "server.log").write_text(WHISPER_LOG, encoding="utf-8")
        ran = self._scan(node, idle_seconds=1)
        prefix = f"ended {WHISPER} {JOB} idle "
        assert ran.stdout.startswith(prefix)
        assert ran.stdout.endswith(" s with no request, past its 1 s; whisper-server stopped\n")
        assert node.calls("systemctl")[-1] == f"--user stop {WHISPER_UNIT}"
        assert (served / "use").read_text(encoding="utf-8") == "0 0\n"

    def test_a_whisper_server_that_exited_is_ended_naming_it(self, node: Node) -> None:
        node.serve(port=8092, job=JOB, use="0 0", server=ModelServer.WHISPER)
        (node.state / "active").unlink()
        assert self._scan(node, idle_seconds=1800).stdout == (
            f"ended {WHISPER} {JOB} whisper-server exited with status 1, result exit-code, "
            "with no unload job\n"
        )

    def test_metrics_that_do_not_answer_count_no_use(self, node: Node) -> None:
        node.serve(port=8091, job=JOB, use="0 0")
        before = _uptime()
        ran = self._scan(node, idle_seconds=10**9)
        idle = int(ran.stdout.split(" ")[-1])
        assert ran.stdout == f"serving ornith-1.5-9b {JOB} 8091 {idle}\n"
        assert idle >= before
        assert (node.served / "use").read_text(encoding="utf-8") == "0 0\n"

    def test_a_server_that_exited_is_ended_with_its_status(self, node: Node) -> None:
        node.serve(port=8091, job=JOB, use="0 0")
        (node.state / "active").unlink()
        ran = self._scan(node, idle_seconds=1800)
        assert ran.stdout == (
            f"ended ornith-1.5-9b {JOB} llama-server exited with status 1, result exit-code, "
            "with no unload job\n"
        )


class TestForget:
    """Removing what the queue has recorded, and nothing a later load holds."""

    def test_only_the_state_naming_the_ended_job_is_removed(self, node: Node) -> None:
        node.serve(port=8091, job=JOB, use="0 0")
        node.flag("failed")
        assert node.run(forget_script((("ornith-1.5-9b", OTHER_JOB),))).stdout == ""
        assert node.served.exists()
        ran = node.run(forget_script((("ornith-1.5-9b", JOB), ("glimmer-30b", JOB))))
        assert ran.stdout == "forgot ornith-1.5-9b\n"
        assert not node.served.exists()
        assert f"--user reset-failed {UNIT}" in node.calls("systemctl")
