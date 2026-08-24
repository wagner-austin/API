"""Tests for the cluster composition root.

The claim this file has to defend is that a compute node runs the SAME
training code the service does. So the tests are about wiring: that the three
hooks come out pointing at local implementations, that the payload reaches
``process_train_job`` unchanged, and that the arguments naming directories are
required rather than guessed.
"""

from __future__ import annotations

import hashlib
import logging
import pathlib
import runpy
import sys
from collections.abc import Generator

import pytest
from platform_core.json_utils import JSONObject, dump_json_str

from model_trainer.cluster import _test_hooks as cluster_hooks
from model_trainer.cluster import entry as cluster_entry
from model_trainer.core import _test_hooks

_PAYLOAD: JSONObject = {"run_id": "abl-armB-s42", "user_id": 1, "request": {"seed": 42}}


def _restore_module_globals() -> Generator[None, None, None]:
    """Put back the run-job hook and sys.argv after each test.

    Both are process-global. Left swapped, the recorder installed by one test
    would stand in for the real training entry point in every later test in
    the same worker.

    Yields:
        None, for the duration of one test.
    """
    original_argv = list(sys.argv)
    yield
    cluster_hooks.reset_hooks()
    sys.argv[:] = original_argv


restore_module_globals = pytest.fixture(autouse=True)(_restore_module_globals)


class _Recorder:
    """Stands in for process_train_job, capturing what it was handed."""

    __slots__ = ("payloads",)

    def __init__(self) -> None:
        """Start with nothing recorded."""
        self.payloads: list[JSONObject] = []

    def __call__(self, payload_raw: JSONObject) -> None:
        """Record one invocation.

        Args:
            payload_raw: The payload the entry point decoded.
        """
        self.payloads.append(payload_raw)


def _args(tmp_path: pathlib.Path, payload: JSONObject | None = None) -> list[str]:
    """Write a payload and build a full argument list.

    Args:
        tmp_path: Test directory.
        payload: Payload to write, defaulting to a valid one.

    Returns:
        Arguments excluding the program name.
    """
    path = tmp_path / "payload.json"
    path.write_text(dump_json_str(_PAYLOAD if payload is None else payload), encoding="utf-8")
    return [
        cluster_entry.PAYLOAD_FLAG,
        str(path),
        cluster_entry.CORPUS_FLAG,
        str(tmp_path / "corpora"),
        cluster_entry.ARTIFACTS_FLAG,
        str(tmp_path / "artifacts"),
    ]


class TestInstalledHooks:
    """Asserted by behaviour rather than by type.

    That every hook returns the class this module happens to import proves
    nothing about a compute node. What matters is that the store answers with
    no Redis reachable, the fetcher reads the staged directory, and the
    artifact lands on local disk -- which is what these exercise.
    """

    def test_the_kv_store_answers_with_no_redis_reachable(self, tmp_path: pathlib.Path) -> None:
        cluster_entry.install_cluster_hooks(corpus_dir=tmp_path / "c", artifacts_dir=tmp_path / "a")
        store = _test_hooks.kv_store_factory("redis://nothing-is-listening-here:6379/0")
        assert store.ping() is True
        store.set("run:state", "processing")
        assert store.get("run:state") == "processing"

    def test_the_configured_redis_url_changes_nothing(self, tmp_path: pathlib.Path) -> None:
        """The factory signature is shared with the deployment that needs a
        URL. Refusing one here would make the two wirings different shapes
        for no gain -- but it must not be consulted either."""
        cluster_entry.install_cluster_hooks(corpus_dir=tmp_path / "c", artifacts_dir=tmp_path / "a")
        for url in ("redis://a:6379/0", "redis://totally-different:1/9", ""):
            store = _test_hooks.kv_store_factory(url)
            store.set("k", "v")
            assert store.get("k") == "v"

    def test_the_corpus_fetcher_resolves_from_the_staging_directory(
        self, tmp_path: pathlib.Path
    ) -> None:
        staged = tmp_path / "corpora"
        staged.mkdir()
        (staged / "abc123").write_bytes(b"corpus bytes")
        cluster_entry.install_cluster_hooks(corpus_dir=staged, artifacts_dir=tmp_path / "a")

        fetcher = _test_hooks.corpus_fetcher_factory("http://unused", "key", tmp_path / "cache")
        assert fetcher.fetch("abc123").read_bytes() == b"corpus bytes"

    def test_the_artifact_store_writes_to_the_given_directory(self, tmp_path: pathlib.Path) -> None:
        run = tmp_path / "run-x"
        run.mkdir()
        (run / "f.txt").write_text("out", encoding="utf-8")
        cluster_entry.install_cluster_hooks(
            corpus_dir=tmp_path / "c", artifacts_dir=tmp_path / "artifacts"
        )

        store = _test_hooks.artifact_store_factory("http://unused", "key")
        result = store.upload_artifact(run, artifact_name="run-x", request_id="r1")
        written = list((tmp_path / "artifacts").glob("run-x-*.tar.gz"))
        assert len(written) == 1
        assert result["sha256"] == hashlib.sha256(written[0].read_bytes()).hexdigest()

    def test_a_published_event_reaches_the_log_not_a_file(self, tmp_path: pathlib.Path) -> None:
        """Nothing may be written per-event to the shared filesystem: /pub is
        BeeGFS and its metadata servers are shared with every other user."""
        cluster_entry.install_cluster_hooks(
            corpus_dir=tmp_path / "c", artifacts_dir=tmp_path / "artifacts"
        )
        store = _test_hooks.kv_store_factory("redis://ignored")
        store.publish("trainer:events", "step 1")
        assert list(tmp_path.rglob("*")) == []


def _rendered(record: logging.LogRecord, fmt: str) -> str:
    """Render a record through a real formatter and return the result.

    Reading the extras as attributes would mean touching ``LogRecord.__dict__``,
    which is untyped. Formatting is both fully typed and a stronger check: it
    proves the fields can actually be RENDERED, which is what the service's
    JSON formatter does to them and where the original defect surfaced.

    Args:
        record: The captured record.
        fmt: A ``%``-style format naming the fields to render.

    Returns:
        The formatted line.
    """
    return logging.Formatter(fmt).format(record)


class TestPublishingGoesThroughRealLogging:
    """Driven against the actual ``logging`` module, not a fake sink.

    The fake-sink tests above passed while this path was broken. ``logging``
    refuses any ``extra`` key that would overwrite a reserved ``LogRecord``
    attribute, and ``message`` is one -- so ``extra={"message": ...}`` raises
    ``KeyError`` the first time anything is published. Nothing that stopped
    short of the real logger could see it, and the first real cluster run
    died on ``publish_started()`` before step one.
    """

    def test_publishing_does_not_collide_with_a_reserved_record_field(self) -> None:
        records: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        handler = _Capture()
        logger = logging.getLogger("model_trainer.cluster.entry")
        logger.addHandler(handler)
        previous = logger.level
        logger.setLevel(logging.INFO)
        try:
            cluster_entry._publish_to_log("trainer:events", "step 1 of 26912")
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous)

        assert len(records) == 1
        assert _rendered(records[0], "%(message)s|%(channel)s|%(event_body)s") == (
            "event|trainer:events|step 1 of 26912"
        )

    def test_the_store_publishes_through_that_same_path(self, tmp_path: pathlib.Path) -> None:
        """The wiring, not just the function: LocalKV.publish must reach the
        real logger without raising, which is what actually failed."""
        records: list[logging.LogRecord] = []

        class _Capture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        handler = _Capture()
        logger = logging.getLogger("model_trainer.cluster.entry")
        logger.addHandler(handler)
        previous = logger.level
        logger.setLevel(logging.INFO)
        try:
            cluster_entry.install_cluster_hooks(
                corpus_dir=tmp_path / "c", artifacts_dir=tmp_path / "a"
            )
            store = _test_hooks.kv_store_factory("redis://ignored")
            store.publish("trainer:events", "started")
        finally:
            logger.removeHandler(handler)
            logger.setLevel(previous)

        # install_cluster_hooks logs to this same logger, so select the
        # published events rather than assuming they are the only records.
        published = [r for r in records if r.getMessage() == "event"]
        assert [_rendered(r, "%(channel)s|%(event_body)s") for r in published] == [
            "trainer:events|started"
        ]


class TestMain:
    def test_it_hands_the_payload_to_the_training_entry_point(self, tmp_path: pathlib.Path) -> None:
        """Unchanged, so a cluster run and a queued run are the same job."""
        recorder = _Recorder()
        cluster_hooks.run_job = recorder
        assert cluster_entry.main(_args(tmp_path)) == 0
        assert recorder.payloads == [_PAYLOAD]

    def test_it_installs_the_hooks_before_running(self, tmp_path: pathlib.Path) -> None:
        """Wiring after the job started would have the trainer reach for
        Redis on its first call."""
        cluster_hooks.run_job = _Recorder()
        cluster_entry.main(_args(tmp_path))
        store = _test_hooks.kv_store_factory("redis://nothing-is-listening")
        store.set("k", "v")
        assert store.get("k") == "v"

    def test_a_payload_that_is_not_an_object_is_refused(self, tmp_path: pathlib.Path) -> None:
        args = _args(tmp_path)
        # Written AFTER _args, which writes a valid payload to the same path.
        (tmp_path / "payload.json").write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValueError, match="must hold a JSON object"):
            cluster_entry.main(args)


class TestArguments:
    def test_every_directory_flag_is_required(self, tmp_path: pathlib.Path) -> None:
        """None has a sane default: guessing where a corpus lives would find
        the wrong one as readily as none."""
        full = _args(tmp_path)
        for flag in (
            cluster_entry.PAYLOAD_FLAG,
            cluster_entry.CORPUS_FLAG,
            cluster_entry.ARTIFACTS_FLAG,
        ):
            index = full.index(flag)
            partial = full[:index] + full[index + 2 :]
            with pytest.raises(ValueError, match=flag):
                cluster_entry.main(partial)

    def test_an_unknown_argument_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="Unknown argument"):
            cluster_entry.main([*_args(tmp_path), "--turbo", "yes"])

    def test_a_flag_without_a_value_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ValueError, match="needs a value"):
            cluster_entry.main([*_args(tmp_path), cluster_entry.CORPUS_FLAG])


class TestTheModuleGuardActuallyRuns:
    """`python -m model_trainer.cluster.entry` must not silently succeed.

    Without the guard the module imports, defines its functions, and exits 0
    having trained nothing. That is not hypothetical -- it cost a real cluster
    job: 30 seconds, exit code 0, no artifacts, no output whatsoever. A batch
    script is precisely where nobody is watching a terminal, so this is the
    one invocation that must never quietly do nothing.

    Executed with runpy rather than asserted about, matching how the sibling
    worker_entry guard is covered.
    """

    def test_running_as_main_reaches_the_training_entry_point(self, tmp_path: pathlib.Path) -> None:
        recorder = _Recorder()
        cluster_hooks.run_job = recorder
        sys.argv[:] = ["prog", *_args(tmp_path)]

        module_name = "model_trainer.cluster.entry"
        saved = sys.modules.pop(module_name, None)
        try:
            with pytest.raises(SystemExit) as excinfo:
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            if saved is not None:
                sys.modules[module_name] = saved

        assert excinfo.value.code == 0
        assert recorder.payloads == [_PAYLOAD]

    def test_running_as_main_with_no_arguments_refuses(self, tmp_path: pathlib.Path) -> None:
        """The exact shape of the original failure: invoked with nothing, it
        exited 0. It must raise instead."""
        cluster_hooks.run_job = _Recorder()
        sys.argv[:] = ["prog"]

        module_name = "model_trainer.cluster.entry"
        saved = sys.modules.pop(module_name, None)
        try:
            with pytest.raises(ValueError, match=cluster_entry.PAYLOAD_FLAG):
                runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        finally:
            if saved is not None:
                sys.modules[module_name] = saved


class TestEntrypoint:
    def test_it_reads_the_process_arguments_and_exits(self, tmp_path: pathlib.Path) -> None:
        """Exercised for real rather than excluded from coverage: reading
        sys.argv and raising SystemExit only happens through this door."""
        recorder = _Recorder()
        cluster_hooks.run_job = recorder
        sys.argv[:] = ["prog", *_args(tmp_path)]
        with pytest.raises(SystemExit) as excinfo:
            cluster_entry.entrypoint()
        assert excinfo.value.code == 0
        assert recorder.payloads == [_PAYLOAD]
