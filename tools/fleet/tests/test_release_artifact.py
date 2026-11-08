"""A declared release archive, read by the build and carried on the verdict (board task 0c8ace8d).

The lines run here under ``sh -eu`` with the node's ``python3``, as the builds
run them, over zip archives this case writes. Each archive's expected reading
comes from outside the code under test: its CRC-32s are what ``unzip -lv``
listed for an archive of the same two entries, and its digest is what
``sha256sum`` printed for the sorted ``<name> <crc-32>`` lines joined by a
newline, e.g. ``printf '%s\\n%s' 'META-INF/MANIFEST.MF 205adf7a'
'org/lab/Recorder.class 673d928f' | sha256sum``.
"""

from __future__ import annotations

import pathlib
import subprocess
import zipfile

import pytest
from platform_core.json_utils import JSONTypeError, JSONValue

from fleet.contracts.release_artifact import (
    ArtifactKind,
    ReleaseArtifact,
    decode_release_artifact,
    encode_release_artifact,
)
from fleet.core import dialect, names, verdict
from fleet.core.linux_isolated_build import EXEC_USER
from fleet.core.release_artifact import read_artifact, sh_artifact_lines

#: The declaration osrs-lab/recorder-release carries.
JAR = ReleaseArtifact(path="recorder/build/libs/osrs-recorder.jar", kind=ArtifactKind.ZIP)

#: The two entries, and their CRC-32s as ``unzip -lv`` listed them.
MANIFEST = ("META-INF/MANIFEST.MF", b"Manifest-Version: 1.0\n")  # 205adf7a
RECORDER = ("org/lab/Recorder.class", b"recorder\n")  # 673d928f
#: The recorder entry rebuilt with other bytes.
RECORDED = ("org/lab/Recorder.class", b"recorded\n")  # 7ba52758

#: ``sha256sum`` of 'META-INF/MANIFEST.MF 205adf7a' and 'org/lab/Recorder.class 673d928f'.
DIGEST = "5a9b0ec1516081b10243afbc797a0f3a849794adb86f4f1ba29002418f446405"
#: The same with the recorder entry's CRC-32 7ba52758.
CHANGED_DIGEST = "ed5fc10938899e248cd67a591e18efd6098a48a1e54c3fb8d2140e34398d2cae"


def _archive(root: pathlib.Path, entries: tuple[tuple[str, bytes], ...]) -> None:
    """Leave a zip of these entries where the declaration names it.

    Args:
        root: The staged tree's root.
        entries: Each entry's name and bytes.
    """
    path = root / JAR["path"]
    path.parent.mkdir(parents=True)
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in entries:
            archive.writestr(name, data)


def _run(root: pathlib.Path, artifact: ReleaseArtifact | None, run_as: str = "") -> str:
    """Run the lines over a tree and read what they appended to its transcript.

    Args:
        root: The staged tree's root, which holds the transcript too.
        artifact: The declaration, or None.
        run_as: The prefix the reading runs under.

    Returns:
        What the lines appended.
    """
    log = root / "check.log"
    log.write_text("make check\n", encoding="utf-8")
    lines = sh_artifact_lines(log=str(log), root=str(root), artifact=artifact, run_as=run_as)
    subprocess.run(["sh", "-eu", "-c", "\n".join(lines)], check=True, timeout=30)
    return log.read_text(encoding="utf-8").removeprefix("make check\n")


class TestTheLinesOnTheNode:
    def test_the_archive_is_read_as_its_entries_count_and_digest(
        self, tmp_path: pathlib.Path
    ) -> None:
        _archive(tmp_path, (RECORDER, MANIFEST))

        printed = _run(tmp_path, JAR)

        assert printed == f"FLEET_ARTIFACT: {JAR['path']} 2 {DIGEST}\n"
        assert read_artifact(printed) == f"{JAR['path']}:2:{DIGEST}"

    def test_a_changed_entry_changes_the_digest(self, tmp_path: pathlib.Path) -> None:
        _archive(tmp_path, (RECORDED, MANIFEST))

        assert read_artifact(_run(tmp_path, JAR)) == f"{JAR['path']}:2:{CHANGED_DIGEST}"

    def test_an_archive_the_recipe_did_not_leave_is_missing_never_none(
        self, tmp_path: pathlib.Path
    ) -> None:
        printed = _run(tmp_path, JAR)

        assert printed == f"FLEET_ARTIFACT: {JAR['path']} ARTIFACT_MISSING\n"
        assert read_artifact(printed) == f"ARTIFACT_MISSING:{JAR['path']}"

    def test_a_file_that_is_no_zip_is_unreadable(self, tmp_path: pathlib.Path) -> None:
        path = tmp_path / JAR["path"]
        path.parent.mkdir(parents=True)
        path.write_text("not an archive\n", encoding="utf-8")

        assert read_artifact(_run(tmp_path, JAR)) == f"ARTIFACT_UNREADABLE:{JAR['path']}"

    def test_a_reading_that_cannot_run_is_unreadable_and_the_build_goes_on(
        self, tmp_path: pathlib.Path
    ) -> None:
        _archive(tmp_path, (RECORDER, MANIFEST))

        assert read_artifact(_run(tmp_path, JAR, run_as="false ")) == (
            f"ARTIFACT_UNREADABLE:{JAR['path']}"
        )

    def test_a_project_that_declares_none_says_so(self, tmp_path: pathlib.Path) -> None:
        printed = _run(tmp_path, None)

        assert printed == "FLEET_ARTIFACT: none\n"
        assert read_artifact(printed) == "none"


class TestTheBuilds:
    def test_the_runner_s_build_reads_the_archive_its_recipe_left(
        self, tmp_path: pathlib.Path
    ) -> None:
        (tmp_path / "Makefile").write_text("check:\n\ttrue\n", encoding="utf-8")
        _archive(tmp_path, (RECORDER, MANIFEST))
        body = dialect.build_script(
            target=tmp_path.as_posix(),
            path="",
            workers=1,
            install=(),
            cache_root=(tmp_path / "cache").as_posix(),
            isolated_docker=False,
            agent="opus-research-w1-1106",
            loopback_unit=None,
            artifact=JAR,
        )

        subprocess.run(["/bin/sh"], input=body, text=True, capture_output=True, check=True)

        tail = pathlib.Path(names.log_path(tmp_path.as_posix())).read_text(encoding="utf-8")
        assert read_artifact(tail) == f"{JAR['path']}:2:{DIGEST}"

    def test_the_isolated_build_reads_it_as_execdocker_before_removing_the_copy(self) -> None:
        target = "/s/stage/osrs-lab-recorder-release-1793990000"
        lines = dialect.build_script(
            target=target,
            path="recorder-release",
            workers=4,
            install=(),
            cache_root="/s/cache",
            isolated_docker=True,
            agent="opus-research-w1-1106",
            loopback_unit=None,
            artifact=JAR,
        ).splitlines()

        (reading,) = sh_artifact_lines(
            log=names.log_path(target),
            root="$exec_root",
            artifact=JAR,
            run_as=f"sudo -n -u {EXEC_USER} ",
        )
        assert reading.startswith(f"sudo -n -u {EXEC_USER} env PATH=")
        assert f'"$exec_root/{JAR["path"]}"' in reading
        # The reading, the check results' copy (MCPs board task 25ab85cf), the
        # run's summary, then the copy's removal and the status.
        assert lines[lines.index(reading) + 3 :] == [
            'sudo -n rm -rf "$exec_root"',
            f"printf '%s\\n' \"$status\" > '{target}/{names.RESULT_NAME}'",
        ]


class TestTheReading:
    def test_a_tail_without_the_line_is_unread_and_the_last_line_wins(self) -> None:
        assert read_artifact("make check\n") == "unread"
        tail = f"FLEET_ARTIFACT: none\nFLEET_ARTIFACT: {JAR['path']} 2 {DIGEST}\n"
        assert read_artifact(tail) == f"{JAR['path']}:2:{DIGEST}"

    def test_the_verdict_line_carries_the_reading_after_the_registry(self) -> None:
        run = "osrs-lab-recorder-release-colossus-1793990000"
        judged = verdict.judge(
            job_id="0c8ace8d-0000-4000-8000-000000000000",
            project="osrs-lab/recorder-release",
            sha="54b40a8" + "0" * 33,
            node="colossus",
            exit_code=0,
            tail=f"FLEET_ARTIFACT: {JAR['path']} 2 {DIGEST}\n",
            ended_unix=1793990000,
            log_path=f"/home/corvis/fleet/stage/logs/{run}.log",
            results_path=f"/home/corvis/fleet/stage/logs/{run}.results",
            run_id=run,
            workers=4,
        )

        assert judged["artifact"] == f"{JAR['path']}:2:{DIGEST}"
        assert verdict.render_verdict(judged).endswith(
            f" registry=unread artifact={JAR['path']}:2:{DIGEST} mutation=unread"
            f" results=colossus:/home/corvis/fleet/stage/logs/{run}.results"
        )


class TestTheDeclaration:
    def test_it_decodes_encodes_and_is_absent_as_none(self) -> None:
        declared: JSONValue = {"path": JAR["path"], "kind": "zip"}

        assert decode_release_artifact(declared) == JAR
        assert encode_release_artifact(JAR) == declared
        assert decode_release_artifact(None) is None
        assert encode_release_artifact(None) is None

    @pytest.mark.parametrize(
        ("value", "named"),
        [
            ("recorder/build/libs/osrs-recorder.jar", "exactly path and kind"),
            ({"path": JAR["path"]}, "exactly path and kind"),
            ({"path": JAR["path"], "kind": "zip", "sha": "x"}, "exactly path and kind"),
            ({"path": 7, "kind": "zip"}, "path"),
            ({"path": "/srv/osrs-recorder.jar", "kind": "zip"}, "inside the staged tree"),
            ({"path": "recorder/../../etc/passwd", "kind": "zip"}, "inside the staged tree"),
            ({"path": "build/my jar.jar", "kind": "zip"}, "inside the staged tree"),
            ({"path": "build/it's.jar", "kind": "zip"}, "inside the staged tree"),
            ({"path": JAR["path"], "kind": "tar"}, "must be one of"),
            ({"path": JAR["path"], "kind": 3}, "kind"),
        ],
    )
    def test_a_misspelt_declaration_is_refused(self, value: JSONValue, named: str) -> None:
        with pytest.raises(JSONTypeError, match=named):
            decode_release_artifact(value)
