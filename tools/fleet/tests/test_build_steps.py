"""The slowest docker build steps a transcript timed (MCPs board task eb6586e4).

The lines are BuildKit's ``--progress plain`` output for grandma-api's
runtime image, with the step numbers and seconds of its cold build on
colossus on 2026-11-01 (133 s in all): the builder and runtime stages build
side by side, so their headers interleave and are printed again when a step
resumes, and the internal steps carry no stage.
"""

from __future__ import annotations

from fleet.core.build_steps import read_build_steps

COLD_BUILD = (
    "#1 [internal] load build definition from Dockerfile\n"
    "#1 DONE 0.0s\n"
    "#6 [builder  1/10] FROM docker.io/library/python:3.11.9-slim-bookworm@sha256:8fb0991\n"
    "#6 DONE 3.1s\n"
    "#8 [builder  2/10] RUN apt-get update     && apt-get install -y build-essential curl\n"
    "#9 [runtime-base 2/9] RUN apt-get update     && apt-get install -y ffmpeg\n"
    "#8 1.322 Fetched 9398 kB in 1s (8059 kB/s)\n"
    "#9 7.941 Fetched 132 MB in 5s (26.2 MB/s)\n"
    "#8 DONE 19.1s\n"
    "#10 [builder  3/10] RUN pip install --upgrade pip     && pip install poetry==1.8.3\n"
    "#9 [runtime-base 2/9] RUN apt-get update     && apt-get install -y ffmpeg\n"
    "#9 DONE 24.9s\n"
    "#10 DONE 7.9s\n"
    "#11 [runtime-base 3/9] RUN addgroup --system app\n"
    "#11 CACHED\n"
    "#13 [runtime-base 5/9] RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu\n"
    "#13 11.63    184.1/184.1 MB 36.8 MB/s eta 0:00:00\n"
    "#13 DONE 33.0s\n"
    "#23 [runtime-base 8/9] RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu\n"
    "#23 DONE 27.8s\n"
    "#25 exporting to image\n"
    "#25 DONE 40.3s\n"
)


def test_a_cold_build_names_its_three_slowest_steps_slowest_first() -> None:
    assert read_build_steps(COLD_BUILD) == (
        "runtime-base-5/9-RUN-pip:33s,runtime-base-8/9-RUN-pip:28s,runtime-base-2/9-RUN-apt-get:25s"
    )


def test_a_from_step_loses_its_tag_and_digest() -> None:
    tail = (
        "#6 [builder  1/10] FROM docker.io/library/python:3.11.9-slim-bookworm@sha256:8fb0991\n"
        "#6 DONE 3.1s\n"
    )
    assert read_build_steps(tail) == "builder-1/10-FROM-docker.io/library/python:3s"


def test_a_step_whose_header_left_the_tail_is_not_counted() -> None:
    tail = "#20 DONE 33.0s\n#21 [api 1/1] RUN true\n#21 DONE 1.0s\n"
    assert read_build_steps(tail) == "api-1/1-RUN-true:1s"


def test_a_later_build_s_header_renames_a_reused_number() -> None:
    tail = "#5 [builder 4/4] RUN make\n#5 DONE 2.0s\n#5 [runtime 2/3] COPY\n#5 DONE 9.0s\n"
    assert read_build_steps(tail) == "runtime-2/3-COPY:9s,builder-4/4-RUN-make:2s"


def test_a_tie_is_broken_by_name() -> None:
    tail = "#2 [b 1/1] RUN x\n#2 DONE 4.0s\n#3 [a 1/1] RUN y\n#3 DONE 4.0s\n"
    assert read_build_steps(tail) == "a-1/1-RUN-y:4s,b-1/1-RUN-x:4s"


def test_a_tail_with_no_timed_step_is_unread() -> None:
    assert read_build_steps("#11 [runtime-base 3/9] RUN addgroup\n#11 CACHED\n") == "unread"
    assert read_build_steps("8 passed in 31.02s\n") == "unread"
