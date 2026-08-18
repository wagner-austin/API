---
title: A cross-process comparison is a free function because nothing is left to inject
tags: [architecture, dependency-injection, persistence]
related: [[cpu-determinism-survives-os-and-version-change]], [[mjx-determinism-does-not-cross-backends]]
sources: [src/navprobe/crossprocess.py, src/navprobe/storage.py, tests/test_crossprocess.py]
fact_checked: 2026-08-13
confidence: high
---

# A cross-process comparison is a free function because nothing is left to inject

`ProbeService` is constructed with a simulator factory, because running a trial means building simulators. Comparing two *recordings* has no simulator: by the time that comparison runs, the processes that produced the numbers have exited, possibly on another operating system.

So the cross-process layer is module-level functions over paths, not a method on the service.[^1] There is nothing to inject, because there is nothing left running — and a service constructed with a factory it will never call would be a lie about its own dependencies.

## Persistence had no consumer, and that was a real gap

`storage.py`'s docstring stated that persistence exists because the fresh-process condition cannot be measured any other way. That was true and nothing in the package acted on it: `save_run_record` and `load_run_record` were exercised only by their own round-trip tests, and every comparison the package could actually perform was in-process.

A module whose stated purpose has no caller is an overclaim. `crossprocess.py` is what makes the claim true, and the finding in [[mjx-determinism-does-not-cross-backends]] is what it was for.

## Every repetition is persisted, not just the reference

`record_trial` writes one file per repetition plus the trial summary.[^2] Writing only the summary would have been smaller and would have made the divergence point unrecoverable: the summary carries a verdict, and localising *where* two environments part requires the per-step digests. That is the entire content of the 57-versus-19 result.

## The tamper check earns its place here

A file is the one artefact in this package that something outside the package can edit. A record whose run digest no longer follows from its steps describes no real rollout, and comparing it would report agreement on the run digest while the steps disagree — so the comparison refuses it under its own code instead.[^3]

That check existed before persistence had a consumer, and it was reachable only by constructing the contradiction by hand. With recordings on disk it is a check against a thing that can actually happen.

## Verified with an actual second interpreter

The suite spawns a real child process that records a trial, then compares against it through the files alone.[^4] Every other test in the package shares an interpreter with the code under test, which is precisely the condition this module exists to escape — so a fresh-process claim tested in-process would have asserted nothing.

[^1]: `src/navprobe/crossprocess.py` — `record_trial(service, directory, spec)`, `compare_recorded_runs(left, right)`, and `compare_recordings(left_directory, right_directory, index)` are all module-level.
[^2]: `src/navprobe/crossprocess.py`, `record_trial` — writes `run-{index}.txt` per repetition and `trial.txt` for the summary.
[^3]: `tests/test_crossprocess.py::TestCompareRecordedRuns::test_a_tampered_record_cannot_pass_as_agreement` — a step digest edited on disk raises `NP-COMPARE-002`.
[^4]: `tests/test_crossprocess.py::TestActualSeparateProcess::test_a_rollout_survives_a_process_restart` — a child interpreter records; the parent compares via `compare_recordings`.
