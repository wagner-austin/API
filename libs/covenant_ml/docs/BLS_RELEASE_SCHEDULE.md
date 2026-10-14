# Scheduled September CPI observation

Board task: `22b04c15-0468-4e2f-9e3f-487a0e4e6886`.
Installed and verified October 14, 2026 UTC. Public GET acquisition only;
zero orders, deposits or changes to the frozen predictions.

## Frozen runtime

The standalone runtime is
`C:/Users/Test/PROJECTS/files/bls-release-feasibility-20261014/cpi-runtime-20261014`.
Its 811 copied source, script, documentation, manifest and lock files were
individually SHA-256 verified. `source-receipts.json` records each path and hash.
A fresh virtual environment was installed using `poetry sync --only main`,
including the compiled Rust dependency. The actual `covenant_ml` and
`platform_core` imports resolve inside this runtime.

The installed implementation digest equals the checked research worktree:

`28068575f121f85723db415680ca2ebc5c30112d67ef0de6eae81fc33ed6bad2`

The scheduled action invokes this runtime's Python directly. Its required
`--implementation-sha256` rejects drift before constructing a network client.
The stock, weather and demo observer runtimes and schedules were not changed.

## Real scheduled verification

Temporary task `API-CPIRelease-DeploymentCheck-20261014` executed the production
command under Windows Task Scheduler, using the Test S4U account with limited
privileges. It started at `2026-10-14T09:21:06Z`, returned exit code 0 and captured
the current official document at `2026-10-14T09:21:08.017405+00:00`.

The diagnostic deliberately requested the already-published August release.
Its validated result was August 2026, publication September 11 at 12:30 UTC,
release `USDL-26-1496`, annual CPI-U 3.4 percent. It does not label September.

Evidence lives under
`C:/Users/Test/PROJECTS/files/bls-release-feasibility-20261014`:

- `scheduled-diagnostic-capture/`: original native response envelope.
- `scheduled-diagnostic.json` and its RunRecord: real scheduled result.
- `scheduled-diagnostic-replay.json` and its RunRecord: independent runtime
  process replay; both report and sidecar are byte-identical to the original.
- Diagnostic report SHA-256:
  `34a88430d8afbfa47c699c353ce3d149166d8990cd48424135d3d898fc2b7b34`.
- Three verified paths: report, RunRecord and source envelope.
- `scheduled-diagnostic-task.xml`: exported action and scheduler configuration.

The temporary task was removed after successful execution and evidence
verification; its absence was then checked. It cannot run again accidentally.

## Registered future acquisition

Task **`API-CPIRelease-September-20261014`** is registered and reads **Ready**.
Its verified next run is **2026-10-14T12:35:00Z**, or **5:35 a.m. Pacific on
October 14**. This is five minutes after the expected 12:30 UTC publication.

The direct Python action invokes `scripts.cpi_release` with:

- `--target-month 2026-09`;
- `--published-at 2026-10-14T12:30:00Z`;
- `--mode capture` and the implementation digest above;
- new archive `september-capture-20261014/`;
- new output `september-release-20261014.json` and its RunRecord.

The archive and output reside in the evidence root above. Executable, arguments,
working directory and next-run UTC time were read back and compared after
registration. `scheduled-september-task.xml` and `deployment-receipt.json`
preserve that verified state, including the diagnostic execution result.

The task requests wake from sleep, permits battery operation and starts after
a missed trigger when the machine becomes available. It uses a 30-minute
execution limit, ignores overlapping instances and has no automatic retry or
startup trigger. The computer and public source must be available. A stale
release, unexpected publication, changed implementation or failed request
produces a nonzero exit; it cannot become a successful September result.

Registration does not prove the future run has happened. Its five-minute delay
also does not attest to the first publicly released bytes or rule out an earlier
correction. Subsequent settlement scoring must preserve this actual receipt,
check all frozen native contracts and report disagreements without relabeling
the source. No automated settlement scorer or funded-trading action is installed
by this task.

## Verification baseline

No production code changed for deployment. The deployed implementation's full
`make check` passed 3,806 tests, all guards and strict typing, with 100% of 17,853
statements and 3,648 branches in 133/300 seconds, committed in
`bda586ffbbaaf5830aeb4411e66af6e47c43fa7b` (repo api). Installation, all copied
file hashes, runtime-local imports, actual scheduled HTTP capture, independent
replay, diagnostic removal and future registration were then checked directly.
