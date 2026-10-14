# Market fixture pacing and the check budget

Board task: `6448b02b-6380-4743-bc3e-c7b2caf29f52`.

The sports collector audit failed its runtime gate even though all 3,414 tests
passed with complete coverage. Fleet job
`b1b8e8cc-2ddd-445d-bbc3-79cbf6fa6227`, on serendipity at
`81e0f954e8d626682b98fc56a705d1a4a822c9d4`, took 344 seconds against the
300-second budget. Its test suite took 294.70 seconds. The remote transcript is
`C:/fleet/stage/logs/libs-covenant_ml-serendipity-1791940579.log` on that node.

Three calibration-evidence tests each acquired the same 96-contract fixture
through their own local HTTP server. They slept for the production 300 ms
request interval even though their assertions concerned evidence integrity,
offline reproduction and invalid inputs. Their durations were 52.87, 50.81
and 50.23 seconds. A forecast-exclusion test repeated the same waits and took
45.64 seconds. These four acquisitions consumed roughly 190 seconds in wall
clock delays across workers.

These repeated fixtures now inject a typed list's append method as the pause
dependency and assert exactly one 300 ms pacing request for every HTTP request.
They still call the real collector, HTTP client, local socket server, native
decoders, archive writer, provenance checks and offline evaluator. Their data,
failure assertions and subprocess tests are unchanged.

The complete forecast capture/replay test retains the real production pause.
The existing real archive acquisition test additionally checks that capture
takes at least 300 ms. Production pacing and the check budget are unchanged.
The repair removes redundant waiting from tests; it does not accelerate or
change requests sent to public market or forecast services.

## Local measurement

The full local check at `8af133613510563ab44f7c76be8f4531378d1c96`
passed in 121 seconds, compared with 206 seconds immediately before this
repair. All 3,476 tests passed, with 100% coverage of 16,573 statements and
3,228 branches. Strict typing and guards also passed. The retained real
forecast capture/replay test took 48.68 seconds. The log is
`C:/Users/Test/PROJECTS/files/sports-strategy-20261013/pacing-make-check.log`.

## Same-node remote verification

Fleet job `930aaf20-bde5-4127-97e2-91219a8b81af` checked that same commit on
serendipity and passed at October 14, 2026 01:58:15 UTC. The complete check
took 221 seconds against the unchanged 300-second budget, compared with the
earlier failed 344-second check on that node. All 3,476 tests passed with
100% coverage. Its archived log is
`serendipity:C:/fleet/stage/logs/libs-covenant_ml-serendipity-1791942847.log`.

This validates the repair across the local and remote environments. It is a
measured reduction for these runs, not a guarantee against future runtime
regressions; the existing enforced budget remains the regression gate.
