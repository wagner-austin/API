# Wiki Operation Log

Append-only. Log structural operations (new hubs, decomposition, audits,
cleanups). Routine page edits don't need a log entry — git history covers
those.

## [2026-09-01] init | wiki scaffolded from the README split
Hubs created: submission, images-and-staging, cluster-facts, operations
Pages written: run-documents, sweeps-and-artifacts, preemption-and-campaigns, submission-rules, chains, unsupported-shapes, image-build-flow, image-ledger-lessons, environment-pins, staging-identity, determinism-posture, known-answers, partitions-and-billing, facts-are-code, job-identity-on-cluster, triage-conditions, ledger-closures, budget-model
Notes: operator-directed split of the 1,045-line README (pre-split state: `README.md@4dc63f17`). The incident narrative moved VERBATIM into atomic pages — the measurements, dates, job ids and command output travelled with their claims; the split did not independently re-verify them, which is why every page's `sources:` names where the claim is checkable (a module, a test, a dated measurement) alongside the pre-split README commit. The README shrank to the command reference and points here. Announced on the agent board before the move (the README was co-edited); one commit carries both sides so the text never left HEAD.

## [2026-09-01] correction | the split cut tested reference, and its commit landed on a red gate
Pages updated: unsupported-shapes (its table now lives only in the README, where the test holds it)
Notes: the first split commit (`ef1a857d`) went in while `make check` showed six failures — `test_examples.py` holds the README to being a LIVE example (eight decodable JSON blocks, three project entries, the cannot-submit table), and the trim had cut all of it as if it were narrative. Both mistakes are corrected in `5eb68519`: the tested examples are restored to the README under "Documents by example" (they are reference, and the test is their drift-guard), the wiki page keeps the reasoning without duplicating the tested table, and the gate is green (1262 passed). The process lesson stands on its own: the gate ran, reported red, and the commit happened anyway in the same compound command — gate and commit belong in separate steps so a red result can stop the hand.

## [2026-09-01] page | node-local scratch, measured -- and a cross-project failure class named
Pages written: node-local-scratch
Notes: written from the rusted project's root-cause session, because the lesson outlives the project that paid for it. Ten campaign members across four batches "crashed at boot"; the engine log showed asset loading crawling against BeeGFS under ~60 concurrent boots until the project's own 60s liveness guard halted the process -- deterministic under concurrency, invisible in any single retry (uncontended boots are fast, so every resubmit succeeded and the class read as "transient"). The fact the page pins: Slurm provisions per-job $TMPDIR on node-local disk (measured 1.9 GB/s on hpc3-l18-04, probe job 55675199, removed with the job), and RCIC's docs state this nowhere findable. The rule: per-job disposable data references $TMPDIR unexpanded in the submitted command; the generated scripts' `set -u` makes a node without it fail loudly. The worked example is rusted's member_command, where the shared-filesystem clone helper was deleted rather than deprecated.

## [2026-09-01] add | capture-source drift, found while adding one dependency for a qlora image
Page written: capture-source-drift
Hub updated: images-and-staging (6 -> 7)
Notes: Written by opus-corpus-docmode-0901, NOT the session that owns the mi campaign.
Coordinated on the board first; this file, hubs/images-and-staging.md and index.md were
edited surgically because opus-hpc3-survey-0822 has uncommitted work in this tree.
Model-Trainer's qlora strategy never applied quantization (the config was validated and
discarded), so making it work needs bitsandbytes in the image. Preparing that surfaced
the drift: /pub/wagnera3/envs/abl-pinned no longer carries cupy-cuda12x, while v31 --
built from the spec that lists it -- does, verified by apptainer exec into the sealed
image. A re-capture off that env would have emitted a spec without cupy and taken
ordered_kernels' NVRTC path with it, with nothing in the build objecting. Capture also
emits system_packages and smoke_commands empty by design and takes required_symbols
only from --symbols, so a re-capture drops the abl spec's 29 smoke commands and 46
symbol assertions unless they are re-supplied. The surgical spec edit is what
cli/image_capture.py's own comment says every version bump has done since onboarding.

## [2026-09-02] register | this wiki joins the fleet, and every page turns out never to have parsed
Pages updated: all 21 (frontmatter converted to the code-paths contract), index.md (page
count 20 -> 21, which was simply wrong), SCHEMA.md (frontmatter + citations sections)
Registered as slug `hpc3` by opus-research-onboarding-0902, board task 429f49fe.
Notes: The wiki existed as a complete three-tier tree for weeks and was reachable only by
opening the directory. It was not in `WIKIS` (mcp-shared/src/source-registry/wikis.ts),
not mounted in the MCPs docker-compose, and the sole inbound link on the whole filesystem
was tools/hpc3/README.md -- API/wiki/index.md did not mention it, while that same index
declared infrastructure depth "a real gap". So `wiki_search_query` could not reach it and
`wiki_audit_*` ran ZERO rules against it.

The finding that justified the whole exercise: ALL 21 PAGES HAD INVALID YAML FRONTMATTER.
The form SCHEMA.md itself documented -- `related: [[a]], [[b]]` -- is a syntax error, a
flow sequence followed by a comma. Every page failed yaml.safe_load, 21 of 21. Nothing had
ever caught it because nothing had ever parsed the wiki. Registration is not only about
being searchable: an unregistered wiki is an unverified one. SCHEMA.md now documents the
quoted form and says why.

Converted every page to `source_paths:` + `source_git_blobs:` + `provenance:`. The old
shorthand (`contracts/budget.py`, `cli/triage`, `README.md@4dc63f17`) named nothing
resolvable; paths are now repo-relative (`src/hpc3/contracts/budget.py`) and every one
carries a blob pin -- 100% coverage, deliberately, because `git-blob-hash-pin` is the only
rule that detects drift and it fires only on pinned paths. Measurement strings (sshare
readings, probe job ids, /pub paths) and the three citations that genuinely point outside
tools/hpc3 (platform_core.determinism_env, model_trainer.cli.known_answer_registry,
RustedWarfareBot's member_command) moved to `provenance:`, following NavProbe's pattern.

Self-check: 21/21 parse, 0 missing paths, 0 unpinned paths, 0 hash drift.

workspaceRoot is /workspace-mounts/api/tools/hpc3, riding the existing /workspace-mounts/api
bind rather than a dedicated one -- a subdir bind carries no .git and would make every pin
false-fire "not tracked in HEAD", which is the 2026-07-23 TankpitBot lesson. The onboarding
generator DOES emit a dedicated workspace-mount line; it is wrong twice over and must not be
pasted. claudeMdPath points at the api monorepo's root CLAUDE.md, which was created the same
day and is the file a session under this tree actually loads.

## [2026-09-01] feature+page | sweeps become one sbatch call -- job arrays, measured first, parsers expanded everywhere
Pages written: job-arrays
Pages updated: unsupported-shapes (arrays moved to the left-the-list section), README table (row removed, held out by test_examples)
Notes: The submission bottleneck was ours, not the cluster's: three SSH round trips per member (~13s each) while Slurm scheduled instantly -- rusted's 96-member waves spent ~18 minutes purely submitting. A sweep now renders into ONE script that IS the member table (case-dispatch on the array task id, no --array directive; the submitter's argument owns the selection, which is what lets a campaign resubmit its sparse gap against a byte-identical script). Identity was MEASURED before it was coded (probe 55678543, throttled): pending tasks aggregate into `N_[a-b%t]` in squeue AND `sacct -X` -- the sacct half was the surprise -- and a pending task id queried directly returns nothing. `contracts/array.py` owns the expansion; the in-flight artifact check and triage's unclaimed check both expand before matching, closing the double-submission race an unexpanded set would have waved through. Per-member job_submitted audit events are gone (telemetry of acts that no longer occur); sweep_submitted carries every task id plus the billing factor. Gate green at 100.00% statements+branches, 1297 tests.

---

## [2026-09-02] update | Where an invariant belongs: PROJECT_UNIMAGED at decode made the first image unobtainable

Written from a session that onboarded a new research project (`tankpit`) end to end and hit the rule from the one direction nothing else does.

**Page written (1):** [[invariant-placement]] — hub-linked from Submission. Index 21 -> 22 pages, Submission 7 -> 8.

**The finding:** two structurally identical "this project is not ready to run" checks sit at different depths. `ENV_PATH_MISSING` is in `core/preflight.py`, on the submit path; `PROJECT_UNIMAGED` was placed in `contracts/workspace.py`, at decode. `decode_workspace` is reached from `cli/_config.py`, the loader every command shares — so a decode-level rule necessarily fires on capture, render and image-build, the three commands whose whole job is to produce the image. `ENV_PATH_MISSING` has never deadlocked anything for exactly one reason: capture does not preflight.

**Measured:** `hpc3-image-capture --config runs/hpc3-tankpit.json --project tankpit` refused with `PROJECT_UNIMAGED`, whose message instructs the reader to run the command that just refused them. To obtain an image you must already have one.

**Second, live:** `runs/hpc3.json` (cleargbm, committed) stopped decoding, and because `hpc3-research-index` reads every workspace it fails outright. One unimaged project takes a tool down for all six.

**Correction, same day:** I first recorded that the project table therefore "could not be regenerated". That was wrong, and the operator caught it — I already had the mechanism in hand, having used a `git archive` extract of HEAD to get past the same rule for the image build. Running the generator from that extract, with `index_path` and `runs_directory` rebound to the real paths, regenerated the table correctly: all six rows, one line changed, `tankpit`'s image digest picked up. The tool being broken for its ordinary invocation is true; "the table cannot be fixed" was not.

**The correction, filed as task 1ba0aac4:** move the check to the submit path. Every property survives — no project runs unimaged, no exemption field, cleargbm still refused at submission. It is a MOVE: the decode-site check is deleted outright, with no shim, wrapper, fallback, flag, declarable field or legacy path, per operator constraint. The declarable alternatives (`--allow-unimaged`, `status: provisioning`) are worse because they let a project assert its own compliance.

**The general rule this leaves behind:** an invariant about what a project may DO belongs where doing happens; only an invariant about what a document IS belongs in decode. A rule of the first kind placed in the second location does not fail loudly — it passes for every project already finished and refuses only the ones still being built, so it looks correct right up until somebody tries to start something.

**Also filed:** task e106aa83, on the general asymmetry this is the sharpest instance of — the system's refusals are mature and its on-ramps are not.

## [2026-09-05] new page | command-length limits in the remote query layer
Pages written: pages/command-length-limits.md
Pages updated: hubs/operations.md (inclusion link), index.md (4→5 operations, 25→26 total)
Notes: Records the two ceilings the package has now met -- cmd.exe's 8191 (vhsearch2-r0, 2026-09-02) and CreateProcess's 32767 (hpc3-triage at 6645 ledger rows, 2026-09-05) -- plus a third, remote-side, that both accept. The count-based EXISTENCE_CHUNK is replaced by remote.token_batches, measured by width. Fix in fa8f87f9 (repo ~/PROJECTS/API).

## [2026-09-06] update | a pending array is one row standing for many ledger rows
Pages updated: pages/triage-conditions.md, pages/ledger-closures.md, pages/job-arrays.md, pages/command-length-limits.md
Notes: Triage reported all 12 tasks of two HEALTHY queued arrays as `unaccounted`. contracts/array.py already owned the parser AND already stated the rule ("every reader that matches recorded task ids against cluster output must expand aggregates first"); three of triage's readers did not follow it. Expanding was only half -- a reader can only expand a row it was given, and `sacct -j <base>_<index>` returns nothing while that task is inside a pending aggregate, so the query is now built from array base ids. A cancelled-while-pending array (`55765275_[0-5]|CANCELLED by 2422328`) also proved terminal aggregates real, which would have left six tasks unclosable forever. Fix in 249ee403 (repo ~/PROJECTS/API); 12 findings -> 0 on the live cluster. Four pages repinned against it, each re-verified rather than bumped.

## [2026-10-05] update | every environment probe now asks the interpreter where it came from
Pages updated: pages/interpreter-availability.md, pages/environment-pins.md, pages/unsupported-shapes.md, pages/capture-source-drift.md, pages/image-build-flow.md, pages/job-arrays.md, pages/node-local-scratch.md
Notes: Board task 3b6f3848. `envs/tankpit` still runs `envs/cleargbm`'s interpreter (re-measured on login-i16), and preflight, array preflight and capture passed it because the probe listed distributions only. The identity check bootstrap already had moved to `core/interpreter.py` and now leads every probe; the pin-less early return is gone; one code, `ENV_INTERPRETER_BORROWED`, replaces `BOOTSTRAP_ENV_NOT_SELF_CONTAINED`. The rule has two shapes because the measurement forced it: all eight registered images run `/opt/env` as a venv over their own `/usr/local`, so an image is held to "base on the container's root device" rather than "base equals env_path", and the bind list could not decide it because `/dfs6b` is mounted unbound. `envs/tankpit` is deliberately not rebuilt in place, since the tankpit image's label names it. Pages cite the change's blobs; `invariant-placement` and `submission-rules` pin an older `preflight.py` blob that predates this change and were left for whoever re-verifies them.

## [2026-10-16] new page | one ledger per repository, whichever worktree submits
Pages written: pages/one-ledger-per-repository.md
Pages updated: hubs/operations.md (inclusion link), index.md (5→6 operations, 26→27 total), pages/preemption-and-campaigns.md (the new refusal, related link), pages/invariant-placement.md (`load_workspace_connection` now at `_config.py:56`), and repins after re-reading the cited sections in pages/image-build-flow.md, pages/job-arrays.md, pages/interpreter-availability.md, pages/job-identity-on-cluster.md, pages/unsupported-shapes.md
Notes: MCPs board task 747c391e. On 2026-10-05 a campaign resumed in `API-research-w2` read its own empty ledger while `API-research-w3` held the rows for twelve live v56 members, and submitted array 57783755 beside them. Every command now resolves a relative ledger against the document's directory in the repository's MAIN checkout (`core/checkout.py`, git's own `.git`/`commondir` layout read from the files), and both submitting paths refuse with `LIVE_JOB_UNRECORDED` when a live job carries the submission's name and this ledger has no row for it, which is what still separates two machines.

## [2026-10-16] new page | an image older than its command is refused at preflight
Pages written: pages/image-freshness.md
Pages updated: hubs/images-and-staging.md (inclusion link), index.md (9→10 images and staging, 27→28 total), pages/submission-rules.md (the new refusal, its footnote, and the preflight.py pin left stale on 2026-10-05, now re-read), and repins after re-reading the cited sections in pages/invariant-placement.md, pages/node-local-scratch.md (preflight.py), pages/job-arrays.md, pages/one-ledger-per-repository.md (array_submit.py), pages/interpreter-availability.md, pages/image-build-flow.md, pages/job-identity-on-cluster.md, pages/unsupported-shapes.md (README.md)
Notes: MCPs board task 2c8161a3. Every registered image already carries its build commit at `/opt/env/GIT_COMMIT` (all nine read on 2026-10-16; the `mi` image v23 says `fab1d991`, which has none of the cartridge commands). Preflight and array preflight now ask the environment, inside its image, whether it can start every `python -m` module the command names, resolved from the directory the command enters, and refuse `ENTRYPOINT_NOT_INSTALLED` naming that commit; a module the image has is admitted however its source changed since, so an old command under its old image stays legal. Measured against v23 itself: `cartridge_composition_sweep` refused, `known_answer_probe` admitted.

## [2026-10-16] update | an image's code is compared with HEAD, and an unstamped image is refused
Pages updated: pages/image-freshness.md (rewritten: the stamp refusal, where a module resolved, the drift comparison and its ledger record, three real preflights), pages/submission-rules.md (the two new refusals and the drift warning, a new footnote and pin), hubs/images-and-staging.md (the inclusion line), pages/determinism-posture.md (ledger.py line citations moved to :316 and :276), and repins after re-reading the cited sections in pages/chains.md (cli/chain.py, core/chain.py), pages/invariant-placement.md, pages/node-local-scratch.md (preflight.py), pages/job-arrays.md (array_submit.py), pages/sweeps-and-artifacts.md (core/sweep.py), pages/one-ledger-per-repository.md (submit.py, array_submit.py, test_campaign_shared_ledger.py, README.md), pages/interpreter-availability.md, pages/image-build-flow.md, pages/job-identity-on-cluster.md, pages/unsupported-shapes.md (README.md)
Notes: MCPs board task 7aab32ef. A module the image carries was admitted with no word even when its source had moved since the image's commit, the case that yields numbers from older code. Preflight now compares its package's source in the project's repo between the image's commit and HEAD, prints a WARNING, records it in the ledger row's required `image_drift`, and refuses an image without a readable stamp (`ENTRYPOINT_IMAGE_UNSTAMPED`) or whose commit the repo lacks (`ENTRYPOINT_IMAGE_COMMIT_UNKNOWN`). Real preflights: v23's known-answer probe 194 files behind HEAD, the v56 pilot 25, cleargbm's checkout module not compared. The hub ledger's 16064 earlier rows were backfilled to `image_drift: null`.

## [2026-10-27] new page | repeatable WRDS experiments
Pages written: pages/wrds-repeatable-attempts.md
Pages updated: hubs/operations.md, index.md (29 to 30 pages)
Notes: Board task d43697b3. Added Grid Engine attempt lifecycle to tools/hpc3 with shared RunRecord output. Jobs 40530016 and 40530018 matched code/input/runtime/worker identities, output digests and observations. Failure 40530005 and cancellation 40530008 produced no success receipt.

## [2026-10-28] update | automatic shared research queue verification
Pages updated: pages/wrds-installed-controller-verification.md
Notes: Board task 3dff6fbf. The existing fleet scheduler reconciled retained HPC3 58578878 and WRDS 40534563 through shared dispatch receipts, with zero-exit bounded ticks and automatic board outcomes. Repeat submission and native inspection preserved identities and result digests. The pinned metadata artifact records queue trails, scheduler rows and the frozen runtime; licensed data remains private. README now explains prepared research references alongside the maintained direct CLI routes.

## [2026-10-28] update | fresh research queue submission
Pages updated: pages/wrds-installed-controller-verification.md
Notes: Board task 8b19319c. Automatic ticks submitted and reconciled allocation
58586013 in 11 seconds, exit 0:0, from a fresh admitted queue request. Before/after
replay returned the same receipt and preserved all four earlier journal rows.
Retained WRDS evidence revalidated without new extraction. Added metadata-only
provenance and recorded that the diagnostic candidate/seed budget is consumed.

## [2026-10-28] update | executable replay evidence and CRSP schema discovery
Pages updated: pages/wrds-installed-controller-verification.md
Notes: Board task 8b19319c. Replaced the rejected A2 evidence approach with a
served case that actually repeats submission and reads the live controller journal.
The case passed with two identical receipts and three identical journal readings;
the broader served suite still has three unrelated failures and the board audit
is not declared passed. Metadata-only WRDS job 40538087 returned 331 columns
across 14 relations through existing grid helpers. Exact SQL and CSV stay private;
the committed digest/count record identifies the monthly adapter's candidate
inputs without claiming resolved return semantics or an implemented evaluator.
