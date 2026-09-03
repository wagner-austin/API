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
