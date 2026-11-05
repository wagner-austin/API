---
title: What this package cannot submit, as decisions rather than discoveries
tags: [submission, scope]
hubs: [submission]
related: ["[[chains]]", "[[submission-rules]]", "[[facts-are-code]]"]
source_paths:
  - "src/hpc3/contracts/job.py"
  - "src/hpc3/contracts/cluster.py"
  - "src/hpc3/clusters/hpc3.py"
  - "src/hpc3/core/env_probe.py"
  - "src/hpc3/core/sweep.py"
  - "tests/test_examples.py"
  - "examples/hpc3.json"
  - "examples/chain-sirius-zodiac.json"
  - "runs/hpc3.json"
  - "specs/cleargbm-image.json"
  - "README.md"
source_git_blobs:
  "src/hpc3/contracts/job.py": "9defe2ae9f5f195b2f7d466765669df7ff1977c5"
  "src/hpc3/contracts/cluster.py": "8991c80eff45921db9eefa13f8b593ef577a3807"
  "src/hpc3/clusters/hpc3.py": "ab26dc348ecfa27716bc6d0bf2f08e359c2676b6"
  "src/hpc3/core/env_probe.py": "19d41901aac9ea756003760b58338d1b163a9c47"
  "src/hpc3/core/sweep.py": "6fb57817fbca9e2305f01e1a89f8e30846f7077a"
  "tests/test_examples.py": "7cd49fb8222df061c8f53569df461f4c58d2cd9c"
  "examples/hpc3.json": "f1b8d95c8b811cac36df023c0b4a2d4836dea4ab"
  "examples/chain-sirius-zodiac.json": "c9b083c7e23b7ec7cd860374eb56c638121061d7"
  "runs/hpc3.json": "6265d6ba9ad76de416cc1aa914abf65253b72ceb"
  "specs/cleargbm-image.json": "1642ceb6ccb6cacf6b997f3be03611dae7d9708c"
  "README.md": "6d14fcebcfb6a6eb102b927d5855953bcbb192b2"
fact_checked: 2026-11-04
confidence: high
---

# What this package cannot submit, as decisions rather than discoveries

`JobSpec` describes **one single-node job**, with GPUs or without. The
current table of inexpressible shapes — multi-node/MPI, explicit
`--qos`, `--constraint`/`--exclusive` — lives in the README's
"What this cannot submit" section, where a test (`test_examples.py`) holds it
present and holds lifted limits OUT of it; this page carries the reasoning
and the history the table cannot.[^table]

The README says none of the absent shapes are hard to add, and that they are
absent because they were never built, not because they were judged wrong —
recorded so the gap is a decision rather than a discovery.[^table] The
cluster-facts layer is only partly ready for them. It records why
`standard-hbm` is left out: that partition needs an explicit
`--qos=standard-hbm`, and the module itself declines to emit a QOS directive
for one partition, which is a judgement as well as an omission. It records no
node feature and no per-node shape, so `--constraint` and multi-node would
start from a measurement of the cluster.[^facts]

## Three things left this list

**Job dependencies** were on it and are not any more — see [[chains]]. A run
document's `depends_on` is a `JobSpec` field, and the README test refuses a
"Job dependency" row in the table.[^lifted]

**Job arrays** were on it and are not any more — inverted, in fact: a sweep
is now submitted AS one array call, and the member-by-member loop the row
described no longer exists.[^array] The measured identity rules and the
script-is-the-member-table design are in [[job-arrays]].

**CPU-only** was on it and is not any more. `"gpu": null` on a CPU partition
submits, so `cleargbm_rs`, SIRIUS and ZODIAC are reachable.[^cpu] One caveat
worth stating: a JVM project pins no Python packages, so the pin check holds
nothing about its payload to a version. Since 2026-10-05 its environment is
still probed — `verify_environment` runs the environment's own `bin/python`
whatever the pins say, and refuses one whose interpreter belongs to another
installation — but what the JVM itself is goes unchecked, which is the "both
paths exist, both pass, the results aren't comparable" failure the pin check
was built for ([[environment-pins]]).[^jvm]

[^table]: `README.md:668` § "What this cannot submit"; `src/hpc3/contracts/job.py:77` `JobSpec`; `tests/test_examples.py:158` `test_the_readme_states_what_it_cannot_submit` and `:170` `test_limits_that_were_lifted_are_not_still_listed_as_limits`.
[^facts]: `src/hpc3/clusters/hpc3.py:39` `HPC3`, whose docstring's "Deliberately absent" list gives the `standard-hbm` reason; `src/hpc3/contracts/cluster.py:77` `PartitionFacts` and `:123` `ClusterFacts` hold usage factor, preemption, hours, GPU models and per-user caps, and no node feature or node count.
[^lifted]: `src/hpc3/contracts/job.py:177` `depends_on`; `tests/test_examples.py:170` `test_limits_that_were_lifted_are_not_still_listed_as_limits`.
[^array]: `src/hpc3/core/sweep.py:27` `submit_sweep`, which submits every member through `submit_array`; the same README test refuses a "Job array" row.
[^cpu]: `tests/test_examples.py:138` `test_the_documented_cpu_only_entry_really_is_cpu_only` and `:228` `test_it_runs_on_the_free_cpu_partition_holding_no_gpu`, over `examples/chain-sirius-zodiac.json`; `runs/hpc3.json` project `cleargbm` (`"gpu": null`, partition `free`), whose image spec `specs/cleargbm-image.json` installs the `cleargbm_rs` wheel.
[^jvm]: `examples/hpc3.json` project `sirius`, `"pinned_packages": {}`; `src/hpc3/core/env_probe.py:233` `verify_environment` (`ENV_INTERPRETER_BORROWED`) and `:79` `probe_command`, which runs only the environment's `bin/python`. The unconditional probe landed in API f5df5fd93 on 2026-10-05.
