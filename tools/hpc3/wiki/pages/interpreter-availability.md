---
title: The interpreter every project needs is not in `module avail python`
tags: [cluster-facts, environments, onboarding]
hubs: [cluster-facts]
related: ["[[environment-pins]]", "[[facts-are-code]]", "[[image-build-flow]]", "[[invariant-placement]]"]
source_paths:
  - "src/hpc3/clusters/hpc3.py"
  - "src/hpc3/core/bootstrap.py"
  - "src/hpc3/core/env_probe.py"
  - "README.md"
  - "pyproject.toml"
source_git_blobs:
  "src/hpc3/clusters/hpc3.py": "e6fedebb13c20222c9269b158f0ebed7fbf84cc9"
  "src/hpc3/core/bootstrap.py": "689051513c67eb0978667f2fe109c83b0baab1d1"
  "src/hpc3/core/env_probe.py": "e83c330acd07bdb53dfdcc8fe1ee8a64de3af529"
  "README.md": "104d4dac210676c88c42b5122146c11687239fc5"
  "pyproject.toml": "feba6fe164021d49f8d3e9c2cba0117ab48ee75e"
provenance:
  - "module -t avail python on hpc3 login-i15, 2026-09-03: python/2.7.17, 3.8.0, 3.10.2, 3.14.3"
  - "/usr/bin/python3 -V on hpc3 login-i15, 2026-09-03: Python 3.9.25; which python3.11 finds nothing"
  - "module -t avail | grep conda on hpc3 login-i15, 2026-09-03: anaconda/{2020.07,2021.11,2022.05,2024.06,2025.12}, miniconda3/{23.5.2,24.9.2}, mamba/{24.3.0,26.1.0}, bioconda/4.8.3"
  - "/pub/wagnera3/envs/{abl-pinned,cleargbm,tankpit}/bin/python -V, 2026-09-03: all Python 3.11.16"
  - "/pub/wagnera3/envs/tankpit/pyvenv.cfg read 2026-09-03: home = /pub/wagnera3/envs/cleargbm/bin, version 3.11.16"
  - "ls -l /pub/wagnera3/envs/tankpit/bin/python3.11, 2026-09-03: symlink to /pub/wagnera3/envs/cleargbm/bin/python3.11"
  - "hpc3-bootstrap run live 2026-09-03: created /pub/wagnera3/envs/_bootstrap_selftest (python 3.11.16, base_prefix its own path, real 25MB bin/python3.11), refused a second run with BOOTSTRAP_ENV_EXISTS, removed afterwards"
  - "sys.version on that interpreter, 2026-09-03: '3.11.16 | packaged by conda-forge | (main, Aug 21 2026, 22:44:51) [GCC 14.4.0]'; sys.base_prefix is /pub/wagnera3/envs/cleargbm"
fact_checked: 2026-09-03
confidence: high
---

# The interpreter every project needs is not in `module avail python`

`module -t avail python` answers with four interpreters and **none of them is
the one anything here runs on**:

```
python/2.7.17   python/3.8.0   python/3.10.2   python/3.14.3
```

The system interpreter is `Python 3.9.25`, and `which python3.11` finds
nothing.[^1] Meanwhile every project in this monorepo declares `python = "^3.11"`
— TankpitBot, RustedWarfareBot, Model-Trainer, `platform_core`, and
`tools/hpc3` itself.[^2]

So the interpreter this whole stack requires exists on that cluster in **no
form the obvious command will show you**: not as a `python` module, not as a
system binary. A new project's first act is therefore to discover that its
first act cannot be `module load python`.

## The answer is real, and it is filed under a different name

3.11 is available. It is not under `python`:[^3]

```
anaconda/{2020.07,2021.11,2022.05,2024.06,2025.12}
miniconda3/{23.5.2,24.9.2}
mamba/{24.3.0,26.1.0}
```

Every environment on this cluster was built through that door.
`/pub/wagnera3/envs/abl-pinned`, `/pub/wagnera3/envs/cleargbm` and
`/pub/wagnera3/envs/tankpit` all report `Python 3.11.16`, and the interpreter
names its own origin when asked:[^4]

```
3.11.16 | packaged by conda-forge | (main, Aug 21 2026, 22:44:51) [GCC 14.4.0]
```

**This is the whole cliff, and it is one sentence long**: search for `python`
and the cluster tells you 3.11 does not exist; search for `conda` and it does.
When this page was written a grep of the README found no mention of Python
versions and none of bootstrapping, so nothing closed that gap for the next
reader. It does now — the quick start opens with `hpc3-bootstrap` and says
why `--python 3.11` cannot come from `module load python`.[^5]

It is also why *every registered project ships an image* is load-bearing
rather than stylistic. The image is where a pinned 3.11 comes from once the
project is running; conda is where it comes from before the image exists.

## The bootstrap left a dependency nobody declared

`envs/tankpit` is not a conda environment. It is a **venv whose interpreter is
a symlink into another project's environment**, which its own `pyvenv.cfg`
records:

```
home = /pub/wagnera3/envs/cleargbm/bin
executable = /dfs6b/pub/wagnera3/envs/cleargbm/bin/python3.11
command = /pub/wagnera3/envs/cleargbm/bin/python3.11 -m venv /pub/wagnera3/envs/tankpit
```

`bin/python3.11` is a symlink to `envs/cleargbm/bin/python3.11`, and
`sys.base_prefix` is `/pub/wagnera3/envs/cleargbm`. **Deleting or moving the
cleargbm environment breaks the tankpit one**, and nothing in either project's
run document says the two are related.[^6]

This is not carelessness by whoever did it — it is the only move available
when the first environment has no supported path. An improvised bootstrap
leaves an undeclared edge between projects, and that edge is invisible to
every check the package has: `check_env_path` proves the directory exists,
`verify_env_packages` proves the packages are pinned, and both keep passing
right up until the day the other project is cleaned up.

## Half of this is code now

`hpc3-bootstrap` was built on 2026-09-03 and carries the conda door as a
pinned constant rather than as advice on this page: `CONDA_MODULE =
"miniconda3/24.9.2"`, joined to `conda create` in ONE command line because
each SSH call is its own shell and a `module load` in a separate call is gone
before the next one runs.[^9]

It also refuses to hand back an environment that is not what was asked for.
After creating one it runs that environment's own interpreter and checks two
things — the version, and `sys.base_prefix`. The second is the one that
matters here: it is what distinguishes an environment that owns its
interpreter from one pointing at somebody else's, and it is measurable on the
three that already exist.[^10]

Proven end to end against the cluster on 2026-09-03, not only in tests: the
command created `/pub/wagnera3/envs/_bootstrap_selftest`, reported Python
3.11.16 self-contained at its own path, and a second identical invocation
refused with `BOOTSTRAP_ENV_EXISTS` rather than writing into it. The created
environment carried a real 25 MB `bin/python3.11`, against the 42-byte symlink
`envs/tankpit` carries. The self-test environment was removed afterwards.[^11]

## What is still not code

**The cluster's interpreter INVENTORY.** `src/hpc3/clusters/hpc3.py` still
mentions neither `python` nor `interpreter`.[^7] Under [[facts-are-code]] the
list at the top of this page belongs there, beside the partitions and the QOS
ceilings, so a rule can ask the cluster module instead of a person remembering
this page. Bootstrap knows which door to open; nothing knows which doors exist.

**Existing environments are still never checked.** The version and
`base_prefix` checks run only on the CREATING path, so they prevent the next
borrowed environment and cannot see the current one. `env_probe` makes an SSH
round trip on every preflight, runs the environment's own interpreter, and
asks it only for its distributions — never for `sys.version_info` or
`sys.base_prefix`.[^8] So `envs/tankpit` remains exactly as described above,
and preflight would pass it today.

That gap is deliberate rather than forgotten. Adding the check to preflight
makes it a thirty-fifth refusal on the running path, which is the asymmetry
this work exists to close; putting it where an environment is BUILT costs
nothing to a project that already works. Whether the running path should also
carry it is a live question, recorded on board task 3b6f3848 rather than
decided here.

[^1]: `module -t avail python`, `/usr/bin/python3 -V` and `which python3.11`, run over SSH on hpc3 login-i15 at 2026-09-03 19:58 UTC. Output verbatim: `python/2.7.17`, `python/3.8.0`, `python/3.10.2`, `python/3.14.3`; `Python 3.9.25`; `no python3.11 in (/opt/rcic/bin:/usr/share/Modules/bin:/usr/local/bin:/usr/bin:...)`.
[^2]: The `python = "^3.11"` line in each of `clients/TankpitBot/pyproject.toml`, `clients/RustedWarfareBot/pyproject.toml`, `services/Model-Trainer/pyproject.toml`, `libs/platform_core/pyproject.toml` and `tools/hpc3/pyproject.toml`, read 2026-09-03.
[^3]: `module -t avail 2>&1 | grep -i -E "conda|mamba|miniforge|anaconda"` on hpc3 login-i15 at 2026-09-03 20:12 UTC, returning the eleven modules listed above. The same host's `module -t avail python` (footnote 1, fourteen minutes earlier) lists none of them, which is why searching by the obvious name answers "no 3.11".
[^4]: `for e in /pub/wagnera3/envs/*/; do "$e/bin/python" -V; done` on hpc3 login-i15 at 2026-09-03 20:11 UTC: `/pub/wagnera3/envs/abl-pinned/`, `/pub/wagnera3/envs/cleargbm/` and `/pub/wagnera3/envs/tankpit/` each report `Python 3.11.16`. The first two carry `/pub/wagnera3/envs/<name>/conda-meta/`; the third does not, and is a venv (see below).
[^5]: Both greps were empty when measured on 2026-09-03 before the command existed: `grep -n "3\.11\|python_version\|module load\|module avail" README.md` and `grep -n -i bootstrap README.md`. The README's "Onboarding a project that does not exist yet" block and its `hpc3-bootstrap` row in the command table were added the same day, in the commit that added the command.
[^6]: `ls -l /pub/wagnera3/envs/tankpit/bin/python3.11` at 2026-09-03 20:12 UTC → `lrwxr-xr-x ... -> /pub/wagnera3/envs/cleargbm/bin/python3.11` (42-byte link, dated Sep 2 22:15); `sys.base_prefix` read from that same interpreter in the same call returns `/pub/wagnera3/envs/cleargbm`. The venv resolves today, so this is a latent dependency rather than a current breakage.
[^7]: `grep -n -i "python\|interpreter" src/hpc3/clusters/hpc3.py` → no matches, 2026-09-03.
[^8]: `src/hpc3/core/env_probe.py`, `_PROBE_SOURCE`: the probe iterates `importlib.metadata.distributions()` and prints name, version and wheel tag. It never reads `sys.version_info`. `verify_env_packages` also returns before making the round trip at all when `pinned == {}`, so a project with no package pins gets no interpreter evidence whatsoever.
[^9]: `src/hpc3/core/bootstrap.py`, `CONDA_MODULE` and `create_command`. The join is asserted by `test_module_load_and_conda_create_are_one_command`, which exists because separate calls fail only against a real cluster and look correct in review.
[^10]: `src/hpc3/core/bootstrap.py`, `check_identity`, raising `BOOTSTRAP_PYTHON_MISMATCH` and `BOOTSTRAP_ENV_NOT_SELF_CONTAINED`. Measured against all three live environments 2026-09-03 20:11 UTC: `abl-pinned` and `cleargbm` report their own paths as `base_prefix`; `tankpit` reports `/pub/wagnera3/envs/cleargbm`.
[^11]: `poetry run python -m hpc3.cli.bootstrap --config runs/hpc3-tankpit.json --project bootstrap-selftest --env-path /pub/wagnera3/envs/_bootstrap_selftest --python 3.11`, run 2026-09-03 22:00 UTC. Second invocation exited 2 with `BOOTSTRAP_ENV_EXISTS`. `ls -l` on the created `bin/python3.11` showed a 25,916,456-byte regular file; the same listing for `envs/tankpit` shows a 42-byte symlink. Environment removed (32 MB) and the three real environments confirmed intact.
