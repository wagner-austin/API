---
title: The interpreter every project needs is not in `module avail python`
tags: [cluster-facts, environments, onboarding]
hubs: [cluster-facts]
related: ["[[environment-pins]]", "[[facts-are-code]]", "[[image-build-flow]]", "[[invariant-placement]]"]
source_paths:
  - "src/hpc3/clusters/hpc3.py"
  - "src/hpc3/core/bootstrap.py"
  - "src/hpc3/core/env_probe.py"
  - "src/hpc3/core/interpreter.py"
  - "README.md"
  - "pyproject.toml"
source_git_blobs:
  "src/hpc3/clusters/hpc3.py": "ab26dc348ecfa27716bc6d0bf2f08e359c2676b6"
  "src/hpc3/core/bootstrap.py": "4b8e0a34981b22430f8465bdafe040a579db03b7"
  "src/hpc3/core/env_probe.py": "19d41901aac9ea756003760b58338d1b163a9c47"
  "src/hpc3/core/interpreter.py": "44152c39c9674c78771a2f01288110c0438c5617"
  "README.md": "7ac1acb5c894dafe7537926323ae96d7d5b57dbe"
  "pyproject.toml": "48c300ade70f6240e0aadf87dbf988e96fbe3dfe"
provenance:
  - "module -t avail python on hpc3 login-i15, 2026-09-03: python/2.7.17, 3.8.0, 3.10.2, 3.14.3"
  - "/usr/bin/python3 -V on hpc3 login-i15, 2026-09-03: Python 3.9.25; which python3.11 finds nothing"
  - "module -t avail | grep conda on hpc3 login-i15, 2026-09-03: anaconda/{2020.07,2021.11,2022.05,2024.06,2025.12}, miniconda3/{23.5.2,24.9.2}, mamba/{24.3.0,26.1.0}, bioconda/4.8.3"
  - "/pub/wagnera3/envs/{abl-pinned,cleargbm,tankpit}/bin/python -V, 2026-09-03: all Python 3.11.16"
  - "/pub/wagnera3/envs/tankpit/pyvenv.cfg read 2026-09-03: home = /pub/wagnera3/envs/cleargbm/bin, version 3.11.16"
  - "ls -l /pub/wagnera3/envs/tankpit/bin/python3.11, 2026-09-03: symlink to /pub/wagnera3/envs/cleargbm/bin/python3.11"
  - "hpc3-bootstrap run live 2026-09-03: created /pub/wagnera3/envs/_bootstrap_selftest (python 3.11.16, base_prefix its own path, real 25MB bin/python3.11), refused a second run with BOOTSTRAP_ENV_EXISTS, removed afterwards"
  - "sys.version on that interpreter, 2026-09-03: '3.11.16 | packaged by conda-forge | (main, Aug 21 2026, 22:44:51) [GCC 14.4.0]'; sys.base_prefix is /pub/wagnera3/envs/cleargbm"
  - "sys.base_prefix of /pub/wagnera3/envs/{abl-cu128,abl-pinned,cleargbm,tankpit}/bin/python on hpc3 login-i16, 2026-10-05: each its own path except tankpit, still /pub/wagnera3/envs/cleargbm"
  - "/opt/env/bin/python inside all eight registered .sif images on hpc3 login-i16, 2026-10-05: version 3.11, sys.base_prefix /usr/local, bin/python a symlink to /usr/local/bin/python"
  - "os.stat st_dev inside tankpit v2 and cleargbm-v1 images, 2026-10-05: / = /usr/local = /opt/env = 2097157; /pub/wagnera3 = /dfs6b/pub/wagnera3/envs/cleargbm = 46; /proc/mounts lists beegfs_dfs6b /dfs6b although only /pub/wagnera3 is bound"
  - "probe_environment run live against hpc3, 2026-10-05: three images and three host envs pass, envs/tankpit refused ENV_INTERPRETER_BORROWED"
fact_checked: 2026-10-05
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
every check the package had: `check_env_path` proved the directory exists,
`verify_env_packages` proved the packages are pinned, and both kept passing
right up until the day the other project is cleaned up. Since 2026-10-05 the
probe also asks the interpreter where it came from, and refuses this edge —
see below.

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

**The interpreter's version is reported but not yet held to anything.** The
probe reads `sys.version_info` on every round trip, but no run document
declares the version a project expects, so outside bootstrap (which compares it
to `--python`) there is nothing to compare it against.

## Existing environments are checked now, by one rule with two shapes

Until 2026-10-05 the `base_prefix` check ran only on the CREATING path, so it
prevented the next borrowed environment and could not see the current one, and
`env_probe` asked the interpreter only for its distributions.[^8] Board task
3b6f3848 settled the open question this page used to record: a borrowed
interpreter is a defect in a running project too, so the check now runs
wherever an environment is probed — bootstrap, preflight (single jobs and
arrays) and image capture.

The probe's answer now OPENS with three identity lines — `major.minor`,
`sys.base_prefix`, and whether that prefix sits on the same device as `/` —
and an answer without them is `ENV_PROBE_UNREADABLE`. The pin-less early
return is gone, so a project that pins nothing (`rusted`) is probed too.[^10]

The rule cannot be "`base_prefix` equals `env_path`" everywhere, and the
measurement is why. Inside all eight registered images, `/opt/env` is a venv
whose `bin/python` links to `/usr/local/bin/python`, so its base is
`/usr/local` — inside the same digest-pinned image, which is not borrowing.
Held to the host rule, every registered project would be refused. And the
image's bind list cannot decide it either: `/dfs6b` is mounted inside the
container although only `/pub/wagnera3` is bound, so a venv pointing at
`/dfs6b/pub/wagnera3/envs/cleargbm` would pass any bind check.[^12] So:

- a **host** environment must be its own base (`base_prefix == env_path`);
- an **image** environment's base must be on the container's root device —
  `/`, `/usr/local` and `/opt/env` report one `st_dev` (the overlay root),
  `/dfs6b/...` and `/pub/wagnera3` another (BeeGFS).

Either failure is `ENV_INTERPRETER_BORROWED`, one code for bootstrap, preflight
and capture alike. Run live against the cluster through the real code on
2026-10-05: the tankpit, cleargbm and rusted images and the `cleargbm`,
`abl-pinned` and `abl-cu128` host environments pass; `envs/tankpit` is refused,
naming `envs/cleargbm`.[^13]

**`envs/tankpit` itself is NOT repaired, deliberately.** It is the environment
the tankpit image's `org.corvis.env-source` label names, so rebuilding it in
place would leave that label pointing at something that never built anything.
What changed is that it can no longer be used silently: `hpc3-image-capture
--env-path /pub/wagnera3/envs/tankpit` now refuses before writing a spec, so a
re-capture has to start from an environment `hpc3-bootstrap` built.

[^1]: `module -t avail python`, `/usr/bin/python3 -V` and `which python3.11`, run over SSH on hpc3 login-i15 at 2026-09-03 19:58 UTC. Output verbatim: `python/2.7.17`, `python/3.8.0`, `python/3.10.2`, `python/3.14.3`; `Python 3.9.25`; `no python3.11 in (/opt/rcic/bin:/usr/share/Modules/bin:/usr/local/bin:/usr/bin:...)`.
[^2]: The `python = "^3.11"` line in each of `clients/TankpitBot/pyproject.toml`, `clients/RustedWarfareBot/pyproject.toml`, `services/Model-Trainer/pyproject.toml`, `libs/platform_core/pyproject.toml` and `tools/hpc3/pyproject.toml`, read 2026-09-03.
[^3]: `module -t avail 2>&1 | grep -i -E "conda|mamba|miniforge|anaconda"` on hpc3 login-i15 at 2026-09-03 20:12 UTC, returning the eleven modules listed above. The same host's `module -t avail python` (footnote 1, fourteen minutes earlier) lists none of them, which is why searching by the obvious name answers "no 3.11".
[^4]: `for e in /pub/wagnera3/envs/*/; do "$e/bin/python" -V; done` on hpc3 login-i15 at 2026-09-03 20:11 UTC: `/pub/wagnera3/envs/abl-pinned/`, `/pub/wagnera3/envs/cleargbm/` and `/pub/wagnera3/envs/tankpit/` each report `Python 3.11.16`. The first two carry `/pub/wagnera3/envs/<name>/conda-meta/`; the third does not, and is a venv (see below).
[^5]: Both greps were empty when measured on 2026-09-03 before the command existed: `grep -n "3\.11\|python_version\|module load\|module avail" README.md` and `grep -n -i bootstrap README.md`. The README's "Onboarding a project that does not exist yet" block and its `hpc3-bootstrap` row in the command table were added the same day, in the commit that added the command.
[^6]: `ls -l /pub/wagnera3/envs/tankpit/bin/python3.11` at 2026-09-03 20:12 UTC → `lrwxr-xr-x ... -> /pub/wagnera3/envs/cleargbm/bin/python3.11` (42-byte link, dated Sep 2 22:15); `sys.base_prefix` read from that same interpreter in the same call returns `/pub/wagnera3/envs/cleargbm`. The venv resolves today, so this is a latent dependency rather than a current breakage.
[^7]: `grep -n -i "python\|interpreter" src/hpc3/clusters/hpc3.py` → no matches, 2026-09-03, and again 2026-10-05 on the blob now pinned.
[^8]: Before 2026-10-05, `src/hpc3/core/env_probe.py`'s `_PROBE_SOURCE` iterated `importlib.metadata.distributions()` and printed name, version and wheel tag, never reading `sys.version_info`, and `verify_env_packages` returned before the round trip when `pinned == {}`. Both are gone: `_PROBE_SOURCE` now prepends `IDENTITY_LINES`, and `verify_environment` always probes.
[^9]: `src/hpc3/core/bootstrap.py`, `CONDA_MODULE` and `create_command`. The join is asserted by `test_module_load_and_conda_create_are_one_command`, which exists because separate calls fail only against a real cluster and look correct in review.
[^10]: `src/hpc3/core/interpreter.py`, `IDENTITY_LINES`, `split_identity` and `check_interpreter_home` (raising `ENV_INTERPRETER_BORROWED`, which replaced `BOOTSTRAP_ENV_NOT_SELF_CONTAINED`); `src/hpc3/core/env_probe.py`, `probe_environment` and `verify_environment`; `src/hpc3/core/bootstrap.py`, `check_identity`, which keeps `BOOTSTRAP_PYTHON_MISMATCH` and delegates the base check. Measured against all three host environments 2026-09-03 20:11 UTC: `abl-pinned` and `cleargbm` report their own paths as `base_prefix`; `tankpit` reports `/pub/wagnera3/envs/cleargbm`.
[^11]: `poetry run python -m hpc3.cli.bootstrap --config runs/hpc3-tankpit.json --project bootstrap-selftest --env-path /pub/wagnera3/envs/_bootstrap_selftest --python 3.11`, run 2026-09-03 22:00 UTC. Second invocation exited 2 with `BOOTSTRAP_ENV_EXISTS`. `ls -l` on the created `bin/python3.11` showed a 25,916,456-byte regular file; the same listing for `envs/tankpit` shows a 42-byte symlink. Environment removed (32 MB) and the three real environments confirmed intact.
[^12]: Over SSH on hpc3 login-i16, 2026-10-05 ~07:50 UTC, `apptainer exec --bind /pub/wagnera3 <sif> sh -c '/opt/env/bin/python -c ...; ls -l /opt/env/bin/python'` for each of the eight images `runs/hpc3*.json` registers: every one printed `3.11` and `/usr/local`, and `/opt/env/bin/python -> /usr/local/bin/python`. In the tankpit v2 and cleargbm-v1 images, `os.stat(p).st_dev` gave 2097157 for `/`, `/usr/local` and `/opt/env`, 46 for `/pub/wagnera3` and `/dfs6b/pub/wagnera3/envs/cleargbm`, 66305 for `/tmp`; `/proc/mounts` listed `overlay /` and `beegfs_dfs6b /dfs6b` as well as the bound `/pub/wagnera3`.
[^13]: `hpc3.core.env_probe.probe_environment` called from a scratch script against host `hpc3` (login-i16), 2026-10-05 ~08:05 UTC, with the source of this change: images tankpit v2, cleargbm-v1 and rusted v5 returned `{'version': '3.11', 'base_prefix': '/usr/local', 'base_on_root_filesystem': True}` with 38, 96 and 6 distributions; host `cleargbm`, `abl-pinned`, `abl-cu128` returned their own paths with `False`; host `tankpit` raised `ENV_INTERPRETER_BORROWED: /pub/wagnera3/envs/tankpit runs Python 3.11 belonging to /pub/wagnera3/envs/cleargbm`.
