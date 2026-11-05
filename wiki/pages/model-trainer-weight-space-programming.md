---
title: Weight-space programming on our own models -- a car policy fixed, a GPT-2 loop edit that backfired
tags: [ml, model-trainer, model-editing, weight-space, gpt2, reinforcement-learning, measurement]
related:
  - "[[corpus-attachment-program]]"
  - "[[model-trainer-run-record-provenance]]"
  - "[[model-trainer-cloze-scoring-path]]"
source_paths:
  - services/Model-Trainer/src/model_trainer/core/contracts/weight_program.py
  - services/Model-Trainer/src/model_trainer/core/services/model/weight_space/program.py
  - services/Model-Trainer/src/model_trainer/core/services/model/weight_space/mountain_car.py
  - services/Model-Trainer/src/model_trainer/core/services/model/weight_space/control_policy.py
  - services/Model-Trainer/src/model_trainer/core/services/model/weight_space/repetition.py
  - services/Model-Trainer/src/model_trainer/core/services/model/weight_space/lm_edit_arm.py
  - services/Model-Trainer/src/model_trainer/core/services/model/editing/verify.py
  - services/Model-Trainer/results/weight-space/mountain-car-v57/edit-record.json
  - services/Model-Trainer/results/weight-space/gpt2-v57/record.json
  - services/Model-Trainer/results/weight-space/gpt2-v57/reader-reasoning.md
source_git_blobs:
  "services/Model-Trainer/src/model_trainer/core/contracts/weight_program.py": d5e612b1a2995aad090162a41c0e09c5164666e1
  "services/Model-Trainer/src/model_trainer/core/services/model/weight_space/program.py": e8bc95ff3f17829f6ef08c40a40e9ce30466f10f
  "services/Model-Trainer/src/model_trainer/core/services/model/weight_space/mountain_car.py": 124a9fca4a57013eb8439c269e6c4e33b87334dd
  "services/Model-Trainer/src/model_trainer/core/services/model/weight_space/control_policy.py": 5f94326d940d1dc1be9ea9bb61337f7c696c9531
  "services/Model-Trainer/src/model_trainer/core/services/model/weight_space/repetition.py": 158a8662be295f2573931c691aa726bc88dbdd90
  "services/Model-Trainer/src/model_trainer/core/services/model/weight_space/lm_edit_arm.py": 1eb8a9a1b0029dc7919f7b328753ece4e51288b7
  "services/Model-Trainer/src/model_trainer/core/services/model/editing/verify.py": 2aaa0fe35c0cd1bc41441fbbc22d7eedb51154f9
  "services/Model-Trainer/results/weight-space/mountain-car-v57/edit-record.json": 74007ff7f0b65d78b5f3f2dda6bb759824e2f737
  "services/Model-Trainer/results/weight-space/gpt2-v57/record.json": ba8d76c155ee6de7213bcf936b0ef2294444ebb8
  "services/Model-Trainer/results/weight-space/gpt2-v57/reader-reasoning.md": f85979ab13993ba6ac4fb7f9d35ea392d64bf1f4
provenance:
  - "the post reproduced: Takehi, Teoh, Zhang, Khattab, 'Sparks of Weight-Space Programming', MIT CSAIL, 2026-11-02 -- Experiment 4 (an LLM fixes a failing RL car policy by editing its weights) and the shape of Experiment 8 (an LLM edits a language model's weights against repetition loops)"
  - "HPC3 job weight-space-mountain-car-train-v57: REINFORCE run, label mountain-car-train-seed1-h32-it200-b64-g0.9-lr0.01, policy digest sha256:b800b62057e499627d5fae8698814ef443b29db1ba498a50a8df343dd4f1dd62, base 0 of 1,024 episodes"
  - "HPC3 job 58906732 (partition free, CPU): weight_space_policy_edit, RunRecord payload_digest sha256:053c2f00429014966f2920eb1071370b3fc3fb62a07b9d3445003b2821272084, /pub/wagnera3/mi/weight-space/mountain-car-v57/edit-velocity-sign-pump.json"
  - "HPC3 job 58906742 (partition free-gpu, Tesla V100-FHHL-16GB, driver 580.82.07, 25 min): weight_space_lm_edit, RunRecord payload_digest sha256:0f64d4f03790a9781b03112fc15bbdbab445b98e8b0f48e3eab1aaefed0ccb35, /pub/wagnera3/mi/weight-space/gpt2-v57/record.json and outcomes.json"
  - "image v57 /pub/wagnera3/images/v57/abl, sha256 07c652ca5efb29317fb9aa32001fc1ebbc6918c306f046bd2e3af16726b287d3, built from API commit 30573d988784417b377153ccb56b81b0d2669b3f"
  - "gpt2 weights file model.safetensors sha256 248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707; cloze set /pub/wagnera3/floor/cloze-items-80f9732a.jsonl, 2,627 items"
  - "both programs were written by fresh Claude Opus 5.5 sessions briefed only with the files under services/Model-Trainer/results/weight-space/*/reader-prompt.md; neither saw any measurement of its edit before the HPC3 run"
  - "board task 7964f9f6-68d6-43c2-8565-bb8e387ca694 carries the full trail, including the A1 and A2 result posts"
fact_checked: "2026-11-05"
confidence: high
hubs: [services]
---

# Weight-space programming on our own models

The post "Sparks of Weight-Space Programming" (MIT CSAIL, 2026-11-02) has a
language model read a network's raw weights and write an edit to them, with
no gradients. Model-Trainer now has a harness for that, and it was run once
at each end of the scale. On a 1,251-weight car policy the edit worked
completely. On GPT-2 small it made the targeted failure worse.

| | untouched control | after the LLM-written edit | weights changed |
|---|---|---|---|
| MountainCar policy, episodes reaching the flag | 0 / 1,024 | **1,024 / 1,024** (restored: 0 / 1,024) | 135 of 1,251 |
| GPT-2 small, greedy continuations that end in a loop | 44 / 256 | **125 / 256** (McNemar p = 1.8e-14) | 344,064 of 124,439,808 |
| GPT-2 small, cloze accuracy (collateral) | 1,374 / 2,627 | 1,329 / 2,627 (p = 0.016) | |

## The harness: a program, written against a digest, removed and checked

An edit is a **weight program**: a list of `set`, `scale` and `project_out`
operations on the rows or columns of named parameters. It also carries
`written_against`, the digest of the weights the writer read.[^contract] The
applier refuses a program whose `written_against` does not match the model in
hand. It also refuses one that changed any parameter it did not name, with the
every-parameter digest check the earlier rank-one editing arm built.[^apply]
`remove_weight_program` restores every snapshot. It then re-digests the whole
network and raises `EDIT_RESTORE_MISMATCH` if anything differs, so every
measured run ends by proving it left the control's weights intact.[^remove]
The weights digest is a sha256 over the sorted per-parameter digests.[^digest]

The writer never sees a result. Each reader is a fresh session that gets one
brief and writes one program, which then goes to HPC3. The briefs are
committed verbatim beside the results.

## A1: the car policy, fixed by one path through the network

`mountain_car.py` is Gym MountainCar-v0's dynamics: engine force 0.001,
gravity 0.0025, 200 steps, starts drawn from [-0.6, -0.4).[^env] A 2-32-32-3
tanh policy[^policy] trained by REINFORCE with a myopic discount of 0.9 settled
into always pushing right. That never builds the swing the hill needs, and the
policy reached the flag in 0 of 1,024 episodes.

The reader got the weights as text (three decimals), the dynamics, and one
recorded episode. It made no tool calls. It diagnosed the failure as the
network ignoring velocity, and wrote the textbook energy-pumping rule into
unit 0 of each hidden layer: `h1[0] = tanh(200 v)`, `h2[0] = tanh(5 h1[0])`,
push-left score `-h2[0]`, push-right score `+h2[0]`, and a no-push bias of -1.
On the same 1,024 starts the edited policy reached the flag in all 1,024, at a
mean of 128.5 steps. The reader had predicted 100 to 130 without simulating.
Removing the edit gave 0 of 1,024 again.

What this does and does not show: the reader did not repair the trained
circuit. It overwrote one path with a controller it already knew. The network
was small enough to read whole, and the brief gave it the dynamics. That
matches the post's experiment in shape. It is not evidence that an LLM can
find and fix a defect it cannot name in advance.

## A2: GPT-2's loops, made worse by cutting the copy circuit

The failure targeted was greedy repetition. A continuation is **looped** when
its last 32 tokens are exactly periodic with period 16 or less.[^loop] Each of
the 256 prompts is the first 8 to 48 tokens of a cloze sentence from the
extraction-eval set, and 96 tokens are decoded greedily from each. The
untouched model is measured first, in the same process on the same card. Loop
flags and cloze correctness are paired per prompt and per item, and judged by
exact McNemar.[^arm]

The reader worked from weight products with numpy, without running the network
on any sequence. It found 4.11 as the previous-token head by K-composition,
and seven induction heads composing with it (5.0, 5.1, 5.5, 5.8, 6.9, 7.2 and
7.10). It hypothesised that greedy loops are the induction circuit feeding on
itself.[^reasoning] Its program zeroed those heads' value columns in
`c_attn`, 0.28% of the weights, and it put its confidence that loops would
drop at 0.6.

They rose. 100 prompts that the control continued without looping now looped,
and 19 that looped under the control now did not: 44 to 125 of 256. Seq-rep-4
rose from 0.515 to 0.636. Cloze accuracy fell 1.7 points (189 items
right-to-wrong, 144 wrong-to-right). The new loops are short-period, of a kind
the control did not produce: ", Lot 23, Lot 23, Lot 23", "and 1 ARA job with
10 open, and 1 ARA job with 10 open". The reader had listed exactly this risk:
with copying removed, "a mostly local predictor can still settle into short
cycles". Taking away the long-range copy path did not stop looping. It left a
more local predictor that loops sooner.

## What one negative does and does not settle

One reader, one program, one shot, with no measurement fed back. The post's
loop experiment was on a different model, and this run does not show that
weight-space programming fails at GPT-2 scale. What it does show is narrower,
and it is what the next attempt should start from:

- **"Loops are induction" is the wrong mechanism to cut at, on this measure.**
  Removing seven induction heads' value paths more than doubled loops. A
  further attempt that targets the same circuit should expect the same result.
- **The harness makes iteration cheap.** A program is a JSON file, and the
  measuring run took 25 minutes on a free V100. The control, the pairing and
  the restore check come with every run. A writer that sees the paired flags
  of its last attempt is the obvious next arm; this run deliberately did not
  give it that.
- **The small-network result transfers less than it looks.** The car fix
  worked because the reader could replace the policy outright with a rule it
  already knew. GPT-2 offers no such rule to write in.

Why the earlier GPT-2 editing work never got here: that arm (board task
d3639e09) computed rank-one updates algebraically from activations, so the
language model was never the one reading the weights. This harness reuses that
arm's snapshot, restore and digest checks, and adds the program contract that
lets a writer, human or LLM, hand in an edit as data.

[^contract]: `services/Model-Trainer/src/model_trainer/core/contracts/weight_program.py:243` -- `decode_weight_program(obj)` requires a non-empty `ops` list, a `program_id` and a `written_against`; `decode_weight_op` at line 179 refuses an axis other than 0 or 1, empty, negative or duplicate indices, non-finite values, a `scale` with other than one value, and a zero `project_out` direction.
[^apply]: `services/Model-Trainer/src/model_trainer/core/services/model/weight_space/program.py:234` -- `apply_weight_program(model, program)` refuses a `written_against` mismatch and any changed parameter the program did not name, with `EDIT_VERIFICATION_FAILED`.
[^remove]: `services/Model-Trainer/src/model_trainer/core/services/model/weight_space/program.py:293` -- `remove_weight_program(model, applied)` restores each snapshot, then compares every parameter's digest with the digests taken before the program ran.
[^digest]: `services/Model-Trainer/src/model_trainer/core/services/model/editing/verify.py:123` -- `weights_digest(digests)` hashes the canonical JSON of the sorted `[name, digest]` pairs that `parameter_digests` (line 109) produces.
[^env]: `services/Model-Trainer/src/model_trainer/core/services/model/weight_space/mountain_car.py:148` -- `evaluate_policy` runs the greedy policy from each start and counts the episodes that reach the goal within the step limit.
[^policy]: `services/Model-Trainer/src/model_trainer/core/services/model/weight_space/control_policy.py:59` -- `ControlPolicy` holds `hidden_in`, `hidden_mid` and `action_out` weights and biases by name; its `forward` is tanh, tanh, then linear, and `greedy` takes the argmax.
[^loop]: `services/Model-Trainer/src/model_trainer/core/services/model/weight_space/repetition.py:154` -- `ends_in_loop(tokens, window, max_period)` asks whether the last `window` tokens repeat exactly with some period up to `max_period`; `seq_rep` at line 182 is the repeated-n-gram fraction.
[^arm]: `services/Model-Trainer/src/model_trainer/core/services/model/weight_space/lm_edit_arm.py:106` -- `run_lm_edit_arm` measures the control, applies the program, measures the edited model and removes the program; `paired_counts` at line 83 gives lost, gained and the exact McNemar p.
[^reasoning]: `services/Model-Trainer/results/weight-space/gpt2-v57/reader-reasoning.md:14` § "What I looked at, all weight products, with no forward pass" -- the reader's own account; its scripts are in `reader-scripts.md` beside it.
