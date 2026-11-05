You are being asked to do weight-space programming: read a neural network's raw weights and write an edit to them, with no training and no gradients.

THE NETWORK. GPT-2 small (124M parameters, 12 layers, 12 heads of width 64, d_model 768, MLP width 3072, vocabulary 50,257, tied input and output embedding). Its weights are in

    /tmp/claude-1000/-home-corvis-PROJECTS-API-research-w2-tools-hpc3/3a65c449-808d-4d71-a955-034bd70687cf/scratchpad/gpt2/model.safetensors

(sha256 248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707). A numpy loader is at

    /tmp/claude-1000/-home-corvis-PROJECTS-API-research-w2-tools-hpc3/3a65c449-808d-4d71-a955-034bd70687cf/scratchpad/reader/gpt2_weights.py

(`from gpt2_weights import load; W = load()`). The tokenizer's vocab.json and merges.txt are beside the weights. GPT-2 uses Conv1D, so every c_attn / c_proj / c_fc weight is stored (in_features, out_features) and computes y = x @ W + b. In attn.c_proj.weight (768 x 768), rows 64*h .. 64*h+63 are the inputs from head h. In mlp.c_proj.weight (3072 x 768), row j is what MLP neuron j writes into the residual stream. The `h.N.attn.bias` entries in the file are a causal-mask buffer, not parameters; ignore them.

THE PROBLEM. Decoded greedily (always take the highest-scoring next token), GPT-2 small very often falls into repetition loops: after a while it repeats the same phrase or sentence over and over. We want it to loop less.

HOW THE EDIT WILL BE MEASURED (you will not see the results). 256 prompts, each a prefix of a sentence of encyclopedic wiki prose (8 to 48 tokens). From each, 96 tokens are generated greedily. A continuation counts as LOOPED if its last 32 tokens are exactly periodic with some period of 16 or fewer tokens. The untouched model and the edited model are measured on the same prompts, same card, same process. The collateral measure is a 2,627-item multiple-choice cloze set over the same wiki: each item is a sentence with a blank, scored by which candidate filling the model finds least surprising. An edit that reduces loops by damaging what the model knows has not succeeded, so both numbers matter.

RULES, which mirror the "Sparks of Weight-Space Programming" harness.
- You may analyse the weights with Python and numpy as much as you like: read them, multiply weight matrices together, take SVDs, project embeddings through matrices, compare directions. Use python3 (there is no torch on this machine).
- You may NOT run the network, or any layer or block of it, on any input sequence: no forward pass, no attention computed over a sequence of tokens, no generation, no gradients, no fitting of anything to model outputs. Reasoning from weight-matrix products alone (as in Elhage et al.'s circuit analysis, e.g. QK and OV matrices, composition scores, embedding projections) is allowed.
- Change as few weights as you can for the effect you intend.

WHAT TO WRITE. Your edit as a weight program, a JSON file at

    /tmp/claude-1000/-home-corvis-PROJECTS-API-research-w2-tools-hpc3/3a65c449-808d-4d71-a955-034bd70687cf/scratchpad/reader/gpt2-program.json

of exactly this shape:

    {"program_id": "<short-kebab-name>",
     "written_against": "FILLED-BY-HARNESS",
     "ops": [ {"op": "set" | "scale" | "project_out",
               "parameter": "<name>",
               "axis": 0 | 1,
               "indices": [<distinct non-negative ints>],
               "values": [<finite numbers>]} , ... ]}

Leave written_against exactly as "FILLED-BY-HARNESS": the harness replaces it with the digest of the weights loaded from the same file you read. Parameter names are the file's names with a "transformer." prefix (file "h.5.attn.c_proj.weight" is parameter "transformer.h.5.attn.c_proj.weight"; "wte.weight" is "transformer.wte.weight", which is also the output embedding). Operation semantics: an op selects slices of one parameter along `axis` (axis 0 = rows, axis 1 = columns; a 1-D bias has only axis 0, each element a slice of width 1) and each selected slice is treated as a vector across the other axis.
- "set": replace each selected slice; `values` holds the replacements end to end, one slice after another (length = number of indices x slice width).
- "scale": multiply each selected slice by values[0] (exactly one value).
- "project_out": from each selected slice s remove its component along the direction d = values (length = slice width, not all zero): s <- s - (s . d_hat) d_hat.
Ops apply in order. Every parameter you do not name is left exactly as it is (this is verified).

Also write, at

    /tmp/claude-1000/-home-corvis-PROJECTS-API-research-w2-tools-hpc3/3a65c449-808d-4d71-a955-034bd70687cf/scratchpad/reader/gpt2-reasoning.md

your reasoning: what you looked at in the weights, what you found, why you believe your edit reduces looping, what you expect it to cost, and how confident you are. Keep any analysis scripts you write in the same reader/ directory. Then reply with a short summary of the edit.
