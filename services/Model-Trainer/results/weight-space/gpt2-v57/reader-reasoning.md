# Reasoning: induction-value-ablation-7heads

## The edit

Scale the value-projection columns of seven attention heads to zero:
`transformer.h.{5,6,7}.attn.c_attn.weight`, axis 1, columns `1536 + 64h .. 1536 + 64h + 63`
for heads 5.0, 5.1, 5.5, 5.8, 6.9, 7.2 and 7.10. That is 3 ops, 448 columns, 344,064 weights.
The value bias `b_V`, the Q and K weights, `c_proj` and everything else stay as they were.
Each head still forms its attention pattern, but what it writes no longer depends on what it
attends to. Its output becomes the constant `b_V @ W_O` (attention weights sum to 1). That
constant is the part of the head's output that does not depend on the input, so keeping it is
the nearest weight-only stand-in for mean-ablation.

## What I looked at, all weight products, with no forward pass

Scripts are in this directory: `heads.py`, `compose.py`, `compose2.py`, `induct.py`,
`negheads.py`, `qcomp.py` and `write_program.py`. LayerNorm was folded in as a fixed
centre/scale plus gain on embedding rows. Nothing was run on a token sequence.

1. **Previous-token head.** Positional QK alone did not find it cleanly: after layer 2 the
   positional signal has been diluted. So I identified it by composition instead. For every
   candidate pair (A in layers 1-4, B later), I built the token-space matching matrix
   `M[t,s] = ln(E_t) Q_B K_B^T (ln(E_s) OV_A g)^T`. This asks whether B's query, holding token t,
   prefers a key whose previous token (as written by A) is t. **A = 4.11** gave by far the
   strongest diagonal for every later head, which fits 4.11 being GPT-2 small's previous-token head.
2. **Induction heads (K-composition with 4.11).** These heads had the largest diagonal z-scores
   (and the share of rows where the diagonal is the row maximum):
   5.1 (3.23, 0.58), 6.9 (3.07, 0.43), 5.5 (2.94, 0.34), 7.2 (2.69, 0.39), 7.10 (2.56, 0.34),
   5.0 (2.48, 0.28) and 5.8 (2.46, 0.27). Next came a gap: 9.9 (2.14, 0.20), 7.11 (1.93), 8.1 (1.86).
   The Frobenius K-composition scores of Elhage et al. rank the same set at the top. They are
   about 2.5x a random-matrix baseline of 0.036. The set matches the induction heads reported
   for GPT-2 small in the literature (5.1, 5.5, 6.9, 7.2, 7.10, plus the weaker 5.0 and 5.8).
3. **Where their output goes.** I computed Q-composition of every layer 8-11 head with the
   induction heads' OV, relative to the other heads of layers 5-7. The two heads whose queries
   read the induction heads most, relative to other heads, are **10.7 and 11.10**, the
   copy-suppression and negative-name-mover heads. So induction output feeds the downstream
   prediction and copying machinery directly.
4. **OV copying in raw embedding space.** I also measured, for each head, where the own-token
   logit ranks in `ln(E) OV (E g_f)^T`. The strong raw-space copiers are mainly in layers 7-11
   (for example 11.3, 11.11, 10.11, 9.3 and 8.7). The induction heads themselves copy only weakly
   in raw space (z of about 0.7-1.3). Their write therefore goes into an intermediate
   representation that later layers turn into logits. Raw-embedding OV is a known-poor proxy
   here: it shows 10.7 as neutral and 11.10 as positive, against the activation-based
   literature. So I did not base the edit on OV signs. I avoided computing MLP0's "effective
   embedding" because that means running a block on token inputs.

## Why this should reduce looping

Greedy loops in GPT-2 are the induction circuit feeding on itself. Once a phrase has occurred
twice, every next token of the third copy is predicted by "find the earlier occurrence of my
current token, copy what followed". That is exactly the 4.11 -> {5.x, 6.9, 7.x} circuit above.
Every repetition adds another matching earlier position, so the prediction grows more confident,
which is the self-reinforcement effect behind loops. Cutting the value path of the heads that
match on the previous-token key removes the copy signal at its root, ahead of the late copying
heads and MLPs that would otherwise amplify it.

## What I expect it to cost

- In-context copying (repeating a name or rare token seen earlier) gets weaker.
- Loss on long documents with repeated entities goes up.
- The cloze items are single sentences, and the candidates are judged mostly on local and world
  knowledge stored in the MLPs and embeddings. None of those are touched. I expect a small cloze
  drop, perhaps 0-2 points, mainly on items where the answer repeats a token from earlier in the
  same sentence.
- Heads 5.5 and 6.9 also feed the IOI-type S-inhibition path, so name disambiguation inside a
  sentence may degrade slightly.

## Risks and confidence

- **Weaker Markov loops may replace the strong ones.** With copying removed, greedy decoding
  depends more on local context, and a mostly local predictor can still settle into short cycles
  (", and the ... , and the"). Backup heads (9.9, 8.1, 7.11) keep some induction ability.
- **Bias only approximates the mean.** Zeroing W_V while keeping b_V approximates a
  mean-ablation but is not one. Any position-0 or attention-sink value that later layers expect
  as a constant is gone, which shifts their inputs slightly.

My confidence that the loop rate drops is moderate, about 0.6, and that it drops substantially
(by a third or more of the baseline loop rate) is about 0.4. My confidence that the cloze cost
stays small (under about 2 points) is about 0.7.
