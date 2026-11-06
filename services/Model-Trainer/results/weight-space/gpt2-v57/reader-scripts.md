# The GPT-2 reader's analysis scripts, verbatim

The fresh session that wrote induction-value-ablation-7heads.json kept these in its reader/ directory, as its brief asked. They are reproduced here unchanged, one fenced block each, so the claim that it read weight products and ran no forward pass can be checked line by line. Paths in them point at the scratch directory the session ran in. The fences are tagged text rather than python because ruff format, which this package's make lint runs, rewrites python-tagged code blocks in markdown, and a reformatted script would no longer be the one the session wrote.

## gpt2_weights.py

```text
"""Load GPT-2 small's raw weights from model.safetensors with numpy only.

Usage inside Python:
    from gpt2_weights import load
    W = load()            # dict: HF parameter name -> np.ndarray (float32)
    W["h.0.attn.c_attn.weight"].shape  -> (768, 2304)

Names here are the file's own (no "transformer." prefix). The measuring run
addresses the same tensors as "transformer.<name>"; e.g. the file's
"h.5.attn.c_proj.weight" is the run's "transformer.h.5.attn.c_proj.weight".
GPT-2 uses Conv1D: every ".weight" of c_attn, c_proj, c_fc is stored
(in_features, out_features), so y = x @ W + b.
"""

import json
import struct

import numpy as np

PATH = "/tmp/claude-1000/-home-corvis-PROJECTS-API-research-w2-tools-hpc3/3a65c449-808d-4d71-a955-034bd70687cf/scratchpad/gpt2/model.safetensors"


def load(path: str = PATH) -> dict:
    with open(path, "rb") as handle:
        header_len = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_len))
        base = 8 + header_len
        out = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            start, end = meta["data_offsets"]
            handle.seek(base + start)
            assert meta["dtype"] == "F32", meta["dtype"]
            out[name] = np.frombuffer(handle.read(end - start), dtype=np.float32).reshape(meta["shape"]).copy()
        return out
```

## _paths.py

```text
"""Under python3 -I neither the script dir nor user site-packages is on sys.path."""
import sys

READER = "/tmp/claude-1000/-home-corvis-PROJECTS-API-research-w2-tools-hpc3/3a65c449-808d-4d71-a955-034bd70687cf/scratchpad/reader"
SITE = "/home/corvis/.local/lib/python3.11/site-packages"
for p in (READER, SITE):
    if p not in sys.path:
        sys.path.append(p)
```

## heads.py

```text
"""Per-head weight-only analysis: OV copying score, positional prev-token score.

No forward pass: each quantity is a product of weight matrices applied to
embedding rows (token or position), with LayerNorm's centring/scaling and
gain folded in as a fixed per-vector normalisation.
"""
import sys
sys.path.append("/tmp/claude-1000/-home-corvis-PROJECTS-API-research-w2-tools-hpc3/3a65c449-808d-4d71-a955-034bd70687cf/scratchpad/reader")
import _paths  # noqa: F401,E402  (python3 -I drops user site and script dir)
import numpy as np
from gpt2_weights import load

W = load()
E = W["wte.weight"]; P = W["wpe.weight"]
D, H, dh = 768, 12, 64


def ln_dir(x, g, b):
    x = x - x.mean(-1, keepdims=True)
    x = x / np.sqrt((x**2).mean(-1, keepdims=True) + 1e-5)
    return x * g + b


def head_mats(l, h):
    Wa = W[f"h.{l}.attn.c_attn.weight"]; ba = W[f"h.{l}.attn.c_attn.bias"]
    Wq = Wa[:, h*dh:(h+1)*dh]; Wk = Wa[:, D+h*dh:D+(h+1)*dh]; Wv = Wa[:, 2*D+h*dh:2*D+(h+1)*dh]
    bq = ba[h*dh:(h+1)*dh]; bk = ba[D+h*dh:D+(h+1)*dh]; bv = ba[2*D+h*dh:2*D+(h+1)*dh]
    Wo = W[f"h.{l}.attn.c_proj.weight"][h*dh:(h+1)*dh, :]
    return Wq, Wk, Wv, Wo, bq, bk, bv


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    toks = rng.choice(np.arange(300, 40000), 800, replace=False)
    gf = W["ln_f.weight"]
    EU = E * gf  # logits = x_hat @ (E*gf).T up to the ln_f bias term
    rows = []
    for l in range(12):
        g1, b1 = W[f"h.{l}.ln_1.weight"], W[f"h.{l}.ln_1.bias"]
        X = ln_dir(E[toks] + P[50], g1, b1)
        Pn = ln_dir(P[:128] + E[toks].mean(0), g1, b1)
        for h in range(H):
            Wq, Wk, Wv, Wo, bq, bk, bv = head_mats(l, h)
            out = (X @ Wv) @ Wo
            out = out - out.mean(-1, keepdims=True)
            lg = out @ EU.T
            diag = lg[np.arange(len(toks)), toks]
            z = (diag - lg.mean(1)) / lg.std(1)
            rank = (lg > diag[:, None]).sum(1)
            top10 = (rank < 10).mean()
            q = Pn @ Wq + bq; k = Pn @ Wk + bk
            s = q @ k.T / 8.0
            mask = np.tril(np.ones((128, 128), bool))
            s = np.where(mask, s, -1e9)
            a = np.exp(s - s.max(1, keepdims=True)); a /= a.sum(1, keepdims=True)
            idx = np.arange(20, 128)
            prev = a[idx, idx-1].mean(); selfa = a[idx, idx].mean()
            rows.append([l, h, z.mean(), top10, prev, selfa])
            print(f"L{l:2d}H{h:2d} copyz={z.mean():6.2f} top10={top10:.2f} prevtok={prev:.2f} self={selfa:.2f}", flush=True)
    np.save("head_scores.npy", np.array(rows))
```

## compose.py

```text
"""Previous-token heads from positional QK, and K-composition (Elhage et al.)
of every later head with them. Weight products only."""
import sys
sys.path.append("/tmp/claude-1000/-home-corvis-PROJECTS-API-research-w2-tools-hpc3/3a65c449-808d-4d71-a955-034bd70687cf/scratchpad/reader")
import _paths  # noqa: F401,E402
import numpy as np  # noqa: E402
from heads import W, head_mats, ln_dir, D, H  # noqa: E402

P = W["wpe.weight"]

# 1. positional QK: score[i, j] = ln(P_i) Wq Wk^T ln(P_j); how much does
#    position i prefer i-1 over the mean of earlier positions (in nats/sqrt(d))?
prev = np.zeros((12, 12))
for l in range(12):
    g1, b1 = W[f"h.{l}.ln_1.weight"], W[f"h.{l}.ln_1.bias"]
    Pn = ln_dir(P[:200], g1, b1)
    for h in range(H):
        Wq, Wk, Wv, Wo, bq, bk, bv = head_mats(l, h)
        s = (Pn @ Wq + bq) @ (Pn @ Wk + bk).T / 8.0
        gaps = []
        for i in range(30, 200):
            row = s[i, :i + 1]
            gaps.append(row[i - 1] - np.delete(row, i - 1).max())
        prev[l, h] = np.mean(gaps)

print("positional prev-token margin (score[i,i-1] - max other), top 12:")
order = np.argsort(-prev.ravel())
for k in order[:12]:
    print(f"  L{k // 12}H{k % 12}: {prev.ravel()[k]:.2f}")


def ov(l, h):
    Wq, Wk, Wv, Wo, *_ = head_mats(l, h)
    return Wv @ Wo


def qk(l, h):
    Wq, Wk, *_ = head_mats(l, h)
    return Wq @ Wk.T


def fro(m):
    return float(np.sqrt((m * m).sum()))


# baseline: composition score for random matrices of the same rank
rng = np.random.default_rng(1)
base = []
for _ in range(20):
    A = rng.standard_normal((D, 64)) @ rng.standard_normal((64, D))
    B = rng.standard_normal((D, 64)) @ rng.standard_normal((64, D))
    base.append(fro(B @ A.T) / (fro(A) * fro(B)))
print("random-baseline composition:", np.mean(base))

prev_heads = [divmod(int(k), 12) for k in order[:6]]
print("using prev-token heads", prev_heads)
kc = np.zeros((12, 12))
OVs = {ph: ov(*ph) for ph in prev_heads}
for l in range(12):
    for h in range(H):
        Q = qk(l, h)
        best = 0.0
        for (pl, ph), A in OVs.items():
            if pl >= l:
                continue
            # key side reads A's output: x_q Q (x_k A)^T = x_q Q A^T x_k^T
            best = max(best, fro(Q @ A.T) / (fro(Q) * fro(A)))
        kc[l, h] = best
order2 = np.argsort(-kc.ravel())
print("K-composition with prev-token heads, top 20:")
for k in order2[:20]:
    print(f"  L{k // 12}H{k % 12}: {kc.ravel()[k]:.3f}")
np.save("prev_scores.npy", prev)
np.save("kcomp_scores.npy", kc)
```

## compose2.py

```text
"""K-composition of each layer-5..11 head with every head of layers 0..4
(full table), to find which upstream heads the induction candidates read
on the key side. Also a softer positional prev-token score."""
import sys
sys.path.append("/tmp/claude-1000/-home-corvis-PROJECTS-API-research-w2-tools-hpc3/3a65c449-808d-4d71-a955-034bd70687cf/scratchpad/reader")
import _paths  # noqa: F401,E402
import numpy as np  # noqa: E402
from heads import W, head_mats, ln_dir, H  # noqa: E402

P = W["wpe.weight"]; E = W["wte.weight"]


def fro(m):
    return float(np.sqrt((m * m).sum()))


OV = {}; QK = {}
for l in range(12):
    for h in range(H):
        Wq, Wk, Wv, Wo, *_ = head_mats(l, h)
        OV[(l, h)] = Wv @ Wo
        QK[(l, h)] = Wq @ Wk.T

# soft prev-token: residual = typical token + position, rel. preference for i-1
rng = np.random.default_rng(2)
toks = rng.choice(np.arange(300, 40000), 200, replace=False)
pt = np.zeros((5, 12))
for l in range(5):
    g1, b1 = W[f"h.{l}.ln_1.weight"], W[f"h.{l}.ln_1.bias"]
    for h in range(H):
        Wq, Wk, Wv, Wo, bq, bk, bv = head_mats(l, h)
        mass = []
        for i in range(40, 120, 8):
            X = ln_dir(E[toks[: i + 1]] + P[: i + 1], g1, b1)  # positions 0..i, distinct tokens
            q = X[i] @ Wq + bq
            k = X @ Wk + bk
            s = k @ q / 8.0
            a = np.exp(s - s.max()); a /= a.sum()
            mass.append(a[i - 1])
        pt[l, h] = np.mean(mass)
print("prev-token attention mass with random distinct tokens (QK only, single query):")
for k in np.argsort(-pt.ravel())[:10]:
    print(f"  L{k // 12}H{k % 12}: {pt.ravel()[k]:.3f}")

prevheads = [divmod(int(k), 12) for k in np.argsort(-pt.ravel())[:4]]
print("prev-token heads:", prevheads)
rows = []
for l in range(5, 12):
    for h in range(H):
        sc = {ph: fro(QK[(l, h)] @ OV[ph].T) / (fro(QK[(l, h)]) * fro(OV[ph])) for ph in prevheads if ph[0] < l}
        rows.append(((l, h), max(sc.values()), sc))
rows.sort(key=lambda r: -r[1])
for (l, h), m, sc in rows[:16]:
    print(f"  L{l}H{h}: max={m:.3f} " + " ".join(f"{a}.{b}={v:.3f}" for (a, b), v in sc.items()))
```

## induct.py

```text
"""Token-space induction matching test, weight products only.

For a later head B and a previous-token head A: the key at position j holds
A's write of token t_{j-1}; the query at position i holds token t_i.  B is an
induction head if its QK prefers keys whose previous token equals the query
token: M[t, s] = ln(E_t) Q_B (ln(E_s) OV_A)^T should have a dominant diagonal.
Reported: mean z-score of the diagonal within each row, and fraction of rows
where the diagonal is the row maximum.
"""
import sys
sys.path.append("/tmp/claude-1000/-home-corvis-PROJECTS-API-research-w2-tools-hpc3/3a65c449-808d-4d71-a955-034bd70687cf/scratchpad/reader")
import _paths  # noqa: F401,E402
import numpy as np  # noqa: E402
from heads import W, head_mats, ln_dir, H  # noqa: E402

E = W["wte.weight"]; P = W["wpe.weight"]
rng = np.random.default_rng(3)
toks = rng.choice(np.arange(300, 40000), 600, replace=False)

prevs = [(4, 11), (2, 2), (1, 1), (2, 9), (3, 3), (3, 7)]
res = {}
for (al, ah) in prevs:
    _, _, Wv, Wo, _, _, bv = head_mats(al, ah)
    gA, bA = W[f"h.{al}.ln_1.weight"], W[f"h.{al}.ln_1.bias"]
    Ak = (ln_dir(E[toks] + P[40], gA, bA) @ Wv) @ Wo  # what A writes about t_{j-1}
    for l in range(al + 1, 12):
        g1, b1 = W[f"h.{l}.ln_1.weight"], W[f"h.{l}.ln_1.bias"]
        Xq = ln_dir(E[toks] + P[41], g1, b1)
        for h in range(H):
            Wq, Wk, *_ = head_mats(l, h)
            # key side: A's write is a residual component; take it through ln_1's gain
            # (scale ~ unit-norm per dimension relative to residual: use the gain only)
            K = (Ak * g1) @ Wk
            M = (Xq @ Wq) @ K.T
            d = np.diag(M)
            z = (d - M.mean(1)) / M.std(1)
            top = (M.argmax(1) == np.arange(len(toks))).mean()
            key = (l, h)
            if key not in res or z.mean() > res[key][0]:
                res[key] = (z.mean(), top, (al, ah))
rows = sorted(res.items(), key=lambda kv: -kv[1][0])
for (l, h), (z, top, a) in rows[:25]:
    print(f"L{l}H{h}: diag z={z:.2f} argmax={top:.2f} via {a[0]}.{a[1]}")
```

## negheads.py

```text
"""Look for negative-copying (copy-suppression) heads in layers 8-11 with
several weight-only views:
  ov_z      : z of own-token logit in ln(E_t) OV (E*g_f)^T (as heads.py)
  ov_bottom : fraction of tokens whose own logit is in the bottom 1% of the row
  ov_eig    : sum(eig)/sum|eig| of the 768x768-reduced token OV circuit
              (Elhage copying score; negative = suppressing)
  qk_z      : z of diagonal of (E*g_f)_t (query reads the UNEMBED direction of
              a predicted token) Q K^T ln(E_s)^T  (key reads token embedding)
"""
import sys
sys.path.append("/tmp/claude-1000/-home-corvis-PROJECTS-API-research-w2-tools-hpc3/3a65c449-808d-4d71-a955-034bd70687cf/scratchpad/reader")
import _paths  # noqa: F401,E402
import numpy as np  # noqa: E402
from heads import W, head_mats, ln_dir, H  # noqa: E402

E = W["wte.weight"]; P = W["wpe.weight"]; gf = W["ln_f.weight"]
EU = E * gf
rng = np.random.default_rng(4)
toks = rng.choice(np.arange(300, 40000), 600, replace=False)
Ec = E[toks] - E.mean(0)
for l in range(8, 12):
    g1, b1 = W[f"h.{l}.ln_1.weight"], W[f"h.{l}.ln_1.bias"]
    X = ln_dir(E[toks] + P[50], g1, b1)
    U = EU[toks] - EU.mean(0)
    Uq = U / np.linalg.norm(U, axis=1, keepdims=True) * np.sqrt(768) * g1
    for h in range(H):
        Wq, Wk, Wv, Wo, bq, bk, bv = head_mats(l, h)
        lg = ((X @ Wv) @ Wo) @ EU.T
        lg -= lg.mean(1, keepdims=True)
        d = lg[np.arange(len(toks)), toks]
        z = (d / lg.std(1)).mean()
        bottom = ((lg < d[:, None]).sum(1) < 0.01 * lg.shape[1]).mean()
        C = (Ec * g1) @ Wv @ Wo @ (E[toks] * gf).T  # token x token block
        ev = np.linalg.eigvals(C)
        eig = ev.real.sum() / np.abs(ev).sum()
        S = (Uq @ Wq) @ (X @ Wk).T
        sd = np.diag(S)
        qz = ((sd - S.mean(1)) / S.std(1)).mean()
        print(f"L{l}H{h}: ov_z={z:6.2f} ov_bottom={bottom:.2f} ov_eig={eig:6.2f} qk_z={qz:6.2f}", flush=True)
```

## qcomp.py

```text
"""Where do the induction heads' outputs go? Q- and V-composition of every
layer 8-11 head with the union of the induction heads' OV, relative to the
same score against the other heads of layers 5-7 (weight products only)."""
import sys
sys.path.append("/tmp/claude-1000/-home-corvis-PROJECTS-API-research-w2-tools-hpc3/3a65c449-808d-4d71-a955-034bd70687cf/scratchpad/reader")
import _paths  # noqa: F401,E402
import numpy as np  # noqa: E402
from heads import W, head_mats, H  # noqa: E402

IND = [(5, 1), (5, 5), (6, 9), (7, 2), (7, 10), (5, 0), (5, 8)]


def fro(m):
    return float(np.sqrt((m * m).sum()))


OV = {}
for l in range(5, 8):
    for h in range(H):
        _, _, Wv, Wo, *_ = head_mats(l, h)
        OV[(l, h)] = Wv @ Wo
others = [k for k in OV if k not in IND]
print("head      Qcomp(ind) Qcomp(other)  Vcomp(ind) Vcomp(other)")
for l in range(8, 12):
    for h in range(H):
        Wq, Wk, Wv, Wo, *_ = head_mats(l, h)
        QK = Wq @ Wk.T; VO = Wv @ Wo
        qi = np.mean([fro(OV[k] @ QK) / (fro(OV[k]) * fro(QK)) for k in IND])
        qo = np.mean([fro(OV[k] @ QK) / (fro(OV[k]) * fro(QK)) for k in others])
        vi = np.mean([fro(OV[k] @ VO) / (fro(OV[k]) * fro(VO)) for k in IND])
        vo = np.mean([fro(OV[k] @ VO) / (fro(OV[k]) * fro(VO)) for k in others])
        print(f"L{l}H{h:<2d}    {qi:.3f}      {qo:.3f}        {vi:.3f}      {vo:.3f}  ratioQ={qi / qo:.2f}")
```

## write_program.py

```text
"""Write gpt2-program.json and check it applies cleanly to the loaded weights
(the check edits arrays only; nothing is evaluated on any input)."""
import json
import sys
sys.path.append("/tmp/claude-1000/-home-corvis-PROJECTS-API-research-w2-tools-hpc3/3a65c449-808d-4d71-a955-034bd70687cf/scratchpad/reader")
import _paths  # noqa: F401,E402
import numpy as np  # noqa: E402
from gpt2_weights import load  # noqa: E402

OUT = "/tmp/claude-1000/-home-corvis-PROJECTS-API-research-w2-tools-hpc3/3a65c449-808d-4d71-a955-034bd70687cf/scratchpad/reader/gpt2-program.json"
INDUCTION = {5: [0, 1, 5, 8], 6: [9], 7: [2, 10]}
ops = []
for layer, heads in sorted(INDUCTION.items()):
    cols = [1536 + 64 * h + i for h in heads for i in range(64)]
    ops.append({"op": "scale", "parameter": f"transformer.h.{layer}.attn.c_attn.weight",
                "axis": 1, "indices": cols, "values": [0.0]})
program = {"program_id": "induction-value-ablation-7heads",
           "written_against": "FILLED-BY-HARNESS", "ops": ops}
with open(OUT, "w") as fh:
    json.dump(program, fh, indent=1)

W = load()
before = {k: v.copy() for k, v in W.items()}
changed = 0
for op in program["ops"]:
    name = op["parameter"].removeprefix("transformer.")
    arr = W[name]
    assert len(set(op["indices"])) == len(op["indices"])
    sl = arr[:, op["indices"]] if op["axis"] == 1 else arr[op["indices"]]
    changed += int((sl != 0).sum())
    if op["axis"] == 1:
        arr[:, op["indices"]] *= op["values"][0]
    else:
        arr[op["indices"]] *= op["values"][0]
diff = [k for k in W if not np.array_equal(W[k], before[k])]
print("params changed:", diff)
print("weights changed:", changed)
for layer, heads in INDUCTION.items():
    Wa = W[f"h.{layer}.attn.c_attn.weight"]
    for h in range(12):
        v = np.abs(Wa[:, 1536 + 64 * h:1536 + 64 * (h + 1)]).sum()
        q = np.abs(Wa[:, 64 * h:64 * (h + 1)]).sum()
        assert (v == 0) == (h in heads), (layer, h)
        assert q > 0
print("ok: only the V columns of the listed heads are zero; Q, K, biases intact")
```
