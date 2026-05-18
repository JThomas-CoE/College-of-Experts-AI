#!/usr/bin/env python3
"""
compute_param_similarity.py
============================
Parameter-space cosine similarity via Gram matrix trick.

Design: all tensors are pre-loaded into RAM in the main process, then
N_WORKERS threads compute Gram matrices concurrently.  numpy BLAS releases
the GIL during matmul, so threads run truly in parallel.  OMP_NUM_THREADS=1
ensures each thread's BLAS uses 1 core â†’ N_WORKERS cores total with no
paging-file pressure (no per-worker file mmap).
"""

import os
os.environ['OMP_NUM_THREADS']      = '1'
os.environ['MKL_NUM_THREADS']      = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import json, pathlib, time
import numpy as np

GEMMA4    = pathlib.Path(__file__).parent
MODEL_DIR = GEMMA4 / "gemma-4-26B-A4B-it"
DATA_DIR  = GEMMA4 / "data"
OUT_JSON  = DATA_DIR / "param_similarity.json"
F_JSON    = DATA_DIR / "residual_similarity.json"

DOMAINS = [
    "math", "physics", "engineering", "cs", "chemistry",
    "biology", "economics", "business", "psychology", "law",
]
K         = 64
N_LAYERS  = 30
N_EXPERTS = 128
N_WORKERS = 30


# â”€â”€ Worker â€” runs in a thread (shares W_all and IND_np via closure) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# numpy BLAS releases the GIL so N_WORKERS threads run truly in parallel.
# OMP_NUM_THREADS=1 ensures each thread's matmul uses exactly 1 core,
# giving N_WORKERS cores total.

def make_compute(W_all, IND_np, budgets):
    def compute_layer(l):
        W      = W_all[l]          # (128, flat) float32 numpy array â€” shared, read-only
        budget = budgets[l]
        n_dom  = IND_np[0].shape[0]

        if budget == 0:
            nan = np.full((n_dom, n_dom), np.nan)
            return l, nan, np.zeros((n_dom, n_dom)), np.zeros(n_dom)

        G  = W @ W.T               # (128,128) â€” GIL released by numpy BLAS
        I  = IND_np[l]             # (n_dom, 128)
        IG = I @ G                 # (n_dom, 128)

        raw_dot = IG @ I.T         # (n_dom, n_dom)
        diag    = (IG * I).sum(axis=1)

        norms_l = np.sqrt(diag)
        with np.errstate(invalid='ignore'):
            cos_l = raw_dot / (norms_l[:, None] * norms_l[None, :])
        cos_l = np.clip(cos_l, -1.0, 1.0)

        return (l,
                cos_l.astype(np.float64),
                (budget * raw_dot).astype(np.float64),
                (budget * diag).astype(np.float64))
    return compute_layer


if __name__ == "__main__":
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from safetensors import safe_open

    def load_mask(domain: str) -> list[set]:
        path = GEMMA4 / f"mask_coe_{domain}_mmlu_swapactive_K64.json"
        raw = json.load(open(path))
        return [set(map(int, raw[str(ll)])) for ll in range(N_LAYERS)]

    print("Loading masksâ€¦")
    masks     = {d: load_mask(d) for d in DOMAINS}
    backbone  = [set.intersection(*[masks[d][l] for d in DOMAINS]) for l in range(N_LAYERS)]
    residuals = {d: [masks[d][l] - backbone[l] for l in range(N_LAYERS)] for d in DOMAINS}
    budgets   = [K - len(backbone[l]) for l in range(N_LAYERS)]

    def make_ind(res_l):
        v = np.zeros(N_EXPERTS, dtype=np.float32)
        for e in res_l:
            v[e] = 1.0
        return v

    IND_np = [
        np.stack([make_ind(residuals[d][l]) for d in DOMAINS])  # (n_dom, 128)
        for l in range(N_LAYERS)
    ]

    idx  = json.load(open(MODEL_DIR / "model.safetensors.index.json"))
    wmap = idx["weight_map"]

    shard_paths = sorted(set(
        MODEL_DIR / wmap[f"model.language_model.layers.{l}.experts.down_proj"]
        for l in range(N_LAYERS)
    ))
    print(f"Opening shards: {[p.name for p in shard_paths]}")
    sf = {p: safe_open(str(p), framework="pt", device="cpu") for p in shard_paths}

    print("Pre-loading all layer down_proj tensors into RAMâ€¦")
    t0 = time.time()
    W_all = []
    for l in range(N_LAYERS):
        key   = f"model.language_model.layers.{l}.experts.down_proj"
        shard = sf[MODEL_DIR / wmap[key]]
        import torch
        W = shard.get_tensor(key).to(torch.float32).numpy()
        W_all.append(W.reshape(W.shape[0], -1))            # (128, 2816*704)
    del sf
    print(f"  loaded {N_LAYERS} layers in {time.time()-t0:.1f}s")

    # â”€â”€ Parallel Gram matrix computation via thread pool â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    n_dom   = len(DOMAINS)
    dot_AB  = np.zeros((n_dom, n_dom), dtype=np.float64)
    norm_sq = np.zeros(n_dom, dtype=np.float64)
    per_layer_cos = [None] * N_LAYERS

    worker = make_compute(W_all, IND_np, budgets)

    print(f"\nComputing Gram matrices â€” {N_WORKERS} threads, OMP_NUM_THREADS=1â€¦")
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        futs = {pool.submit(worker, l): l for l in range(N_LAYERS)}
        done = 0
        for fut in as_completed(futs):
            l, cos_l, wdot, wdiag = fut.result()
            per_layer_cos[l] = cos_l
            dot_AB  += wdot
            norm_sq += wdiag
            done += 1
            print(f"  layer {l+1:2d} done  [{done:2d}/{N_LAYERS}]  {time.time()-t0:.1f}s",
                  flush=True)

    print(f"\nAll layers done in {time.time()-t0:.1f}s")

    # â”€â”€ Global aggregated cosine â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    norms   = np.sqrt(norm_sq)
    cos_mat = dot_AB / (norms[:, None] * norms[None, :])
    cos_mat = np.clip(cos_mat, -1.0, 1.0)

    print("\nâ”€â”€ Parameter-space cosine similarity (budget-weighted aggregate) â”€â”€")
    print(f"{'':>15}" + "".join(f"{d:>12}" for d in DOMAINS))
    for i, da in enumerate(DOMAINS):
        print(f"{da:>15}" + "".join(f"{cos_mat[i,j]:12.4f}" for j in range(n_dom)))

    # â”€â”€ Per-layer statistics for the 20 bench pairs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    PAIRS = [
        ("physics",     "math",        "NEAR"),
        ("law",         "math",        "FAR"),
        ("chemistry",   "physics",     "NEAR"),
        ("law",         "physics",     "FAR"),
        ("physics",     "engineering", "NEAR"),
        ("law",         "engineering", "FAR"),
        ("math",        "cs",          "NEAR"),
        ("law",         "cs",          "FAR"),
        ("physics",     "chemistry",   "NEAR"),
        ("law",         "chemistry",   "FAR"),
        ("psychology",  "biology",     "NEAR"),
        ("business",    "biology",     "FAR"),
        ("psychology",  "economics",   "NEAR"),
        ("chemistry",   "economics",   "FAR"),
        ("math",        "business",    "NEAR"),
        ("law",         "business",    "FAR"),
        ("biology",     "psychology",  "NEAR"),
        ("engineering", "psychology",  "FAR"),
        ("economics",   "law",         "NEAR"),
        ("chemistry",   "law",         "FAR"),
    ]

    f_data = json.load(open(F_JSON))
    f_mat  = f_data["matrix"]

    print("\nâ”€â”€ Comparison: F-score vs param-space cosine (20 bench pairs) â”€â”€")
    print(f"{'Pair':<28}  {'Arm':<4}  {'F-score':>8}  {'Agg-cos':>8}  "
          f"{'Mean-l':>8}  {'Std-l':>7}  {'Min-l':>7}  {'Max-l':>7}")
    for visitor, target, arm in PAIRS:
        i  = DOMAINS.index(visitor)
        j  = DOMAINS.index(target)
        f  = f_mat[visitor][target]
        pc = cos_mat[i, j]
        vals = np.array([per_layer_cos[l][i, j]
                         for l in range(N_LAYERS)
                         if not np.isnan(per_layer_cos[l][i, j])])
        print(f"{visitor+'â†’'+target:<28}  {arm:<4}  {f:8.4f}  {pc:8.4f}  "
              f"{vals.mean():8.4f}  {vals.std():7.4f}  "
              f"{vals.min():7.4f}  {vals.max():7.4f}")

    # â”€â”€ Save â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    out = {
        "domains":   DOMAINS,
        "budgets":   budgets,
        "matrix":    {da: {db: float(cos_mat[i, j])
                           for j, db in enumerate(DOMAINS)}
                      for i, da in enumerate(DOMAINS)},
        "per_layer": [
            {da: {db: float(per_layer_cos[l][i, j])
                  for j, db in enumerate(DOMAINS)}
             for i, da in enumerate(DOMAINS)}
            for l in range(N_LAYERS)
        ],
        "method":   "down_proj Gram matrix, residual indicator, budget-weighted cosine, thread-parallel",
        "n_layers": N_LAYERS,
        "K":        K,
    }
    with open(OUT_JSON, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nSaved: {OUT_JSON}")

def compute_layer(args):
    """
