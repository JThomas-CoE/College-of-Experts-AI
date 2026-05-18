#!/usr/bin/env python3
"""
_analyze_transfer_correlations.py
===================================
For all 20 NEAR/FAR pairwise bench runs, assemble:
  - Y  = relative transfer deficit  (Acc_visitor_on_target - Self_target) / Self_target
  - X  = candidate similarity/difficulty predictors

Then rank predictors by Pearson |r| and run multiple regression.
"""

import json, math, pathlib
import numpy as np
from scipy import stats

GEMMA4  = pathlib.Path(__file__).parent
DATA    = GEMMA4 / "data"
MATRIX  = DATA / "matrix"

# ── §G39.3 Spc% self-baselines (visitor home-domain specialist performance) ──
SELF = {
    "math":        0.943,
    "physics":     0.840,
    "chemistry":   0.885,
    "engineering": 0.711,
    "cs":          0.811,
    "biology":     0.875,
    "economics":   0.846,
    "business":    0.839,
    "psychology":  0.790,
    "law":         0.589,
}

# §G39.3 Par% — parent model accuracy on each home domain (raw MMLU-Pro)
PAR = {
    "math":        0.941,
    "physics":     0.873,
    "chemistry":   0.881,
    "engineering": 0.726,
    "cs":          0.829,
    "biology":     0.899,
    "economics":   0.879,
    "business":    0.873,
    "psychology":  0.831,
    "law":         0.623,
}
# Normalized specialist strength = Spc / Par  (eliminates domain-difficulty confound)
NORM = {d: SELF[d] / PAR[d] for d in SELF}

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

# ── Load similarity data ──────────────────────────────────────────────────────

fsim    = json.load(open(DATA / "residual_similarity.json", encoding="utf-8"))
psim    = json.load(open(DATA / "param_similarity.json",    encoding="utf-8"))
F_mat   = fsim["matrix"]
P_mat   = psim["matrix"]
per_layer = psim["per_layer"]
N_LAYERS  = len(per_layer)

DOMAINS = psim["domains"]

# ── Load bench scores ─────────────────────────────────────────────────────────

def read_acc(visitor, target):
    fname = MATRIX / f"bench_merged_{visitor}_on_{target}_off5.jsonl"
    correct = total = 0
    for line in fname.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        total += 1
        if rec.get("correct") is True:
            correct += 1
    return correct / total, total

# ── Assemble rows ─────────────────────────────────────────────────────────────

rows = []
for visitor, target, arm in PAIRS:
    acc, n       = read_acc(visitor, target)
    self_target  = SELF[target]
    self_visitor = SELF[visitor]
    rel_gap      = (acc - self_target) / self_target   # Y — negative = deficit

    f_score = F_mat[visitor][target]

    agg_cos = P_mat[visitor][target]
    vals    = np.array([per_layer[l][visitor][target] for l in range(N_LAYERS)])
    valid   = vals[~np.isnan(vals)]
    mean_l  = valid.mean()
    std_l   = valid.std()
    min_l   = valid.min()
    max_l   = valid.max()
    rng_l   = max_l - min_l
    std_agg = std_l / abs(agg_cos) if agg_cos != 0 else np.nan

    norm_visitor = NORM[visitor]
    norm_target  = NORM[target]
    par_visitor  = PAR[visitor]
    par_target   = PAR[target]

    rows.append({
        "visitor":       visitor,
        "target":        target,
        "arm":           arm,
        "acc":           acc,
        "n":             n,
        "self_target":   self_target,
        "self_visitor":  self_visitor,
        "norm_visitor":  norm_visitor,   # Spc/Par — parent-normalised visitor strength
        "norm_target":   norm_target,    # Spc/Par — parent-normalised target strength
        "par_visitor":   par_visitor,    # parent raw accuracy on visitor domain
        "par_target":    par_target,     # parent raw accuracy on target domain
        "rel_gap":       rel_gap,
        "f_score":       f_score,
        "agg_cos":       agg_cos,
        "mean_l":        mean_l,
        "std_l":         std_l,
        "min_l":         min_l,
        "max_l":         max_l,
        "rng_l":         rng_l,
        "std_agg":       std_agg,
        # derived predictors
        "self_gap":      self_visitor - self_target,
        "self_ratio":    self_visitor / self_target,
        "norm_gap":      norm_visitor  - norm_target,
        "norm_ratio":    norm_visitor  / norm_target,
        "par_gap":       par_visitor   - par_target,    # parent domain difficulty gap
    })

Y = np.array([r["rel_gap"] for r in rows])

# ── Single-predictor correlations ─────────────────────────────────────────────

predictors = [
    ("F-score (binary overlap)",         "f_score"),
    ("Agg-cos (param budget-weighted)",  "agg_cos"),
    ("Mean-l (per-layer mean cos)",      "mean_l"),
    ("Min-l  (per-layer min cos)",       "min_l"),
    ("Max-l  (per-layer max cos)",       "max_l"),
    ("Std-l  (per-layer std cos)",       "std_l"),
    ("Range-l (max-min per layer)",      "rng_l"),
    ("Std/Agg (relative noise)",         "std_agg"),
    # raw Spc% — confounded with domain difficulty:
    ("Self_target raw (target Spc%)",    "self_target"),
    ("Self_visitor raw (visitor Spc%)",  "self_visitor"),
    ("Self_gap raw (Spc diff)",          "self_gap"),
    ("Self_ratio raw (Spc ratio)",       "self_ratio"),
    # parent-normalised — domain difficulty removed:
    ("Norm_visitor (Spc/Par visitor)",   "norm_visitor"),
    ("Norm_target  (Spc/Par target)",    "norm_target"),
    ("Norm_gap  (norm visitor-target)",  "norm_gap"),
    ("Par_gap (parent domain diff)",     "par_gap"),
]

print("=" * 75)
print("Single-predictor Pearson r  (Y = relative transfer deficit)\n")
print(f"{'Predictor':<40}  {'r':>7}  {'r²':>6}  {'p':>10}  {'slope':>9}")
print("-" * 75)

results = []
for label, key in predictors:
    X = np.array([r[key] for r in rows])
    slope, intercept, r, pval, se = stats.linregress(X, Y)
    results.append((abs(r), label, key, r, r**2, pval, slope))

results.sort(reverse=True)
for _, label, key, r, r2, pval, slope in results:
    star = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else ""))
    print(f"{label:<40}  {r:>+7.4f}  {r2:>6.4f}  {pval:>10.3e}  {slope:>+9.4f}  {star}")

# ── Detailed data table ───────────────────────────────────────────────────────

print("\n" + "=" * 110)
print("Full data table\n")
print(f"{'Pair':<28}  {'Arm':<4}  {'Acc':>6}  {'Self-T':>7}  {'RelGap':>8}  "
      f"{'F-score':>8}  {'Agg-cos':>8}  {'Mean-l':>8}  {'Std-l':>7}  {'Self-V':>7}")
print("-" * 110)
for r in rows:
    print(f"{r['visitor']+'->'+r['target']:<28}  {r['arm']:<4}  "
          f"{r['acc']*100:>5.1f}%  {r['self_target']*100:>6.1f}%  {r['rel_gap']*100:>+7.1f}%  "
          f"{r['f_score']:>8.4f}  {r['agg_cos']:>8.4f}  {r['mean_l']:>8.4f}  {r['std_l']:>7.4f}  "
          f"{r['self_visitor']*100:>6.1f}%")

# ── Multiple regression ───────────────────────────────────────────────────────

print("\n" + "=" * 75)
print("Multiple OLS  (n=20)\n")
X1 = np.array([r["f_score"]       for r in rows])
X2 = np.array([r["agg_cos"]       for r in rows])
X3 = np.array([r["self_target"]   for r in rows])
X4  = np.array([r["self_visitor"]  for r in rows])
X4n = np.array([r["norm_visitor"]  for r in rows])
X5  = np.array([r["max_l"]         for r in rows])

def ols_r2(Xmat, n=20):
    k = Xmat.shape[1] if Xmat.ndim > 1 else 1
    Xa = np.column_stack([np.ones(n), Xmat])
    coef, _, _, _ = np.linalg.lstsq(Xa, Y, rcond=None)
    Yhat = Xa @ coef
    ss_res = ((Y - Yhat)**2).sum()
    ss_tot = ((Y - Y.mean())**2).sum()
    r2 = 1 - ss_res / ss_tot
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    return r2, adj_r2, coef

for desc, Xmat in [
    ("Self_visitor raw only",              X4.reshape(-1, 1)),
    ("Norm_visitor (Spc/Par) only",        X4n.reshape(-1, 1)),
    ("F-score only",                       X1.reshape(-1, 1)),
    ("Agg-cos only",                       X2.reshape(-1, 1)),
    ("Max-l only",                         X5.reshape(-1, 1)),
    ("F + Agg-cos",                        np.column_stack([X1, X2])),
    ("F + Self_visitor raw",               np.column_stack([X1, X4])),
    ("F + Norm_visitor",                   np.column_stack([X1, X4n])),
    ("Agg-cos + Norm_visitor",             np.column_stack([X2, X4n])),
    ("Max-l + Norm_visitor",               np.column_stack([X5, X4n])),
    ("F + Agg + Norm_visitor",             np.column_stack([X1, X2, X4n])),
    ("F + Agg + Self_visitor raw",         np.column_stack([X1, X2, X4])),
]:
    r2, adj_r2, coef = ols_r2(Xmat)
    print(f"  {desc:<42}  R² = {r2:.4f}   adj-R² = {adj_r2:.4f}")

# ── Partial correlations controlling for Self_visitor ────────────────────────

print("\n" + "=" * 75)
print("Partial r (controlling for Norm_visitor = Spc/Par):  structural predictors only\n")

def partial_r(X_ctrl, X_pred, Y):
    """Partial correlation of X_pred ~ Y after removing variance explained by X_ctrl."""
    Xa = np.column_stack([np.ones(len(Y)), X_ctrl])
    def resid(v):
        coef, _, _, _ = np.linalg.lstsq(Xa, v, rcond=None)
        return v - Xa @ coef
    r_y  = resid(Y)
    r_xp = resid(X_pred)
    return stats.pearsonr(r_xp, r_y)

struct_preds = [
    ("F-score",       X1),
    ("Agg-cos",       X2),
    ("Mean-l",        np.array([r["mean_l"]   for r in rows])),
    ("Max-l",         X5),
    ("Min-l",         np.array([r["min_l"]    for r in rows])),
    ("Std-l",         np.array([r["std_l"]    for r in rows])),
    ("Std/Agg",       np.array([r["std_agg"]  for r in rows])),
    ("Par_gap",       np.array([r["par_gap"]  for r in rows])),
    ("Self_target",   X3),
]
print(f"  {'Predictor':<20}  {'ctrl=Self_V_raw':>16}  {'p':>8}    {'ctrl=Norm_V':>12}  {'p':>8}")
print("  " + "-" * 72)
for label, X in struct_preds:
    pr_raw,  pv_raw  = partial_r(X4,  X, Y)
    pr_norm, pv_norm = partial_r(X4n, X, Y)
    s1 = "***" if pv_raw  < 0.001 else ("**" if pv_raw  < 0.01 else ("*" if pv_raw  < 0.05 else ""))
    s2 = "***" if pv_norm < 0.001 else ("**" if pv_norm < 0.01 else ("*" if pv_norm < 0.05 else ""))
    print(f"  {label:<20}  {pr_raw:>+10.4f} ({pv_raw:>6.4f}) {s1:<3}   {pr_norm:>+10.4f} ({pv_norm:>6.4f}) {s2}")

# ── NEAR vs FAR group stats ───────────────────────────────────────────────────

print("\n" + "=" * 75)
print("NEAR vs FAR group means\n")
near_gaps = [r["rel_gap"]*100 for r in rows if r["arm"] == "NEAR"]
far_gaps  = [r["rel_gap"]*100 for r in rows if r["arm"] == "FAR"]
print(f"  NEAR: mean gap = {np.mean(near_gaps):+.2f}%  std = {np.std(near_gaps):.2f}%")
print(f"  FAR:  mean gap = {np.mean(far_gaps):+.2f}%  std = {np.std(far_gaps):.2f}%")
t, pval = stats.ttest_ind(near_gaps, far_gaps)
print(f"  t-test NEAR vs FAR:  t = {t:.3f},  p = {pval:.4f}")
