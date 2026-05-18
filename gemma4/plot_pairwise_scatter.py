#!/usr/bin/env python3
"""
plot_pairwise_scatter.py
========================
Scatter plot: residual similarity F-score (X) vs accuracy transfer deficit
relative to home-domain specialist (Y), for all 20 near/far pairwise bench runs.

Y = Acc(visitor on target) - Self%(target)   [§G39.3 Spc% combined off1+off3+off4+off9]
X = F(visitor → target)  from residual_similarity.json
Error bars: ±1.96 * sqrt(p*(1-p)/n)   [95% CI on visitor accuracy only]
"""

import json
import math
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats

GEMMA4   = pathlib.Path(__file__).parent
DATA     = GEMMA4 / "data"
MATRIX   = DATA / "matrix"
SIM_JSON = DATA / "residual_similarity.json"
OUT_PNG  = DATA / "pairwise_scatter_g39.png"

# §G39.3 Spc% self-baselines (combined off1+off3+off8)
SELF = {
    "math": 0.943, "physics": 0.840, "chemistry": 0.885,
    "engineering": 0.711, "cs": 0.811, "biology": 0.875,
    "economics": 0.846, "business": 0.839, "psychology": 0.790,
    "law": 0.589,
}

# home specialist n per domain (off1+off3+off8 combined)
N_HOME = {
    "math": 405, "physics": 390, "chemistry": 340,
    "engineering": 291, "cs": 123, "biology": 215,
    "economics": 254, "business": 237, "psychology": 239,
    "law": 330,
}

# The 20 pairings: (visitor, target, label)
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

sim = json.load(open(SIM_JSON, encoding="utf-8"))
F_matrix = sim["matrix"]


def read_acc(visitor, target):
    fname = MATRIX / f"bench_merged_{visitor}_on_{target}_off5.jsonl"
    if not fname.exists():
        raise FileNotFoundError(fname)
    correct = total = 0
    for line in fname.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        total += 1
        if rec.get("correct") is True:
            correct += 1
    return correct, total


points = []
for visitor, target, arm in PAIRS:
    correct, n = read_acc(visitor, target)
    p = correct / n
    f = F_matrix[visitor][target]
    self_p = SELF[target]
    n_h = N_HOME[target]
    gap = (p - self_p) / self_p   # relative deficit (fraction of home score)
    # Error propagation for g = (p_v - p_h) / p_h
    # ∂g/∂p_v = 1/p_h,  ∂g/∂p_h = -p_v/p_h²
    se_v = math.sqrt(p * (1 - p) / n)
    se_h = math.sqrt(self_p * (1 - self_p) / n_h)
    se = math.sqrt((se_v / self_p)**2 + (se_h * p / self_p**2)**2)
    points.append({
        "visitor": visitor, "target": target, "arm": arm,
        "f": f, "p": p, "n": n, "self": self_p,
        "gap": gap, "se": se,
    })

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))

NEAR_COL = "#2563EB"   # blue
FAR_COL  = "#DC2626"   # red
OUTLINE  = "#1e293b"

for pt in points:
    col = NEAR_COL if pt["arm"] == "NEAR" else FAR_COL
    # marker size proportional to sqrt(n)
    ms = 4 * math.sqrt(pt["n"])
    ax.errorbar(
        pt["f"], pt["gap"] * 100,
        yerr=1.96 * pt["se"] * 100,
        fmt="o", color=col, ecolor=col, elinewidth=1.2,
        markersize=ms / 5, markeredgecolor=OUTLINE,
        markeredgewidth=0.6, capsize=3, zorder=3,
    )
    # label: visitor→target, offset to avoid overlap
    label = f"{pt['visitor']}→{pt['target']}"
    dx, dy = 0.004, 0.4
    # special nudges for crowded points
    nudges = {
        ("physics", "engineering"): (-0.06, -2.5),
        ("chemistry", "physics"):   (0.004,  1.5),
        ("physics", "chemistry"):   (0.004, -2.5),
        ("law", "cs"):              (-0.05,  1.5),
        ("math", "cs"):             (0.004,  1.2),
        ("psychology", "biology"):  (0.004,  1.2),
        ("psychology", "economics"):(-0.10, -2.5),
        ("chemistry", "economics"): (0.004, -2.5),
        ("biology", "psychology"):  (0.004,  1.2),
        ("engineering", "psychology"): (0.004, -2.5),
        ("economics", "law"):       (0.004,  1.5),
        ("chemistry", "law"):       (0.004, -2.5),
    }
    ndx, ndy = nudges.get((pt["visitor"], pt["target"]), (dx, dy))
    ax.annotate(
        label,
        xy=(pt["f"], pt["gap"] * 100),
        xytext=(pt["f"] + ndx, pt["gap"] * 100 + ndy),
        fontsize=6.5, color=OUTLINE,
        arrowprops=None,
    )

# y=0 reference line (visitor matches home specialist)
ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", zorder=1)

# OLS regression line (all 20 points)
xs = np.array([pt["f"] for pt in points])
ys = np.array([pt["gap"] * 100 for pt in points])
slope, intercept, r, pval, _ = stats.linregress(xs, ys)
xfit = np.linspace(xs.min() - 0.02, xs.max() + 0.02, 200)
ax.plot(xfit, slope * xfit + intercept, color="black", linewidth=1.0,
        linestyle="-", zorder=2, label=f"OLS  r={r:.2f}  p={pval:.2e}")

# Legend
near_patch = mpatches.Patch(color=NEAR_COL, label="NEAR visitor")
far_patch  = mpatches.Patch(color=FAR_COL,  label="FAR visitor")
ax.legend(handles=[near_patch, far_patch] + ax.get_lines(), fontsize=8,
          loc="upper left")

ax.set_xlabel("Residual expert overlap  F(visitor → target)", fontsize=11)
ax.set_ylabel("Transfer deficit  (visitor − self) / self  [%]", fontsize=11)
ax.set_title(
    "Cross-domain transfer deficit vs residual expert similarity\n"
    "Gemma4-26B-A4B specialists · MMLU-Pro off5 · 20 near/far pairs · §G39.8",
    fontsize=11,
)
ax.set_xlim(0.38, 0.90)
ax.grid(True, alpha=0.25, zorder=0)

# annotate marker-size legend
for n_ex, label_ex in [(41, "n=41"), (110, "n=110")]:
    ms_ex = 4 * math.sqrt(n_ex) / 5
    ax.scatter([], [], s=ms_ex**2, color="gray",
               label=label_ex, alpha=0.5)
ax.legend(fontsize=8, loc="upper left")

plt.tight_layout()
fig.savefig(OUT_PNG, dpi=150)
print(f"Saved: {OUT_PNG}")

# Print summary table
print("\n── All 20 points ─────────────────────────────────────────────────────")
print(f"{'Visitor→Target':<26}  {'Arm':<4}  {'F':>6}  {'Acc':>6}  {'Self':>6}  {'RelGap':>8}  {'n':>4}")
for pt in sorted(points, key=lambda x: x["f"], reverse=True):
    print(f"{pt['visitor']+'→'+pt['target']:<26}  {pt['arm']:<4}  "
          f"{pt['f']:>6.3f}  {pt['p']*100:>5.1f}%  {pt['self']*100:>5.1f}%  "
          f"{pt['gap']*100:>+7.1f}%   {pt['n']:>4}")

print(f"\nOLS (all 20):  slope={slope:.1f}%/unit  r={r:.3f}  p={pval:.2e}")
