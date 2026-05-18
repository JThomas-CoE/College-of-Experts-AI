#!/usr/bin/env python3
"""
plot_transfer_efficiency_g39.py
================================
Y = visitor_acc_on_target / parent_acc_on_target
    "How much of the parent model's capability does the visitor recover?"
    1.0 = visitor matches parent; <1 = shortfall; >1 = visitor exceeds parent.

Two panels (same Y):
  LEFT  — X = F-score (binary residual expert-set overlap, Jaccard-style)
  RIGHT — X = OLS composite of parameter-space vector similarity factors:
              Agg-cos  +  Std/Agg  +  Norm_visitor(Spc/Par)
"""

import json, pathlib
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

GEMMA4   = pathlib.Path(__file__).parent
DATA     = GEMMA4 / "data"
MATRIX   = DATA / "matrix"
OUT_PNG  = GEMMA4 / "plot_transfer_efficiency_g39.png"

# ── Ground-truth tables ───────────────────────────────────────────────────────
SELF = {
    "math": 0.943, "physics": 0.840, "chemistry": 0.885,
    "engineering": 0.711, "cs": 0.811, "biology": 0.875,
    "economics": 0.846, "business": 0.839, "psychology": 0.790, "law": 0.589,
}
PAR = {
    "math": 0.941, "physics": 0.873, "chemistry": 0.881,
    "engineering": 0.726, "cs": 0.829, "biology": 0.899,
    "economics": 0.879, "business": 0.873, "psychology": 0.831, "law": 0.623,
}
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
fsim      = json.load(open(DATA / "residual_similarity.json", encoding="utf-8"))
psim      = json.load(open(DATA / "param_similarity.json",    encoding="utf-8"))
F_mat     = fsim["matrix"]
P_mat     = psim["matrix"]
per_layer = psim["per_layer"]
N_LAYERS  = len(per_layer)

def read_acc(visitor, target):
    fname = MATRIX / f"bench_merged_{visitor}_on_{target}_off5.jsonl"
    correct = total = 0
    for line in fname.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        rec = json.loads(line)
        total += 1
        if rec.get("correct") is True:
            correct += 1
    return correct / total, total

# ── Assemble data ─────────────────────────────────────────────────────────────
rows = []
for visitor, target, arm in PAIRS:
    acc, n    = read_acc(visitor, target)
    par_t     = PAR[target]
    eff       = acc / par_t                         # Y — transfer efficiency
    # ±1σ error on Y: propagate binomial SE through the division by par_t
    sigma_acc = (acc * (1 - acc) / n) ** 0.5
    sigma_eff = sigma_acc / par_t

    f_score   = F_mat[visitor][target]
    agg_cos   = P_mat[visitor][target]
    vals      = np.array([per_layer[l][visitor][target] for l in range(N_LAYERS)])
    valid     = vals[~np.isnan(vals)]
    std_l     = valid.std()
    std_agg   = std_l / abs(agg_cos) if agg_cos != 0 else np.nan
    max_l     = valid.max()
    norm_v    = NORM[visitor]

    rows.append(dict(
        visitor=visitor, target=target, arm=arm,
        acc=acc, par_t=par_t, eff=eff, sigma_eff=sigma_eff, n=n,
        f_score=f_score, agg_cos=agg_cos,
        std_l=std_l, std_agg=std_agg, max_l=max_l,
        norm_v=norm_v,
    ))

Y      = np.array([r["eff"]       for r in rows])
Yerr   = np.array([r["sigma_eff"] for r in rows])
Xf     = np.array([r["f_score"]   for r in rows])
Xagg   = np.array([r["agg_cos"] for r in rows])
Xs     = np.array([r["std_agg"] for r in rows])
Xn     = np.array([r["norm_v"]  for r in rows])

# ── OLS composite: Agg-cos + Std/Agg + Norm_visitor ──────────────────────────
Xmat = np.column_stack([np.ones(len(Y)), Xagg, Xs, Xn])
coef, _, _, _ = np.linalg.lstsq(Xmat, Y, rcond=None)
Y_hat  = Xmat @ coef
ss_res = ((Y - Y_hat)**2).sum()
ss_tot = ((Y - Y.mean())**2).sum()
R2_comp     = 1 - ss_res / ss_tot
adj_R2_comp = 1 - (1 - R2_comp) * (len(Y) - 1) / (len(Y) - 3 - 1)
X_comp = Y_hat

# F-score simple OLS (single predictor)
sl_f, ic_f, r_f, p_f, _ = stats.linregress(Xf, Y)
R2_f = r_f**2

r_comp, p_comp = stats.pearsonr(X_comp, Y)

print(f"Left  (F-score):          r={r_f:.4f}  R²={R2_f:.4f}  p={p_f:.3e}")
print(f"Right (Agg+Std/Agg+Norm): r={r_comp:.4f}  R²={R2_comp:.4f}  adj-R²={adj_R2_comp:.4f}  p={p_comp:.3e}")
print(f"Composite coef: intercept={coef[0]:.4f}  Agg-cos={coef[1]:.4f}  Std/Agg={coef[2]:.4f}  Norm_v={coef[3]:.4f}")

# ── Plot ──────────────────────────────────────────────────────────────────────
COLORS  = {"NEAR": "#1f77b4", "FAR": "#d62728"}
MARKERS = {"NEAR": "o", "FAR": "s"}

def scatter_panel(ax, X, rows, Y, Yerr, xlabel, r_val, r2_val, p_val,
                  model_label, slope=None, intercept=None):
    for r, xv, yv, ye in zip(rows, X, Y, Yerr):
        col = COLORS[r["arm"]]
        ax.errorbar(xv, yv, yerr=ye, fmt=MARKERS[r["arm"]], color=col,
                    ecolor=col, elinewidth=1.0, capsize=3, capthick=0.8,
                    ms=7, zorder=4, markeredgecolor="white", markeredgewidth=0.6)
        lbl = f"{r['visitor']}→{r['target']}"
        ox, oy = 0.005, 0.012
        if r["visitor"] == "chemistry" and r["target"] == "economics": oy = -0.026
        if r["visitor"] == "psychology" and r["target"] == "economics": ox = -0.004; oy = 0.014
        if r["visitor"] == "chemistry" and r["target"] == "law":        oy = -0.026
        if r["visitor"] == "math"      and r["target"] == "cs":         oy = -0.026
        ax.text(xv + ox, yv + oy, lbl, fontsize=6.5, color=col, va="bottom")

    xline = np.linspace(X.min() - 0.04, X.max() + 0.04, 200)
    if slope is not None:
        ax.plot(xline, slope * xline + intercept, color="#555555", lw=1.5, ls="--", zorder=3)
    else:
        ax.plot(xline, xline, color="#555555", lw=1.5, ls="--", zorder=3)

    ax.axhline(1.0, color="#2ca02c", lw=0.9, ls=":", alpha=0.7)
    ax.text(0.04, 0.95,
            f"$r$ = {r_val:.3f}   $R^{{2}}$ = {r2_val:.3f}\n"
            f"$p$ = {p_val:.2e}\n{model_label}",
            transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#aaaaaa", alpha=0.9))
    ax.set_xlabel(xlabel, fontsize=9)
    ax.grid(True, alpha=0.3)

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
fig.subplots_adjust(wspace=0.06)

ax_l.set_ylabel("Transfer Efficiency\n(visitor acc on target  /  parent acc on target)", fontsize=9)

scatter_panel(ax_l, Xf, rows, Y, Yerr,
              xlabel="F-score  (binary residual expert-set overlap)",
              r_val=r_f, r2_val=R2_f, p_val=p_f,
              model_label="F-score only",
              slope=sl_f, intercept=ic_f)
ax_l.set_title("Panel A — Binary Set Overlap (F-score)", fontsize=10, fontweight="bold")

scatter_panel(ax_r, X_comp, rows, Y, Yerr,
              xlabel="Expert Vector Similarity Index\n"
                     "(OLS composite: Agg-cos + Std/Agg + Norm$_v$)",
              r_val=r_comp, r2_val=R2_comp, p_val=p_comp,
              model_label=f"Agg-cos + Std/Agg + Norm$_v$\nadj-$R^{{2}}$={adj_R2_comp:.3f}",
              slope=None, intercept=None)
ax_r.set_title("Panel B — Parameter-Space Vector Similarity (OLS composite)", fontsize=10, fontweight="bold")

near_patch = mpatches.Patch(color=COLORS["NEAR"], label="NEAR domain pair  ●")
far_patch  = mpatches.Patch(color=COLORS["FAR"],  label="FAR domain pair  ■")
fig.legend(handles=[near_patch, far_patch], loc="lower center",
           ncol=2, fontsize=9, bbox_to_anchor=(0.5, 0.01))

fig.suptitle(
    "Cross-Domain Transfer Efficiency vs Expert-Space Similarity  "
    "|  Gemma4-26B-A4B  |  K=64 specialists  |  MMLU-Pro off5  |  n=20 pairs",
    fontsize=11, y=1.01)

fig.tight_layout()
fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
print(f"\nSaved → {OUT_PNG}")
