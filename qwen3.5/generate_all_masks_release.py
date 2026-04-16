"""
generate_all_masks_release.py  —  CoE mask generation from histograms (release copy)

This is the GitHub release version of generate_all_masks.py with machine-specific
paths replaced by paths relative to this script's location.

Reads pre-computed histograms from histograms/final/ and writes two mask sets:
  Set A — 18 specialist masks  →  masks/specialist/coverage_{NAME}_K128.pt
  Set B — 8 combined masks     →  masks/combined/coverage_{NAME}_K128.pt

USAGE:
  conda activate <your-env>      # see environment.yml
  cd qwen3.5/
  python generate_all_masks_release.py

Pre-computed histograms are included in the repository at histograms/final/.
To regenerate histograms from scratch, run master_profiler_v44_release.py
(requires the BF16 base model, ~70 GB RAM).
"""

import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import torch
import numpy as np

# ── paths (relative to this script) ───────────────────────────────────────
_HERE    = os.path.dirname(os.path.abspath(__file__))
HIST_DIR = os.path.join(_HERE, "histograms", "final")
SPEC_DIR = os.path.join(_HERE, "masks", "specialist")
COMB_DIR = os.path.join(_HERE, "masks", "combined")
os.makedirs(SPEC_DIR, exist_ok=True)
os.makedirs(COMB_DIR, exist_ok=True)

# ── constants ──────────────────────────────────────────────────────────────
N_LAYERS = 40
K        = 128
RANK_W   = torch.tensor([(8 - k) / 36.0 for k in range(8)], dtype=torch.float64)

# ── helpers ────────────────────────────────────────────────────────────────
def load_hist(fn):
    path = os.path.join(HIST_DIR, fn + ".pt")
    return torch.load(path, map_location="cpu", weights_only=False).to(torch.float64)

def agg_util(fns, norm=False):
    """Aggregate util maps from a list of histogram file stems.
    norm=True: normalise each source to unit row-sum before summing (§30.2)."""
    total = None
    for fn in fns:
        H = load_hist(fn)
        u = (H * RANK_W.view(1, 1, 8)).sum(dim=2)   # [40, 256]
        if norm:
            row_sums = u.sum(dim=1, keepdim=True).clamp(min=1e-12)
            u = u / row_sums
        total = u if total is None else total + u
    return total

def top_k_mask(u, k=K):
    return [torch.topk(u[l].float(), k).indices for l in range(N_LAYERS)]

def coverage(u_q, mask):
    covs = []
    for l in range(N_LAYERS):
        ut = u_q[l]; s = ut.sum().item()
        covs.append(ut[mask[l]].sum().item() / s if s > 0 else 0.0)
    return float(np.mean(covs))

def jaccard(m1, m2):
    scores = []
    for l in range(N_LAYERS):
        a = set(m1[l].tolist()); b = set(m2[l].tolist())
        scores.append(len(a & b) / len(a | b))
    return float(np.mean(scores))

def save_mask(mask, path):
    torch.save(mask, path)

# ══════════════════════════════════════════════════════════════════════════
# PRE-FLIGHT: verify merge candidates before committing to Set B design
# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("MERGE VERIFICATION")
print("=" * 70)

checks = [
    ("ENG",      ["applied_engineering_textual_histogram"],
     "VOC",      ["vocational_trades_textual_histogram"]),
    ("HUM",      ["humanities_textual_histogram"],
     "ARCHEO-T", ["archeology_textual_histogram"]),
    ("BIO-T",    ["bio_chem_textual_histogram"],
     "EARTH-T",  ["earth_science_textual_histogram"]),
    ("SYSTEMS",  ["coding_systems_textual_histogram"],
     "OS",       ["coding_os_textual_histogram"]),
]

merge_ok = {}
for n1, f1, n2, f2 in checks:
    u1 = agg_util(f1); u2 = agg_util(f2)
    m1 = top_k_mask(u1); m2 = top_k_mask(u2)
    j = jaccard(m1, m2)
    verdict = "MERGE" if j >= 0.60 else "KEEP SEPARATE"
    merge_ok[f"{n1}+{n2}"] = j >= 0.60
    print(f"  {n1:10s} vs {n2:10s}  J={j:.3f}  → {verdict}")

print()

# ══════════════════════════════════════════════════════════════════════════
# SET A — SPECIALIST MASKS (one per histogram file)
# ══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("SET A — SPECIALIST MASKS  (K=128, single-source)")
print("=" * 70)

SPECIALIST_DEFS = {
    # name              : histogram stem
    "PHYS_T"   : "physics_textual_histogram",
    "PHYS_V"   : "physics_visual_histogram",
    "MATH_T"   : "math_textual_histogram",
    "MATH_V"   : "math_visual_histogram",
    "BIO_T"    : "bio_chem_textual_histogram",
    "BIO_V"    : "bio_chem_visual_histogram",
    "EARTH_T"  : "earth_science_textual_histogram",
    "EARTH_V"  : "earth_science_visual_histogram",
    "ARCHEO_T" : "archeology_textual_histogram",
    "ARCHEO_V" : "archeology_visual_histogram",
    "HUM"      : "humanities_textual_histogram",
    "ENG"      : "applied_engineering_textual_histogram",
    "VOC"      : "vocational_trades_textual_histogram",
    "SYSTEMS"  : "coding_systems_textual_histogram",
    "OS"       : "coding_os_textual_histogram",
    "SQL"      : "coding_sql_textual_histogram",
    "WEB_T"    : "coding_web_textual_histogram",
    "WEB_V"    : "coding_web_visual_histogram",
}

setA_util = {}   # keep util tensors for coverage reporting
setA_coverage = {}

print(f"  {'Name':12s}  {'Coverage':>10s}  {'Saved'}")
print(f"  {'-'*12}  {'-'*10}  ------")
for name, stem in SPECIALIST_DEFS.items():
    u = agg_util([stem])           # single source — no norm needed
    mask = top_k_mask(u)
    cov = coverage(u, mask)
    fname = f"coverage_{name}_K128.pt"
    save_mask(mask, os.path.join(SPEC_DIR, fname))
    setA_util[name] = u
    setA_coverage[name] = cov
    print(f"  {name:12s}  {cov*100:8.1f}%    {fname}")

print(f"\n  Total: {len(SPECIALIST_DEFS)} specialist masks saved to {SPEC_DIR}")

# ══════════════════════════════════════════════════════════════════════════
# SET B — COMBINED DOMAIN MASKS (normalised multi-source, K=128)
# ══════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("SET B — COMBINED MASKS  (K=128, per-domain normalised aggregation)")
print("=" * 70)

# ENG+VOC merge conditional on verified Jaccard
eng_voc_files = ["applied_engineering_textual_histogram",
                 "vocational_trades_textual_histogram"]
eng_voc_name  = "ENG_VOC"

COMBINED_DEFS = {
    # Slot 1: backend/systems coding (no visual histograms exist)
    "CODING"     : ["coding_systems_textual_histogram",
                    "coding_os_textual_histogram",
                    "coding_sql_textual_histogram"],

    # Slot 2: web/frontend — highest T/V mass overlap (0.399); visual is integral
    "WEB"        : ["coding_web_textual_histogram",
                    "coding_web_visual_histogram"],

    # Slot 3: humanities + archaeology textual (cultural/historical reasoning)
    "HUMANITIES" : ["humanities_textual_histogram",
                    "archeology_textual_histogram"],

    # Slot 4: mathematics — T+V justified by same analysis as PHYS (−4.8pp T cost)
    "MATH"       : ["math_textual_histogram",
                    "math_visual_histogram"],

    # Slot 5: physics — best T/V cohesion (J=0.458); T+V justified (−4.0pp)
    "PHYSICS"    : ["physics_textual_histogram",
                    "physics_visual_histogram"],

    # Slot 6: life & earth science textual (J=0.623 cross-domain)
    "LIFE_SCI"   : ["bio_chem_textual_histogram",
                    "earth_science_textual_histogram"],

    # Slot 7: applied & vocational engineering (textual-only domains)
    # — included regardless; low J still justifies single routing slot
    "ENG_VOC"    : ["applied_engineering_textual_histogram",
                    "vocational_trades_textual_histogram"],

    # Slot 8: visual science aggregate — cross-domain visual reasoning
    #         = all visual histograms that lack a dedicated combined slot above
    #         (ARCHEO-V included; BIO-V/EARTH-V/MATH-V/PHYS-V share visual experts
    #          better captured in their own slots, so include only the orphaned ones)
    "VISUAL_SCI" : ["math_visual_histogram",
                    "physics_visual_histogram",
                    "bio_chem_visual_histogram",
                    "earth_science_visual_histogram",
                    "archeology_visual_histogram",
                    "coding_web_visual_histogram"],
}

setB_util     = {}
setB_coverage = {}

print(f"  {'Name':12s}  {'Coverage':>10s}  Sources")
print(f"  {'-'*12}  {'-'*10}  -------")
for name, stems in COMBINED_DEFS.items():
    u = agg_util(stems, norm=True)   # per-domain normalisation
    mask = top_k_mask(u)
    cov = coverage(u, mask)
    fname = f"coverage_{name}_K128.pt"
    save_mask(mask, os.path.join(COMB_DIR, fname))
    setB_util[name] = u
    setB_coverage[name] = cov
    src = " + ".join(s.replace("_histogram", "").replace("_textual", "-T")
                      .replace("_visual", "-V").replace("coding_", "")
                      .replace("bio_chem", "BIO").replace("earth_science", "EARTH")
                      .replace("archeology", "ARCHEO").replace("vocational_trades", "VOC")
                      .replace("applied_engineering", "ENG").replace("humanities", "HUM")
                      .replace("physics", "PHYS").replace("math", "MATH") for s in stems)
    print(f"  {name:12s}  {cov*100:8.1f}%    {src}")

print(f"\n  Total: {len(COMBINED_DEFS)} combined masks saved to {COMB_DIR}")

# ══════════════════════════════════════════════════════════════════════════
# CROSS-SET COVERAGE: each Set A specialist queried under its Set B parent
# Measures coverage loss from combining
# ══════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("COVERAGE LOSS: specialist query under parent combined mask")
print("=" * 70)
print(f"  {'Specialist':12s}  {'→ Combined':12s}  {'Spec cov':>9s}  {'Comb cov':>9s}  {'Loss':>7s}")
print(f"  {'-'*12}  {'-'*12}  {'-'*9}  {'-'*9}  {'-'*7}")

PARENT_MAP = {
    "PHYS_T"  : "PHYSICS",    "PHYS_V"  : "PHYSICS",
    "MATH_T"  : "MATH",       "MATH_V"  : "MATH",
    "BIO_T"   : "LIFE_SCI",   "EARTH_T" : "LIFE_SCI",
    "BIO_V"   : "VISUAL_SCI", "EARTH_V" : "VISUAL_SCI",
    "ARCHEO_T": "HUMANITIES",  "ARCHEO_V": "VISUAL_SCI",
    "HUM"     : "HUMANITIES",
    "ENG"     : "ENG_VOC",    "VOC"     : "ENG_VOC",
    "SYSTEMS" : "CODING",     "OS"      : "CODING",     "SQL" : "CODING",
    "WEB_T"   : "WEB",        "WEB_V"   : "WEB",
}

for sp_name, comb_name in PARENT_MAP.items():
    u_q    = setA_util[sp_name]
    mask_sp   = top_k_mask(u_q)
    mask_comb = top_k_mask(setB_util[comb_name])
    cov_sp   = coverage(u_q, mask_sp)
    cov_comb = coverage(u_q, mask_comb)
    loss     = cov_sp - cov_comb
    flag = " !" if abs(loss) > 0.06 else ""
    print(f"  {sp_name:12s}  {comb_name:12s}  {cov_sp*100:7.1f}%   {cov_comb*100:7.1f}%   {loss*100:+5.1f}pp{flag}")

# ══════════════════════════════════════════════════════════════════════════
# INTER-SET B JACCARD: confirm routing discriminability
# ══════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("SET B INTER-MASK JACCARD (routing discriminability)")
print("Higher = more overlap = harder to discriminate at router")
print("=" * 70)

comb_names = list(COMBINED_DEFS.keys())
comb_masks = {n: top_k_mask(setB_util[n]) for n in comb_names}

print(f"  {'':12s}", end="")
for n in comb_names:
    print(f"  {n[:7]:>7s}", end="")
print()

for n1 in comb_names:
    print(f"  {n1:12s}", end="")
    for n2 in comb_names:
        if n1 == n2:
            print(f"  {'—':>7s}", end="")
        else:
            j = jaccard(comb_masks[n1], comb_masks[n2])
            print(f"  {j:7.3f}", end="")
    print()

# ── summary ────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Set A (specialist): {len(SPECIALIST_DEFS)} masks  →  {SPEC_DIR}")
print(f"  Set B (combined):   {len(COMBINED_DEFS)} masks  →  {COMB_DIR}")
print()
print("  Set B mean inter-mask Jaccard (routing separation):")
all_j = []
for i, n1 in enumerate(comb_names):
    for n2 in comb_names[i+1:]:
        all_j.append(jaccard(comb_masks[n1], comb_masks[n2]))
print(f"    mean={np.mean(all_j):.3f}  min={np.min(all_j):.3f}  max={np.max(all_j):.3f}")
print()
print("Done.")
