#!/usr/bin/env python3
"""
build_coe_mask.py — Three-pass CoE expert mask builder.

Pass 1: Domain baseline      — WL-union seed + util fill (or plain top-K if no --wl-sources).
                               WL union: expert is guaranteed if avg_rank <= --wl-thresh in ANY
                               of the --wl-sources FILE::KEY pairs (per-subdomain union).
                               Remaining budget filled by blended util rank descending.
Pass 2: CoT injection        — inject reasoning experts by evicting low-tier generalists.
                               Guard (ordinal, both from Pass 1 snapshot):
                                 cot_ordinal < dom_ordinal           → swap
                                 cot_ordinal == dom_ordinal          → swap if avg_rank_cot < avg_rank_dom else STOP
                                 cot_ordinal > dom_ordinal           → STOP
                               Tier 4 experts (avg_rank < Q1, near-primary) exempt from eviction.
                               Cap = --pass2-cap per layer.  Unused budget does NOT roll over.
Pass 3: Structural injection — force KB-processing experts (highest delta_matched first)
                               into mask where missing.  Eviction pool: all non-structural
                               in M[l], tier-stratified, tier 4 eligible.  Cap = --pass3-cap.

Usage examples:
  python build_coe_mask.py --domain medical --budget 64
  python build_coe_mask.py --domain medical --no-pass2 --no-pass3
  python build_coe_mask.py --domain pharma  --cot-domain cot_combined --pass2-cap 6
  python build_coe_mask.py --domain-weights coding_python:0.35,coding_cpp:0.25 \\
      --output-name coding --probe-json histograms/final/probe_sets_coding.json
  python build_coe_mask.py \\
      --pt-file histograms/final/physics_base_results.pt \\
      --domain-weights physics:1.0,physics_hydrated:1.0 \\
      --cot-pt-file histograms/final/medical_legal_pharma_results.pt \\
      --wl-sources "histograms/final/physics_base_results.pt::physics,histograms/final/physics_base_results.pt::physics_hydrated,histograms/physics_thinking_results.pt::physics_qa,histograms/physics_thinking_results.pt::physics_th" \\
      --output-name physics --budget 64
"""

import argparse
import json
import os
import sys
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
PT_FILE       = os.path.join(SCRIPT_DIR, "histograms", "final",
                              "medical_legal_pharma_results.pt")
ABLATION_JSON = os.path.join(SCRIPT_DIR, "kb_ablation_results.json")
OUTPUT_DIR    = SCRIPT_DIR   # masks written alongside existing mask_*.json files

# ---------------------------------------------------------------------------
# Legacy domain key aliases (short name -> full PT key)
# Unknown keys are passed through verbatim.
# ---------------------------------------------------------------------------
DOMAIN_ALIAS = {
    "medical":  "medical_clinical",
    "legal":    "legal_professional",
    "pharma":   "pharmacology",
}

# ---------------------------------------------------------------------------
# Util computation
# ---------------------------------------------------------------------------

def compute_util(domain_result):
    """Return util (L, E) = sum_k(freq * weight_mean)."""
    freq        = domain_result["freq"]        # (L, E, K) float32
    weight_mean = domain_result["weight_mean"] # (L, E, K) float32
    return (freq * weight_mean).sum(dim=-1)    # (L, E)


def available_keys(results):
    return [k for k in results if not k.startswith("_")]


def load_util(results, key):
    if key not in results or results[key] is None:
        sys.exit(f"[ERROR] Domain key '{key}' missing or None in PT file.\n"
                 f"       Available keys: {available_keys(results)}")
    return compute_util(results[key])


def load_util_weighted(results, domain_weights):
    """
    Compute a weighted blend of util tensors.

    domain_weights: list of (key, weight) pairs.  Weights need not sum to 1;
    they are normalised internally so the blend is always a convex combination.

    Returns util (L, E) float32.
    """
    total_w = sum(w for _, w in domain_weights)
    if total_w <= 0:
        sys.exit("[ERROR] --domain-weights sum is zero.")
    blend = None
    for key, w in domain_weights:
        u = load_util(results, key)
        if blend is None:
            blend = (w / total_w) * u
        else:
            blend = blend + (w / total_w) * u
    return blend


def load_cot_util(results, cot_key):
    if cot_key == "cot_combined":
        u_think  = load_util(results, "cot_think")
        u_answer = load_util(results, "cot_answer")
        return (u_think + u_answer) / 2.0
    return load_util(results, cot_key)


def compute_avg_rank(domain_result):
    """
    Return avg_rank (L, E) float32.
    avg_rank[l,e] = sum_k(freq[l,e,k]*k) / sum_k(freq[l,e,k]).
    k=0 = primary (highest-weight pick); k=K-1 = filler (lowest-weight pick).
    Experts with near-zero total freq are assigned avg_rank = K (worst tier).
    """
    freq  = domain_result["freq"]                          # (L, E, K) float32
    K     = freq.shape[-1]
    ranks = torch.arange(K, dtype=torch.float32)
    total    = freq.sum(dim=-1)                            # (L, E)
    weighted = (freq * ranks).sum(dim=-1)                  # (L, E)
    avg_rank = torch.where(
        total > 1e-9,
        weighted / total.clamp(min=1e-9),
        torch.full_like(total, float(K)),
    )
    return avg_rank


def compute_avg_rank_blend(results, domain_weights):
    """Weighted blend of avg_rank tensors (same weights as util blend)."""
    total_w = sum(w for _, w in domain_weights)
    blend   = None
    for key, w in domain_weights:
        ar    = compute_avg_rank(results[key])
        blend = ar * (w / total_w) if blend is None else blend + ar * (w / total_w)
    return blend


def load_cot_avg_rank(cot_results, cot_key):
    """Return avg_rank (L, E) for the CoT domain key."""
    if cot_key == "cot_combined":
        return (compute_avg_rank(cot_results["cot_think"]) +
                compute_avg_rank(cot_results["cot_answer"])) / 2.0
    return compute_avg_rank(cot_results[cot_key])


# ---------------------------------------------------------------------------
# Whitelist / retrieval set loaders
# ---------------------------------------------------------------------------

def load_structural_whitelist_ablation(ablation_json):
    """
    Load structural whitelist from legacy kb_ablation_results.json.
    (Medical domain bisection output — original format.)
    Criterion: confirmed_experts entry with rescued==False and verdict=="structural".
    """
    whitelist = set()
    with open(ablation_json, encoding="utf-8") as f:
        data = json.load(f)
    for q_key, q_val in data.items():
        if q_key.startswith("_"):
            continue
        for half_key in ("half_seed42_halfA", "half_seed42_halfB"):
            half = q_val.get(half_key, {})
            for ce in half.get("confirmed_experts", []):
                if not isinstance(ce, dict):
                    continue
                if ce.get("verdict") == "structural" and not ce.get("rescued", True):
                    whitelist.add((ce["layer"], ce["expert"]))
    return whitelist


def load_retrieval_set_ablation(ablation_json):
    """
    Load recall/retrieval set from legacy kb_ablation_results.json.
    Criterion: verdict=="retrieval_pathway" OR rescued==True.
    """
    retrieval = set()
    with open(ablation_json, encoding="utf-8") as f:
        data = json.load(f)
    for q_key, q_val in data.items():
        if q_key.startswith("_"):
            continue
        for half_key in ("half_seed42_halfA", "half_seed42_halfB"):
            half = q_val.get(half_key, {})
            for ce in half.get("confirmed_experts", []):
                if not isinstance(ce, dict):
                    continue
                if ce.get("verdict") == "retrieval_pathway" or ce.get("rescued", False):
                    retrieval.add((ce["layer"], ce["expert"]))
    return retrieval


def load_probe_sets(probe_json):
    """
    Load structural_set and recall_set from a probe analysis JSON produced
    by analyze_probe.py.  Format:
      {
        "structural_set": [[layer, expert], ...],
        "recall_set":     [[layer, expert], ...]
      }
    Returns (structural_set, recall_set) as sets of (layer, expert) tuples.
    """
    with open(probe_json, encoding="utf-8") as f:
        data = json.load(f)
    structural = {tuple(le) for le in data.get("structural_set", [])}
    recall     = {tuple(le) for le in data.get("recall_set", [])}
    return structural, recall


def load_probe_structural_ordered(probe_json):
    """
    Load structural set as an ordered list from probe analysis JSON.
    Order is delta_matched descending (most KB-activated first), as written by analyze_probe.py.
    Returns list of (layer, expert) tuples.
    """
    with open(probe_json, encoding="utf-8") as f:
        data = json.load(f)
    return [(int(le[0]), int(le[1])) for le in data.get("structural_set", [])]


# Unified loaders — callers use these regardless of source
def load_structural_whitelist(source_path, is_probe=False):
    if is_probe:
        s, _ = load_probe_sets(source_path)
        return s
    return load_structural_whitelist_ablation(source_path)


def load_closure_whitelist(path):
    """
    Load closure expert whitelist from JSON produced by analyze_closure_experts.py.
    Format: {"closure_whitelist": {"layer_str": [expert_ints], ...}, "_meta": {...}}
    Falls back to direct {"layer_str": [expert_ints]} format.
    Returns dict {layer_int: list[int]}.
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    raw = data.get("closure_whitelist", data)
    return {int(k): list(v) for k, v in raw.items() if not k.startswith("_")}


def load_retrieval_set(source_path, is_probe=False):
    if is_probe:
        _, r = load_probe_sets(source_path)
        return r
    return load_retrieval_set_ablation(source_path)


# ---------------------------------------------------------------------------
# Pass 1 — Domain baseline
# ---------------------------------------------------------------------------

def pass1_topk(util_domain, budget):
    """
    Returns M: list of sets, M[l] = set of expert indices for layer l.
    Layers with all-zero util get an empty set (non-MoE dense layers).
    """
    n_layers, n_experts = util_domain.shape
    M = []
    for l in range(n_layers):
        row = util_domain[l]
        if row.sum().item() == 0.0:
            M.append(set())
        else:
            _, top_idx = torch.topk(row, k=min(budget, n_experts))
            M.append(set(top_idx.tolist()))
    return M


def build_wl_union(wl_sources, wl_thresh, min_count=0):
    """
    Build per-layer whitelist union from a list of (results_dict, key) pairs.

    For each layer, an expert is whitelisted if avg_rank <= wl_thresh in ANY
    of the source keys, AND its rank-0 token count >= min_count in that key.
    th_ew keys are skipped (identical histogram to th).

    min_count=0 disables the count gate (include all avg_rank qualifiers).
    min_count=10 (default in CLI) guards against noise from tiny corpora —
    an expert fired as near-primary on < 10 tokens has insufficient evidence
    to warrant a guaranteed slot AND eviction protection.

    Returns wl_union: list of sets, wl_union[l] = frozenset of expert indices.
    """
    if not wl_sources:
        return None

    # Determine shape from first source
    first_dr = wl_sources[0][0][wl_sources[0][1]]
    n_layers, n_experts = first_dr["freq"].shape[:2]
    union = [set() for _ in range(n_layers)]

    for results, key in wl_sources:
        dr = results[key]
        ar = compute_avg_rank(dr)   # (L, E)
        # rank-0 raw token count — shape (L, E)
        hist_r0 = dr["histogram"][:, :, 0].long() if "histogram" in dr else None
        for l in range(n_layers):
            for e in range(n_experts):
                if ar[l, e].item() <= wl_thresh:
                    if min_count > 0 and hist_r0 is not None:
                        if hist_r0[l, e].item() < min_count:
                            continue
                    union[l].add(e)

    return union


def pass1_wl_union_topk(util_domain, budget, wl_union):
    """
    Pass 1 with WL-union guarantees.

    Algorithm per layer:
      1. Seed mask with all WL experts (guaranteed, regardless of util rank).
      2. Fill remaining (budget - |WL|) slots from non-WL experts in descending
         util order.
      3. If WL alone exceeds budget (should not happen given data), include all
         WL experts anyway and log a warning — never silently drop hyper-specialists.

    Returns M: list of sets, M[l] = set of expert indices.
    wl_rescued[l] = number of WL experts that util-topK alone would have missed.
    """
    n_layers, n_experts = util_domain.shape
    M          = []
    wl_rescued = []

    for l in range(n_layers):
        row = util_domain[l]
        if row.sum().item() == 0.0:
            M.append(set())
            wl_rescued.append(0)
            continue

        wl_l = wl_union[l] if wl_union else set()

        # Baseline top-budget by util (for rescue accounting only)
        k_base  = min(budget, n_experts)
        top_idx = set(torch.topk(row, k=k_base, largest=True).indices.tolist())
        rescued = len(wl_l - top_idx)
        wl_rescued.append(rescued)

        if len(wl_l) >= budget:
            # Budget overflow — keep all WL, warn
            print(f"  [WARN] Layer {l}: WL union ({len(wl_l)}) >= budget ({budget}); "
                  f"including all WL experts.")
            M.append(set(wl_l))
            continue

        # Seed with WL, then greedily fill by util descending
        mask_l      = set(wl_l)
        remaining   = budget - len(mask_l)
        # Sort non-WL experts by util descending
        order = sorted(
            [e for e in range(n_experts) if e not in mask_l],
            key=lambda e: -row[e].item(),
        )
        mask_l.update(order[:remaining])
        M.append(mask_l)

    return M, wl_rescued


# ---------------------------------------------------------------------------
# Tier / ordinal helpers (shared by Pass 2 and Pass 3)
# ---------------------------------------------------------------------------

def _avg_rank_tier(ar_value, tier_thresholds):
    """Return integer tier (1=chronic filler, 4=near-primary) for a scalar avg_rank."""
    Q1, Q2, Q3 = tier_thresholds
    if   ar_value >= Q3: return 1
    elif ar_value >= Q2: return 2
    elif ar_value >= Q1: return 3
    else:                return 4


def layer_ordinal_ranks(util_layer):
    """
    Map each expert index → ordinal rank (1 = highest util) for one layer.
    util_layer: 1-D tensor of length n_experts.
    Ties broken by expert index ascending (lower index = marginally better rank).
    Returns a plain list of length n_experts: result[e] = rank of expert e (1-based).
    """
    n     = len(util_layer)
    order = sorted(range(n), key=lambda e: (-util_layer[e].item(), e))
    result = [0] * n
    for rank, e in enumerate(order):
        result[e] = rank + 1
    return result


def build_eviction_pool(experts_in_mask, avg_rank_layer, util_layer,
                        exempt_set, tier_thresholds, allow_tier4):
    """
    Return ordered list of eviction candidates (most evictable first).
    Sorted: tier ascending (tier 1 first), then util_domain ascending within tier.
    exempt_set: expert indices excluded from eviction (e.g. structural experts in mask).
    allow_tier4: if False, tier-4 experts (avg_rank < Q1, near-primary) are excluded.
    """
    pool = [e for e in experts_in_mask if e not in exempt_set]
    if not allow_tier4:
        pool = [e for e in pool
                if avg_rank_layer[e].item() >= tier_thresholds[0]]
    pool.sort(key=lambda e: (_avg_rank_tier(avg_rank_layer[e].item(), tier_thresholds),
                             util_layer[e].item()))
    return pool


# ---------------------------------------------------------------------------
# Pass 2 — CoT injection (reasoning expert inclusion)
# ---------------------------------------------------------------------------

def pass2_cot_injection(M, util_domain, util_cot, avg_rank_dom, avg_rank_cot,
                        tier_thresholds, cap=6, rank_offset=0, dry_run=False,
                        wl_union=None):
    """
    Per layer: inject top-CoT reasoning experts by evicting low-tier generalists.

    Ordinal ranks are fixed from the Pass 1 (minrank) util snapshots — never updated
    mid-pass.  Guard:
      cot_ordinal < dom_ordinal         → swap approved
      cot_ordinal == dom_ordinal        → swap if avg_rank_cot < avg_rank_dom; else STOP
      cot_ordinal > dom_ordinal         → STOP
    Eviction exemptions (strongest first):
      - WL union members in M[l]: explicitly protected, cannot be evicted.
      - Tier-4 experts (avg_rank_dom < Q1): also exempt (implicit, via allow_tier4=False).
    WL members already satisfy avg_rank < Q1 in almost all cases, but the explicit
    exempt_set ensures correctness even at boundary values (avg_rank == Q1).
    Cap: at most `cap` swaps per layer.  Unused budget does NOT roll over to Pass 3.
    rank_offset: skip the top-N CoT candidates per layer before applying cap.
    wl_union: list of sets (per layer) from build_wl_union(); if provided, WL members
              present in M[l] are added to the explicit exempt_set.
    Returns list of swap-log dicts.
    """
    swaps = []
    n_exp = util_domain.shape[1]

    for l in range(len(M)):
        if not M[l]:
            continue

        # Fixed pre-pass ordinal ranks (1=best) — read-only during the swap loop
        dom_ord = layer_ordinal_ranks(util_domain[l])
        cot_ord = layer_ordinal_ranks(util_cot[l])

        # Explicit WL protection: WL members in the mask are never eviction targets
        wl_exempt = (wl_union[l] & M[l]) if (wl_union and wl_union[l]) else set()

        # Eviction pool: tier-stratified generalists, tier-4 exempt + WL exempt
        evict_pool = build_eviction_pool(
            M[l], avg_rank_dom[l], util_domain[l],
            exempt_set=wl_exempt, tier_thresholds=tier_thresholds, allow_tier4=False,
        )

        # CoT candidates: not in M[l]; sorted by util_cot desc, tie by avg_rank_cot asc
        cot_cands = sorted(
            [e for e in range(n_exp) if e not in M[l]],
            key=lambda e: (-util_cot[l, e].item(), avg_rank_cot[l, e].item(), e),
        )
        if rank_offset > 0:
            cot_cands = cot_cands[rank_offset:]  # skip top-N CoT candidates this layer

        layer_count = 0
        for evict_e, cot_e in zip(evict_pool, cot_cands):
            if layer_count >= cap:
                break

            c_ord = cot_ord[cot_e]
            d_ord = dom_ord[evict_e]

            if c_ord < d_ord:
                approved = True
            elif c_ord == d_ord:
                approved = avg_rank_cot[l, cot_e].item() < avg_rank_dom[l, evict_e].item()
                if not approved:
                    break   # tied with no avg_rank advantage — stop
            else:
                break       # guard failed — stop

            if approved:
                ar_cot = round(avg_rank_cot[l, cot_e].item(),   4)
                ar_dom = round(avg_rank_dom[l, evict_e].item(), 4)
                entry = {
                    "pass":             2,
                    "layer":            l,
                    "added":            int(cot_e),
                    "evicted":          int(evict_e),
                    "cot_ordinal":      c_ord,
                    "dom_ordinal":      d_ord,
                    "avg_rank_cot":     ar_cot,
                    "avg_rank_dom_ev":  ar_dom,
                    "evict_tier":       _avg_rank_tier(ar_dom, tier_thresholds),
                    "util_added_cot":   round(util_cot[l, cot_e].item(),      6),
                    "util_evicted_dom": round(util_domain[l, evict_e].item(), 6),
                }
                swaps.append(entry)
                tag = (f"cot_rank={c_ord} avg_r={ar_cot:.2f}  "
                       f"evict E{evict_e:03d} "
                       f"(dom_rank={d_ord} avg_r={ar_dom:.2f} "
                       f"tier={entry['evict_tier']})")
                if dry_run:
                    print(f"  [Pass2 DRY] L{l:02d}: add E{cot_e:03d} {tag}")
                else:
                    M[l].discard(evict_e)
                    M[l].add(cot_e)
                    print(f"  [Pass2] L{l:02d}: add E{cot_e:03d} {tag}")
                layer_count += 1

    return swaps


# ---------------------------------------------------------------------------
# Pass 3 — Structural injection (KB-processing expert inclusion)
# ---------------------------------------------------------------------------

def pass3_structural_injection(M, util_domain, avg_rank_dom, structural_ordered,
                               structural_set, tier_thresholds, cap=4, dry_run=False):
    """
    Per layer: force KB structural experts (highest delta_matched first) into the mask.

    Candidates: structural experts not yet in M[l], in delta_matched descending order
                (priority order preserved from probe analysis JSON).
    Eviction pool: all experts in M[l] EXCEPT structural members already present —
                   sorted (tier asc, util_domain asc).  Tier 4 eligible here.
    Cap: at most `cap` swaps per layer.  Independent of Pass 2 budget.
    No gain guard — structural injection is unconditional within the cap.
    Returns list of swap-log dicts.
    """
    # Index structural candidates by layer, preserving delta_matched priority order
    struct_by_layer = {}
    for l, e in structural_ordered:
        struct_by_layer.setdefault(l, []).append(e)

    swaps = []
    for l in range(len(M)):
        if not M[l]:
            continue

        candidates = [e for e in struct_by_layer.get(l, []) if e not in M[l]]
        if not candidates:
            continue

        # Protect structural members already present in mask from eviction this pass
        struct_in_mask = {e for e in M[l] if (l, e) in structural_set}

        # Eviction pool: all non-structural experts, tier 4 eligible
        evict_pool = build_eviction_pool(
            M[l], avg_rank_dom[l], util_domain[l],
            exempt_set=struct_in_mask, tier_thresholds=tier_thresholds, allow_tier4=True,
        )

        layer_count = 0
        for cand_e, evict_e in zip(candidates, evict_pool):
            if layer_count >= cap:
                break

            ar_ev = round(avg_rank_dom[l, evict_e].item(), 4)
            t     = _avg_rank_tier(ar_ev, tier_thresholds)
            entry = {
                "pass":             3,
                "layer":            l,
                "added":            int(cand_e),
                "evicted":          int(evict_e),
                "evict_tier":       t,
                "avg_rank_evicted": ar_ev,
                "util_evicted_dom": round(util_domain[l, evict_e].item(), 6),
            }
            swaps.append(entry)
            tag = (f"evict E{evict_e:03d} "
                   f"(tier={t} avg_r={ar_ev:.2f} "
                   f"util={entry['util_evicted_dom']:.5f})")
            if dry_run:
                print(f"  [Pass3 DRY] L{l:02d}: add structural E{cand_e:03d}  {tag}")
            else:
                M[l].discard(evict_e)
                M[l].add(cand_e)
                print(f"  [Pass3] L{l:02d}: add structural E{cand_e:03d}  {tag}")
            layer_count += 1

    return swaps


# ---------------------------------------------------------------------------
# Pass CL — Closure expert injection
# ---------------------------------------------------------------------------

def pass_closure_injection(M, util_domain, avg_rank_dom, closure_by_layer,
                           tier_thresholds, dry_run=False):
    """
    Pass CL: Unconditional closure expert injection.

    For each layer l, inject any closure whitelist expert not already in M[l]
    by evicting the weakest generalist (tier-1 first, util_domain asc within tier;
    tier-4 near-primary experts are exempt from eviction).
    Closure experts already present in M[l] are also exempt from eviction.

    No gain guard — the whitelist was pre-screened by 2-corpus intersection.
    No per-layer cap — the closure whitelist size is inherently bounded.
    Returns list of swap-log dicts.
    """
    swaps = []
    for l in range(len(M)):
        if not M[l]:
            continue
        candidates = [e for e in closure_by_layer.get(l, []) if e not in M[l]]
        if not candidates:
            continue

        # Closure experts already in the mask are protected from eviction
        closure_all_l   = set(closure_by_layer.get(l, []))
        closure_in_mask = {e for e in M[l] if e in closure_all_l}

        for cand_e in candidates:
            # Rebuild eviction pool each iteration — mask changes as we inject
            evict_pool = build_eviction_pool(
                M[l], avg_rank_dom[l], util_domain[l],
                exempt_set=closure_in_mask,
                tier_thresholds=tier_thresholds,
                allow_tier4=False,
            )
            if not evict_pool:
                print(f"  [PassCL WARN] L{l:02d}: eviction pool exhausted, "
                      f"cannot inject closure E{cand_e:03d}")
                continue

            evict_e = evict_pool[0]
            ar_ev   = round(avg_rank_dom[l, evict_e].item(), 4)
            t       = _avg_rank_tier(ar_ev, tier_thresholds)
            entry = {
                "pass":             "CL",
                "layer":            l,
                "added":            int(cand_e),
                "evicted":          int(evict_e),
                "evict_tier":       t,
                "avg_rank_evicted": ar_ev,
                "util_evicted_dom": round(util_domain[l, evict_e].item(), 6),
            }
            swaps.append(entry)
            tag = (f"evict E{evict_e:03d} "
                   f"(tier={t} avg_r={ar_ev:.2f} "
                   f"util={entry['util_evicted_dom']:.5f})")
            if dry_run:
                print(f"  [PassCL DRY] L{l:02d}: add closure E{cand_e:03d}  {tag}")
            else:
                M[l].discard(evict_e)
                M[l].add(cand_e)
                closure_in_mask.add(cand_e)  # protect newly added closure expert
                print(f"  [PassCL] L{l:02d}: add closure E{cand_e:03d}  {tag}")

    return swaps


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def mask_to_json(M):
    """Convert list-of-sets to {layer_str: sorted_list} (same format as existing masks)."""
    out = {}
    for l, experts in enumerate(M):
        if experts:
            out[str(l)] = sorted(experts)
    return out


def coverage_fraction(util_matrix, M):
    """Fraction of total domain util captured by the mask."""
    total = util_matrix.sum().item()
    if total == 0:
        return 0.0
    kept  = sum(
        util_matrix[l, list(M[l])].sum().item()
        for l in range(len(M)) if M[l]
    )
    return kept / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_domain_weights(s):
    """
    Parse --domain-weights string "key1:w1,key2:w2,..." into list of (key, float) pairs.
    """
    pairs = []
    for token in s.split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            sys.exit(f"[ERROR] --domain-weights token '{token}' must be 'key:weight'")
        k, w = token.rsplit(":", 1)
        try:
            pairs.append((k.strip(), float(w.strip())))
        except ValueError:
            sys.exit(f"[ERROR] --domain-weights weight '{w}' is not a number")
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Three-pass CoE expert mask builder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # -- Domain selection (mutually exclusive: single key or weighted blend) --
    domain_group = parser.add_mutually_exclusive_group(required=True)
    domain_group.add_argument(
        "--domain",
        help="Single domain key for Pass 1 util baseline.  "
             "Legacy aliases: medical, legal, pharma.  "
             "Any key present in --pt-file is accepted verbatim.",
    )
    domain_group.add_argument(
        "--domain-weights", metavar="KEY:W,...",
        help="Weighted blend of domain keys for Pass 1, e.g. "
             "coding_python:0.35,coding_cpp:0.25,coding_go:0.20,coding_rust:0.20.  "
             "Weights are normalised; need not sum to 1.",
    )

    parser.add_argument(
        "--output-name", metavar="LABEL",
        help="Short label used in output filename: mask_coe_{LABEL}_swapactive_K{budget}.json.  "
             "Defaults to the --domain key (or 'blend' when --domain-weights is used).",
    )
    parser.add_argument(
        "--budget", type=int, default=64,
        help="Number of experts per MoE layer.",
    )
    parser.add_argument(
        "--cot-domain", default="cot_think",
        choices=["cot_think", "cot_answer", "cot_reasoning", "cot_combined"],
        help="CoT domain key for Pass 3 reasoning utility.",
    )
    parser.add_argument(
        "--pt-file", default=PT_FILE,
        help="Path to the .pt results file containing domain util tensors.",
    )
    parser.add_argument(
        "--wl-sources", metavar="FILE::KEY,...", default=None,
        help="Comma-separated list of FILE::KEY pairs used to build the per-layer "
             "whitelist union.  An expert is whitelisted if avg_rank <= --wl-thresh "
             "in ANY listed key.  Keys with suffix '_th_ew' are automatically skipped "
             "(identical histogram to _th).  Example: "
             "histograms/final/physics_base_results.pt::physics,"
             "histograms/final/physics_base_results.pt::physics_hydrated,"
             "histograms/physics_thinking_results.pt::physics_qa,"
             "histograms/physics_thinking_results.pt::physics_th",
    )
    parser.add_argument(
        "--wl-thresh", type=float, default=2.0,
        help="avg_rank threshold for whitelist membership.  "
             "Experts with avg_rank <= this value in any --wl-sources key are "
             "guaranteed slots in Pass 1 regardless of their util rank.",
    )
    parser.add_argument(
        "--wl-thresh-count", type=int, default=10,
        help="Minimum rank-0 token count for whitelist membership (default: 10).  "
             "An expert must appear as the primary selection (rank 0) at least this "
             "many times to qualify for WL inclusion.  Prevents noise experts from "
             "small corpora getting a guaranteed Pass 1 slot AND eviction protection "
             "in Pass 2.  Set to 0 to disable the count gate.",
    )
    parser.add_argument(
        "--cot-pt-file", default=None,
        help="Path to a separate .pt file that contains the CoT domain keys.  "
             "Defaults to --pt-file when not specified.  "
             "Use this when cot_think lives in a different file from the domain data "
             "(e.g. new coding profile but CoT from medical_legal_pharma_results.pt).",
    )
    parser.add_argument(
        "--probe-json", default=None,
        help="Path to probe analysis JSON (from analyze_probe.py) containing "
             "structural_set and recall_set.  "
             "When provided, used for Pass 2/3 instead of --ablation-json.",
    )
    parser.add_argument(
        "--ablation-json", default=ABLATION_JSON,
        help="Path to kb_ablation_results.json (legacy medical bisection output).  "
             "Ignored when --probe-json is provided.  "
             "Pass 2/3 are auto-skipped if the file does not exist and "
             "--probe-json is also absent.",
    )
    parser.add_argument(
        "--output-dir", default=OUTPUT_DIR,
        help="Directory to write mask and swap-log JSON files.",
    )
    parser.add_argument(
        "--no-pass2", action="store_true",
        help="Skip Pass 2 (structural whitelist enforcement).",
    )
    parser.add_argument(
        "--no-pass3", action="store_true",
        help="Skip Pass 3 (KB-rescue arbitrage).",
    )
    parser.add_argument(
        "--closure-whitelist", default=None, metavar="PATH",
        help="Path to closure expert whitelist JSON (from analyze_closure_experts.py).  "
             "When provided, runs Pass CL after Pass 2: injects closure experts not "
             "already in the mask by evicting the weakest tier-1 generalist per slot.  "
             "Tier-4 (near-primary domain) experts are exempt from eviction.",
    )
    parser.add_argument(
        "--pass2-cap", type=int, default=6,
        help="Maximum CoT swaps per layer in Pass 2.  "
             "Unused budget does NOT roll over to Pass 3.",
    )
    parser.add_argument(
        "--pass2-rank-offset", type=int, default=0,
        help="Skip the top-N CoT candidates per layer before applying Pass 2.  "
             "Injects ranks [N+1 .. N+cap] instead of [1 .. cap].  Useful for "
             "testing whether hyper-specialist top-rank experts cause instability.",
    )
    parser.add_argument(
        "--pass3-cap", type=int, default=4,
        help="Maximum structural injection swaps per layer in Pass 3.  "
             "Independent of --pass2-cap.",
    )
    parser.add_argument(
        "--tier-thresholds", type=float, nargs=3, default=[2.0, 4.0, 6.0],
        metavar=("Q1", "Q2", "Q3"),
        help="Avg-rank tier boundaries for eviction stratification.  "
             "Tier 4 (near-primary): avg_rank < Q1  — exempt from Pass 2.  "
             "Tier 1 (filler): avg_rank >= Q3  — evicted first.  "
             "Defaults are quartiles of top-8 routing: 2.0 4.0 6.0.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print swap log without modifying mask or writing files.",
    )
    args = parser.parse_args()

    # -- Resolve domain key / weighted blend --------------------------------
    if args.domain:
        domain_key   = DOMAIN_ALIAS.get(args.domain, args.domain)
        domain_weights_parsed = None
        short_name   = args.output_name or domain_key
    else:
        domain_weights_parsed = parse_domain_weights(args.domain_weights)
        domain_key   = None   # not a single key
        short_name   = args.output_name or "blend"

    print(f"\n{'='*60}")
    print(f"build_coe_mask.py — Three-Pass CoE Mask Builder")
    print(f"{'='*60}")
    if domain_key:
        print(f"Domain       : {args.domain} -> {domain_key}")
    else:
        print(f"Domain blend : {args.domain_weights}")
    print(f"Output label : {short_name}")
    print(f"Budget       : {args.budget}")
    print(f"CoT domain   : {args.cot_domain}")
    print(f"CoT PT file  : {args.cot_pt_file or '(same as --pt-file)'}")
    print(f"Probe JSON   : {args.probe_json or '(none — using ablation-json)'}")
    print(f"Passes active: 1=yes  2={'no' if args.no_pass2 else 'yes'}  "
          f"3={'no' if args.no_pass3 else 'yes'}  "
          f"CL={'yes' if args.closure_whitelist else 'no'}")
    print(f"Pass caps     : p2={args.pass2_cap}  p3={args.pass3_cap}  p2-rank-offset={args.pass2_rank_offset}")
    if args.closure_whitelist:
        print(f"Closure WL    : {args.closure_whitelist}")
    print(f"Tier thresholds: Q1={args.tier_thresholds[0]}  "
          f"Q2={args.tier_thresholds[1]}  Q3={args.tier_thresholds[2]}")
    print(f"Dry-run      : {args.dry_run}")
    print()

    # ------------------------------------------------------------------
    # Load PT results file(s)
    # ------------------------------------------------------------------
    if not os.path.exists(args.pt_file):
        sys.exit(f"[ERROR] PT file not found: {args.pt_file}")
    print(f"Loading {args.pt_file} ...")
    results = torch.load(args.pt_file, weights_only=False)

    # Separate CoT PT file (optional)
    cot_pt_file = args.cot_pt_file or args.pt_file
    if cot_pt_file != args.pt_file:
        if not os.path.exists(cot_pt_file):
            sys.exit(f"[ERROR] --cot-pt-file not found: {cot_pt_file}")
        print(f"Loading CoT PT: {cot_pt_file} ...")
        cot_results = torch.load(cot_pt_file, weights_only=False)
    else:
        cot_results = results

    # ------------------------------------------------------------------
    # Compute util matrices
    # ------------------------------------------------------------------
    print("Computing util matrices ...")
    if domain_key:
        util_domain = load_util(results, domain_key)
    else:
        util_domain = load_util_weighted(results, domain_weights_parsed)
        keys_used = ", ".join(f"{k}:{w}" for k, w in domain_weights_parsed)
        print(f"  weighted blend: {keys_used}")
    util_cot = load_cot_util(cot_results, args.cot_domain)
    n_layers, n_experts = util_domain.shape
    print(f"  util_domain : {tuple(util_domain.shape)}  "
          f"total={util_domain.sum().item():.4f}")
    print(f"  util_cot    : {tuple(util_cot.shape)}  "
          f"total={util_cot.sum().item():.4f}")

    # Avg-rank tensors — computed from fixed Pass 1 baseline, read-only during passes
    print("Computing avg_rank tensors ...")
    if domain_key:
        avg_rank_dom = compute_avg_rank(results[domain_key])
    else:
        avg_rank_dom = compute_avg_rank_blend(results, domain_weights_parsed)
    avg_rank_cot = load_cot_avg_rank(cot_results, args.cot_domain)
    tier_thresholds = args.tier_thresholds
    print(f"  avg_rank_dom: {tuple(avg_rank_dom.shape)}  "
          f"mean={avg_rank_dom.mean().item():.3f}")
    print(f"  avg_rank_cot: {tuple(avg_rank_cot.shape)}  "
          f"mean={avg_rank_cot.mean().item():.3f}")

    # ------------------------------------------------------------------
    # Load whitelist / retrieval sets
    # ------------------------------------------------------------------
    whitelist     = set()
    retrieval_set = set()

    if args.probe_json:
        if not os.path.exists(args.probe_json):
            sys.exit(f"[ERROR] --probe-json not found: {args.probe_json}")
        print(f"\nLoading probe JSON: {args.probe_json} ...")
        whitelist     = load_structural_whitelist(args.probe_json, is_probe=True)
        retrieval_set = load_retrieval_set(args.probe_json, is_probe=True)
    elif os.path.exists(args.ablation_json):
        print(f"\nLoading ablation JSON: {args.ablation_json} ...")
        whitelist     = load_structural_whitelist(args.ablation_json, is_probe=False)
        retrieval_set = load_retrieval_set(args.ablation_json, is_probe=False)
    else:
        print(f"\n[INFO] No probe-json or ablation-json found — "
              f"Pass 2/3 will have empty whitelist/retrieval sets.")
    dual_role     = whitelist & retrieval_set
    print(f"  Structural whitelist : {len(whitelist)} slots")
    print(f"  Retrieval set        : {len(retrieval_set)} slots")
    print(f"  Dual-role (both)     : {len(dual_role)} slots")
    if dual_role:
        for (l, e) in sorted(dual_role):
            print(f"    dual-role: L{l:02d} E{e:03d}")

    # Ordered structural list for Pass 3 priority (delta_matched desc) — probe JSON only
    structural_ordered = []
    if args.probe_json and os.path.exists(args.probe_json):
        structural_ordered = load_probe_structural_ordered(args.probe_json)
        print(f"  Structural ordered   : {len(structural_ordered)} entries "
              f"(delta_matched desc)")

    # -- Load closure whitelist (optional Pass CL) ----------------------
    closure_by_layer = {}
    if args.closure_whitelist:
        if not os.path.exists(args.closure_whitelist):
            sys.exit(f"[ERROR] --closure-whitelist not found: {args.closure_whitelist}")
        print(f"\nLoading closure whitelist: {args.closure_whitelist} ...")
        closure_by_layer = load_closure_whitelist(args.closure_whitelist)
        total_closure = sum(len(v) for v in closure_by_layer.values())
        print(f"  Closure candidates   : {total_closure} across "
              f"{len(closure_by_layer)} layers")

    # ------------------------------------------------------------------
    # Build per-layer WL union (optional)
    # ------------------------------------------------------------------
    wl_union = None
    if not args.wl_sources and "_wl_union" in results:
        raw = results["_wl_union"]
        if raw:
            wl_union = [set(layer_list) for layer_list in raw]
            total_wl = sum(len(s) for s in wl_union)
            thresh   = results.get("_wl_thresh", args.wl_thresh)
            print(f"\nUsing embedded _wl_union from .pt file  "
                  f"({total_wl} pairs, thresh={thresh})")
    if not wl_union and args.wl_sources:
        print("\nBuilding WL union ...")
        wl_file_cache = {}
        wl_source_pairs = []
        for token in args.wl_sources.split(","):
            token = token.strip()
            if not token:
                continue
            if "::" not in token:
                sys.exit(f"[ERROR] --wl-sources token '{token}' must be 'FILE::KEY'")
            fpath, key = token.split("::", 1)
            fpath = fpath.strip(); key = key.strip()
            if key.endswith("_th_ew"):
                print(f"  Skipping {key} (identical histogram to _th)")
                continue
            if fpath not in wl_file_cache:
                if not os.path.exists(fpath):
                    sys.exit(f"[ERROR] --wl-sources file not found: {fpath}")
                print(f"  Loading {os.path.basename(fpath)} ...")
                wl_file_cache[fpath] = torch.load(fpath, weights_only=False)
            wl_source_pairs.append((wl_file_cache[fpath], key))
        wl_union = build_wl_union(wl_source_pairs, args.wl_thresh,
                                   min_count=args.wl_thresh_count)
        total_wl = sum(len(s) for s in wl_union)
        wl_layer_counts = [len(s) for s in wl_union]
        import statistics as _stats
        print(f"  WL union: {total_wl} total (l,e) pairs  "
              f"min={min(wl_layer_counts)}  max={max(wl_layer_counts)}  "
              f"mean={_stats.mean(wl_layer_counts):.1f}/layer  "
              f"thresh={args.wl_thresh}  min_count={args.wl_thresh_count}")

    # ------------------------------------------------------------------
    # Pass 1 — domain baseline
    # ------------------------------------------------------------------
    if wl_union:
        print(f"\n--- Pass 1: Domain baseline (WL-union + util fill to {args.budget}/layer) ---")
        M, wl_rescued = pass1_wl_union_topk(util_domain, args.budget, wl_union)
        total_rescued = sum(wl_rescued)
        print(f"  WL experts rescued (not in plain util-top-{args.budget}): {total_rescued}")
    else:
        print(f"\n--- Pass 1: Domain baseline (top-{args.budget} by util_domain) ---")
        M = pass1_topk(util_domain, args.budget)
        wl_rescued = [0] * n_layers
    active_layers = sum(1 for s in M if s)
    print(f"  Active MoE layers  : {active_layers}/{n_layers}")
    cov_p1 = coverage_fraction(util_domain, M)
    print(f"  Domain util coverage: {cov_p1:.1%}")

    wl_in_p1 = sum(1 for (l, e) in whitelist if l < len(M) and e in M[l])
    print(f"  Whitelist slots already in mask: {wl_in_p1}/{len(whitelist)}")

    # ------------------------------------------------------------------
    # Pass 2 — CoT injection
    # ------------------------------------------------------------------
    p2_swaps = []
    if not args.no_pass2:
        print(f"\n--- Pass 2: CoT injection (cap={args.pass2_cap}/layer, "
              f"tier thresholds={tier_thresholds}) ---")
        p2_swaps = pass2_cot_injection(
            M, util_domain, util_cot, avg_rank_dom, avg_rank_cot,
            tier_thresholds=tier_thresholds,
            cap=args.pass2_cap,
            rank_offset=args.pass2_rank_offset,
            dry_run=args.dry_run,
            wl_union=wl_union,
        )
        print(f"  Pass 2 swaps: {len(p2_swaps)}")
    else:
        print("\n--- Pass 2: SKIPPED (--no-pass2) ---")

    # ------------------------------------------------------------------
    # Pass CL — Closure expert injection
    # ------------------------------------------------------------------
    pcl_swaps = []
    if closure_by_layer:
        print(f"\n--- Pass CL: Closure injection "
              f"({sum(len(v) for v in closure_by_layer.values())} candidates, "
              f"tier thresholds={tier_thresholds}) ---")
        pcl_swaps = pass_closure_injection(
            M, util_domain, avg_rank_dom,
            closure_by_layer=closure_by_layer,
            tier_thresholds=tier_thresholds,
            dry_run=args.dry_run,
        )
        print(f"  Pass CL swaps: {len(pcl_swaps)}")
    else:
        print("\n--- Pass CL: SKIPPED (no --closure-whitelist) ---")

    # ------------------------------------------------------------------
    # Pass 3 — Structural injection
    # ------------------------------------------------------------------
    p3_swaps = []
    if not args.no_pass3:
        print(f"\n--- Pass 3: Structural injection (cap={args.pass3_cap}/layer) ---")
        if not structural_ordered:
            print("  (structural_ordered is empty — provide --probe-json for Pass 3 "
                  "to be effective)")
        p3_swaps = pass3_structural_injection(
            M, util_domain, avg_rank_dom,
            structural_ordered=structural_ordered,
            structural_set=whitelist,
            tier_thresholds=tier_thresholds,
            cap=args.pass3_cap,
            dry_run=args.dry_run,
        )
        print(f"  Pass 3 swaps: {len(p3_swaps)}")
    else:
        print("\n--- Pass 3: SKIPPED (--no-pass3) ---")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    mask_json    = mask_to_json(M)
    total_slots  = sum(len(v) for v in mask_json.values())
    cov_final    = coverage_fraction(util_domain, M)

    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"  MoE layers in mask  : {len(mask_json)}")
    print(f"  Total expert slots  : {total_slots}")
    print(f"  Domain util coverage: {cov_final:.1%}  "
          f"(pass-1 baseline: {cov_p1:.1%})")
    if wl_union:
        print(f"  WL rescued (Pass 1) : {sum(wl_rescued)}")
    print(f"  Pass 2 swaps        : {len(p2_swaps)}")
    print(f"  Pass CL swaps       : {len(pcl_swaps)}")
    print(f"  Pass 3 swaps        : {len(p3_swaps)}")

    # Per-pass layer breakdown
    if p2_swaps:
        from collections import Counter
        p2_by_layer = Counter(s["layer"] for s in p2_swaps)
        print(f"  Pass 2 by layer: { dict(sorted(p2_by_layer.items())) }")
    if pcl_swaps:
        from collections import Counter
        pcl_by_layer = Counter(s["layer"] for s in pcl_swaps)
        print(f"  Pass CL by layer: { dict(sorted(pcl_by_layer.items())) }")
    if p3_swaps:
        from collections import Counter
        p3_by_layer = Counter(s["layer"] for s in p3_swaps)
        print(f"  Pass 3 by layer: { dict(sorted(p3_by_layer.items())) }")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # ------------------------------------------------------------------
    # Write output files
    # ------------------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    out_fname  = f"mask_coe_{short_name}_swapactive_K{args.budget}.json"
    out_path   = os.path.join(args.output_dir, out_fname)
    log_fname  = f"mask_coe_{short_name}_swapactive_K{args.budget}_swaplog.json"
    log_path   = os.path.join(args.output_dir, log_fname)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mask_json, f, indent=2)
    print(f"\nSaved mask    : {out_path}")

    log_data = {
        "domain":               args.domain or args.domain_weights,
        "domain_key":           domain_key,
        "output_name":          short_name,
        "budget":               args.budget,
        "cot_domain":           args.cot_domain,
        "cot_pt_file":          cot_pt_file,
        "probe_json":           args.probe_json,
        "pass2_cap":            args.pass2_cap,
        "pass2_rank_offset":    args.pass2_rank_offset,
        "pass3_cap":            args.pass3_cap,
        "tier_thresholds":      tier_thresholds,
        "no_pass2":             args.no_pass2,
        "no_pass3":             args.no_pass3,
        "n_active_layers":      len(mask_json),
        "total_expert_slots":   total_slots,
        "util_coverage_p1":     round(cov_p1,    4),
        "util_coverage_final":  round(cov_final, 4),
        "wl_sources":           args.wl_sources,
        "wl_thresh":            args.wl_thresh if args.wl_sources else None,
        "wl_rescued_total":     sum(wl_rescued) if wl_union else 0,
        "wl_rescued_per_layer": wl_rescued,
        "whitelist_slots":      len(whitelist),
        "retrieval_slots":      len(retrieval_set),
        "dual_role_slots":      len(dual_role),
        "structural_ordered_n": len(structural_ordered),
        "p2_swaps":             p2_swaps,
        "pcl_swaps":            pcl_swaps,
        "p3_swaps":             p3_swaps,
        "closure_whitelist":    str(args.closure_whitelist) if args.closure_whitelist else None,
        "closure_candidates":   sum(len(v) for v in closure_by_layer.values()),
    }
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)
    print(f"Saved swap log: {log_path}")


if __name__ == "__main__":
    main()
