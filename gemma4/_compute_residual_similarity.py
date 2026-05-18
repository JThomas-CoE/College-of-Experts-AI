#!/usr/bin/env python3
"""
_compute_residual_similarity.py
================================
Compute the 14x14 residual directed shared-fraction matrix across
the MMLU off4 swapactive-K64 domain masks.

Method
------
1. Load all 14 mask JSONs  ->  active set per domain = set of (layer, expert) tuples
2. Universal set U = intersection of all 14 active sets (backbone experts)
3. Residual set R(D) = active(D) - U  (domain-specific experts)
4. Directed fraction  F(A->B) = |R(A) ∩ R(B)| / |R(A)|
   "What fraction of A's domain-specific experts does B also keep?"
5. For each domain: nearest = argmax F(A->B), furthest = argmin F(A->B)  (excluding self)

Output
------
  data/residual_similarity.json   -- full matrix + nearest/furthest rankings
  Printed summary table to stdout
"""

import json
import os

GEMMA4 = os.path.dirname(os.path.abspath(__file__))

DOMAINS = [
    "math", "physics", "engineering", "cs", "chemistry",
    "biology", "health", "economics", "business", "psychology",
    "law", "history", "philosophy", "other",
]

MASK_TEMPLATE = os.path.join(GEMMA4, "mask_coe_{domain}_mmlu_swapactive_K64.json")
OUT_PATH      = os.path.join(GEMMA4, "data", "residual_similarity.json")


def load_active_set(domain: str) -> frozenset:
    path = MASK_TEMPLATE.format(domain=domain)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Mask not found: {path}")
    m = json.load(open(path, encoding="utf-8"))
    return frozenset((int(layer), int(expert)) for layer, experts in m.items() for expert in experts)


def main():
    print("Loading masks...")
    active = {}
    for d in DOMAINS:
        active[d] = load_active_set(d)
        print(f"  {d:<14}  {len(active[d])} active (layer,expert) tuples")

    # Universal set -- tuples present in ALL 14 domains
    universal = frozenset.intersection(*active.values())
    print(f"\nUniversal (backbone) experts: {len(universal)} tuples")
    print(f"Residual sizes:")

    residual = {}
    for d in DOMAINS:
        residual[d] = active[d] - universal
        print(f"  {d:<14}  {len(residual[d])} residual tuples")

    # 14x14 directed fraction matrix
    print("\nComputing 14x14 directed fraction matrix...")
    matrix = {}
    for a in DOMAINS:
        matrix[a] = {}
        ra = residual[a]
        for b in DOMAINS:
            if a == b:
                matrix[a][b] = 1.0
            else:
                rb = residual[b]
                matrix[a][b] = len(ra & rb) / len(ra) if ra else 0.0

    # Nearest and furthest per domain (excluding self)
    rankings = {}
    for a in DOMAINS:
        others = [(b, matrix[a][b]) for b in DOMAINS if b != a]
        others.sort(key=lambda x: x[1], reverse=True)
        rankings[a] = {
            "nearest":  others[0][0],
            "nearest_frac":  round(others[0][1], 4),
            "furthest": others[-1][0],
            "furthest_frac": round(others[-1][1], 4),
            "ranked": [(b, round(f, 4)) for b, f in others],
        }

    # Print summary table
    header = f"{'Domain':<14}  {'Nearest':<14}  {'Frac':>6}  {'Furthest':<14}  {'Frac':>6}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for a in DOMAINS:
        r = rankings[a]
        print(f"{a:<14}  {r['nearest']:<14}  {r['nearest_frac']:>6.4f}  {r['furthest']:<14}  {r['furthest_frac']:>6.4f}")

    # Print full matrix
    print("\nFull directed fraction matrix  F(row -> col):")
    col_w = 8
    header_row = f"{'':14}" + "".join(f"{d:>{col_w}}" for d in DOMAINS)
    print(header_row)
    for a in DOMAINS:
        row = f"{a:<14}" + "".join(f"{matrix[a][b]:>{col_w}.4f}" for b in DOMAINS)
        print(row)

    # Save
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    out = {
        "domains": DOMAINS,
        "universal_count": len(universal),
        "residual_counts": {d: len(residual[d]) for d in DOMAINS},
        "matrix": {a: {b: round(matrix[a][b], 6) for b in DOMAINS} for a in DOMAINS},
        "rankings": rankings,
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
