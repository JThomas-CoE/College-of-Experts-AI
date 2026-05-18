#!/usr/bin/env python3
"""
_run_pairwise_cross_domain_bench.py  —  Stage-2 near/far cross-domain benchmark
=================================================================================
For each of 10 coherent target domains (excluding health/history/other/philosophy),
runs the nearest and farthest specialist on that domain's off5 question set.

Pairings derived from residual_similarity.json (matrix field), excluding
health/history/other/philosophy from both target and visitor pools.

Nearest/farthest = highest/lowest Jaccard-residual overlap among remaining 9
specialists (self excluded):

  math:        near=physics   (0.787)   far=law         (0.485)
  physics:     near=chemistry (0.855)   far=law         (0.448)
  engineering: near=physics   (0.854)   far=law         (0.421)
  cs:          near=math      (0.723)   far=law         (0.485)
  chemistry:   near=physics   (0.855)   far=law         (0.413)
  biology:     near=psychology(0.749)   far=business    (0.576)
  economics:   near=psychology(0.740)   far=chemistry   (0.571)
  business:    near=math      (0.756)   far=law         (0.541)
  psychology:  near=biology   (0.749)   far=engineering (0.486)
  law:         near=economics (0.701)   far=chemistry   (0.413)

Output files
------------
  data/matrix/bench_merged_{visitor}_on_{target}_off5.jsonl

The output schema is identical to bench_mmlu_pro.py standard JSONL.

Usage
-----
    # Run all 20 near/far benches:
    # Run all 20 near/far benches (overnight):
    python _run_pairwise_cross_domain_bench.py

    # Dry-run (print commands only, no execution):
    python _run_pairwise_cross_domain_bench.py --dry-run

    # Restrict to specific target domain(s):
    python _run_pairwise_cross_domain_bench.py --only math physics chemistry
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

GEMMA4     = Path(__file__).parent
DATA_DIR   = GEMMA4 / "data"
OOD_DIR    = DATA_DIR / "ood_samples"
MATRIX_DIR = DATA_DIR / "matrix"
LOG_PATH   = DATA_DIR / "orch_pairwise_stage2.log"

PYTHON   = r"C:\RyzenAI\envs\zimage\python.exe"
BENCH_PY = str(GEMMA4 / "bench_mmlu_pro.py")

# ── Hard-coded pairings from residual_similarity.json ────────────────────────
# Excluded from pool: health, history, other, philosophy
# Each tuple: (target_domain, visitor_model_key, label)
PAIRINGS = [
    ("math",        "mmlu_physics_merged",     "near"),
    ("math",        "mmlu_law_merged",          "far"),
    ("physics",     "mmlu_chemistry_merged",    "near"),
    ("physics",     "mmlu_law_merged",          "far"),
    ("engineering", "mmlu_physics_merged",      "near"),
    ("engineering", "mmlu_law_merged",          "far"),
    ("cs",          "mmlu_math_merged",         "near"),
    ("cs",          "mmlu_law_merged",          "far"),
    ("chemistry",   "mmlu_physics_merged",      "near"),
    ("chemistry",   "mmlu_law_merged",          "far"),
    ("biology",     "mmlu_psychology_merged",   "near"),
    ("biology",     "mmlu_business_merged",     "far"),
    ("economics",   "mmlu_psychology_merged",   "near"),
    ("economics",   "mmlu_chemistry_merged",    "far"),
    ("business",    "mmlu_math_merged",         "near"),
    ("business",    "mmlu_law_merged",          "far"),
    ("psychology",  "mmlu_biology_merged",      "near"),
    ("psychology",  "mmlu_engineering_merged",  "far"),
    ("law",         "mmlu_economics_merged",    "near"),
    ("law",         "mmlu_chemistry_merged",    "far"),
]

SPLIT = "off5"   # held-out OOD split (not used for specialist activation collection)


def visitor_short(model_key: str) -> str:
    """mmlu_physics_merged -> physics"""
    return model_key.replace("mmlu_", "").replace("_merged", "")


def count_records(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def score_jsonl(path: Path) -> dict:
    n = correct = loops = 0
    if not path.exists():
        return dict(n=0, correct=0, loops=0)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            if rec.get("correct") is True:
                correct += 1
            if rec.get("loop_detected"):
                loops += 1
    return dict(n=n, correct=correct, loops=loops)


def tlog(msg: str, log_fh):
    ts   = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    log_fh.write(line + "\n")
    log_fh.flush()


def run_one(target: str, model_key: str, out_path: Path, dry_run: bool, log_fh) -> bool:
    src = OOD_DIR / f"mmlu_pro_ood_{target}_{SPLIT}.jsonl"
    if not src.exists():
        tlog(f"[SKIP] Source not found: {src}", log_fh)
        return False

    n_src  = count_records(src)
    n_done = count_records(out_path)
    if n_done >= n_src > 0:
        s   = score_jsonl(out_path)
        acc = s["correct"] / s["n"] * 100 if s["n"] else 0
        tlog(f"[SKIP] already complete {n_done}/{n_src}  acc={acc:.1f}%", log_fh)
        return True

    cmd = [
        PYTHON, BENCH_PY,
        "--category", target,
        "--models",   model_key,
        "--modes",    "think_off",
        "--source",   str(src),
        "--out",      str(out_path),
    ]
    tlog(f"CMD: {' '.join(cmd)}", log_fh)

    if dry_run:
        return True

    MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r  = subprocess.run(cmd, cwd=str(GEMMA4))
    elapsed = time.time() - t0

    s   = score_jsonl(out_path)
    acc = s["correct"] / s["n"] * 100 if s["n"] else 0
    status = "OK" if r.returncode == 0 else f"ERROR rc={r.returncode}"
    tlog(f"DONE  {status}  {s['correct']}/{s['n']}={acc:.1f}%  loops={s['loops']}  ({elapsed/60:.1f} min)", log_fh)
    return r.returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing.")
    parser.add_argument("--only", nargs="+", metavar="DOMAIN",
                        help="Restrict to specific target domain(s).")
    args = parser.parse_args()

    pairings = PAIRINGS
    if args.only:
        only_set = set(args.only)
        pairings = [(t, m, l) for t, m, l in pairings if t in only_set]
        print(f"[INFO] Filtered to targets: {sorted(only_set)} — {len(pairings)} runs")

    total = len(pairings)
    print(f"[INFO] Stage-2 near/far bench — {total} runs  split={SPLIT}")
    print(f"[INFO] dry-run: {args.dry_run}\n")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    n_ok = n_skip = n_err = 0

    with open(str(LOG_PATH), "a", encoding="utf-8") as log_fh:
        tlog(f"=== Stage-2 near/far bench started  runs={total}  dry={args.dry_run} ===", log_fh)

        for i, (target, model_key, label) in enumerate(pairings, 1):
            visitor   = visitor_short(model_key)
            out_fname = f"bench_merged_{visitor}_on_{target}_{SPLIT}.jsonl"
            out_path  = MATRIX_DIR / out_fname

            n_src  = count_records(OOD_DIR / f"mmlu_pro_ood_{target}_{SPLIT}.jsonl")
            n_done = count_records(out_path)

            tlog(f"--- [{i:02d}/{total}] {label.upper():4s}  {visitor} → {target}  "
                 f"done={n_done}/{n_src}  out={out_fname}", log_fh)

            ok = run_one(target, model_key, out_path, args.dry_run, log_fh)
            if ok:
                n_after = count_records(out_path)
                if n_after >= n_src > 0 and n_done >= n_src > 0:
                    n_skip += 1
                else:
                    n_ok += 1
            else:
                n_err += 1

        tlog(f"=== Stage-2 done  ok={n_ok}  skipped={n_skip}  errors={n_err} ===", log_fh)

    print(f"\n{'='*60}")
    print(f"  Completed: {n_ok}   Skipped: {n_skip}   Errors: {n_err}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
