# Gemma4 College of Experts — MMLU-Pro Automated Pipeline

**Base model:** [google/gemma-4-26b-it](https://huggingface.co/google/gemma-4-26b-it)  
**Architecture:** MoE — 26B total / ≈4B active (1 shared + 8 routed from 128 experts per layer, 30 MoE layers)  
**Method:** Activation-directed expert surgery — 128 → 64 experts per layer (50% reduction, zero gradient steps)  
**Released models:** [JThomas-CoE on HuggingFace](https://huggingface.co/JThomas-CoE)

This directory contains the pipeline scripts, expert masks, and analysis code for the **MMLU-Pro automated domain specialist** batch. The central claim: domain-relevant expert populations can be identified using existing public benchmark questions alone, making the surgery pipeline fully automatable — no hand-curated domain corpus required.

---

## Memory Efficiency

| Configuration | VRAM (16k ctx) | VRAM (64k ctx) | Active params |
|---|---|---|---|
| Gemma4-26B parent (Q4_K_M) | 19.4 GB | 20.5 GB | ≈4B |
| Specialist K=64 (Q4_K_M) | **12.3 GB** | **13.4 GB** | ≈4B |
| Savings vs parent | **−7.1 GB (37%)** | **−7.1 GB (35%)** | unchanged |

All figures directly measured in Ollama. Throughput is identical — 9 experts fire per token in both cases.

---

## Quick Start

All models require **think-off mode**. With the Ollama API:

```python
import requests, json

response = requests.post("http://localhost:11434/api/chat", json={
    "model": "coe-gemma4-physics-mmlu_pro-14b-a4b:q4",
    "think": False,
    "options": {"temperature": 0.6, "num_ctx": 16384},
    "messages": [{"role": "user", "content": "Your question here."}]
})
```

With a raw llama.cpp-compatible API, inject a closed think block as an assistant prefill:

```python
messages = [
    {"role": "system",    "content": "You are an expert physics practitioner."},
    {"role": "user",      "content": "Your question here."},
    {"role": "assistant", "content": "<think></think>\n"},   # required
]
```

**Why think-off is required:** loop-suppression experts are not domain-concentrated and may be pruned. In think-on mode this causes reasoning loops that exhaust the token budget. Temperature ≥ 1.0 also materially increases loop rates — use T = 0.6.

**Available domains:** `math`, `physics`, `chemistry`, `engineering`, `biology`, `economics`, `business`, `psychology`, `cs`, `law`

---

## Pipeline

Four scripts, run in order. Each produces a file consumed by the next.

### Data Acquisition — MMLU-Pro Subsets

Questions are drawn from [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) (public, HuggingFace Hub):

```python
from datasets import load_dataset
ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
```

**1-in-10 offset (off-K) formalism:** Questions within each domain are indexed 0 … N−1. Subset **off-K** contains every question whose zero-based position satisfies `i % 10 == K`. This partitions each domain into ten non-overlapping ≈10% subsets: **off0 … off9**.

| Subset | Role |
|---|---|
| **off4** | Profiling corpus — run through parent model (Step 1), then activation-profiled (Step 2) |
| **off9** | Second profiling corpus — identical pipeline, kept separate for independent verification before merge |
| **off5** | OOD evaluation — **never used in profiling**; all home-domain accuracy numbers in this README use off5 |

The off4 and off9 histograms are **merged (summed)** before Step 3 mask construction, doubling the corpus token count and reducing variance in the utility score estimates. Released masks are named `_merged_` to reflect this.

### Step 1 — Corpus generation

```
python generate_domain_corpus_gemma4.py --domain physics
```

Queries the parent model (via Ollama) with MMLU-Pro questions, saves Q+A pairs and think traces to `data/<domain>_qa_generated.jsonl`.

### Step 2 — MoE activation profiling

```
python profile_domains_gemma4.py \
    --output-pt histograms/physics_results.pt \
    --domain physics \
    data/physics_qa_generated.jsonl question+answer \
    --model-path /path/to/gemma-4-26B-A4B-it
```

Attaches router forward-hooks to the full-precision HF model and accumulates 3D histograms `(layer, expert, rank)` over the corpus. Requires sufficient memory to fit the full precision model but for just the forward pass this can be done in system memory using the cpu with acceptable latency.

### Step 3 — Expert mask construction

```
python build_coe_mask.py \
    --domain physics \
    --pt-file histograms/physics_results.pt \
    --output-name physics_mmlu_merged \
    --budget 64 \
    --cot-pt-file data/cot_think_results.pt \
    --cot-domain cot_think \
    --no-pass3
```

Three-pass mask build: (1) top-64 by utility score, (2) structural whitelist (avg\_rank ≤ 2.0, count ≥ 10), (3) CoT arbitrage (up to 6 swaps per layer from `data/cot_think_results.pt`). Output: `mask_coe_physics_mmlu_merged_swapactive_K64.json`.

### Step 4 — GGUF surgery

```
python surgery_gemma4.py \
    --mask mask_coe_physics_mmlu_merged_swapactive_K64.json \
    --src /path/to/gemma4-26B-A4B-Q4_K_M.gguf \
    --out coe_gemma4_physics_mmlu_K64.gguf
```

Removes `ffn_gate`, `ffn_up`, `ffn_down` tensors for non-mask experts. Attention, embeddings, and the shared expert are untouched. Load the output into Ollama:

```
ollama create coe-gemma4-physics-mmlu_pro-14b-a4b:q4 -f Modelfile.physics-mmlu-K64
```

For the remaining domains substitute the domain name in both the tag and the Modelfile arguments wherever 'physics' appears in the example case.

---

## Home-Domain Accuracy

MMLU-Pro OOD off5 split, Q4_K_M, think\_off mode.

| Domain | Parent | Specialist | Δ | n | Released |
|---|---|---|---|---|---|
| Math | 94.1% | **94.3%** | +0.2 pp | ~135 | ✅ |
| Chemistry | 88.1% | **88.5%** | +0.4 pp | ~113 | ✅ |
| Biology | 89.9% | 87.5% | −2.4 pp | ~72 | ✅ |
| Economics | 87.9% | 84.6% | −3.3 pp | ~84 | ✅ |
| Business | 87.3% | 83.9% | −3.4 pp | ~79 | ✅ |
| Physics | 87.3% | 84.0% | −3.3 pp | ~130 | ✅ |
| Psychology | 83.1% | 79.0% | −4.1 pp | ~80 | ✅ |
| CS | 82.9% | **81.1%** | −1.8 pp | ~41 | ✅ |
| Engineering | 72.6% | 71.1% | −1.5 pp | ~97 | ✅ |
| Law | 62.3% | 58.9% | −3.4 pp | ~110 | ✅ |

At n ≈ 80–135, binomial SE ≈ ±4–5 pp. No released specialist falls more than 1σ below its parent on the home domain.

Four domains were evaluated but not released: Health (−32.9 pp, ~8σ), Other (−19.4 pp, ~4σ), Philosophy (−14.0 pp, ~2σ), and History (−2.6 pp, within noise but small n and domain coherence concerns). See [HF_README_coe-gemma4-mmlu_pro_batch.md](HF_README_coe-gemma4-mmlu_pro_batch.md) for the full analysis.

---

## Near/Far Transfer — Semantic Localization

To test whether expert surgery produces genuine semantic localization, we ran a 20-pair cross-domain transfer experiment. Each pair sends one specialist to answer questions on a different domain's benchmark. Transfer efficiency Y = specialist accuracy / parent accuracy on the same questions (Y = 1.0 means full parent-level recovery).

**Key result:** Every NEAR pair outperforms its FAR counterpart. The near/far gap never reverses.

| # | Arm | Visitor → Target | n | Y=Acc/Par | F-score | Agg-cos |
|---|---|---|---|---|---|---|
| 01 | NEAR | physics → math | 135 | 0.945 | 0.787 | 0.764 |
| 02 | FAR | law → math | 135 | 0.197 | 0.485 | 0.440 |
| 03 | NEAR | chemistry → physics | 130 | 0.952 | 0.855 | 0.842 |
| 04 | FAR | law → physics | 130 | 0.115 | 0.448 | 0.396 |
| 05 | NEAR | physics → engineering | 97 | 0.880 | 0.854 | 0.841 |
| 06 | FAR | law → engineering | 97 | 0.170 | 0.421 | 0.367 |
| 07 | NEAR | math → cs | 41 | **1.089** | 0.723 | 0.700 |
| 08 | FAR | law → cs | 41 | 0.618 | 0.485 | 0.437 |
| 09 | NEAR | physics → chemistry | 113 | 0.844 | 0.855 | 0.842 |
| 10 | FAR | law → chemistry | 113 | 0.151 | 0.413 | 0.363 |
| 11 | NEAR | psychology → biology | 72 | 0.834 | 0.749 | 0.721 |
| 12 | FAR | business → biology | 72 | 0.603 | 0.576 | 0.534 |
| 13 | NEAR | psychology → economics | 84 | 0.745 | 0.739 | 0.713 |
| 14 | FAR | chemistry → economics | 84 | 0.745 | 0.571 | 0.535 |
| 15 | NEAR | math → business | 79 | 0.884 | 0.756 | 0.732 |
| 16 | FAR | law → business | 79 | 0.159 | 0.541 | 0.499 |
| 17 | NEAR | biology → psychology | 80 | 0.782 | 0.749 | 0.721 |
| 18 | FAR | engineering → psychology | 80 | 0.752 | 0.486 | 0.436 |
| 19 | NEAR | economics → law | 110 | 0.730 | 0.701 | 0.676 |
| 20 | FAR | chemistry → law | 110 | 0.540 | 0.413 | 0.363 |

**Correlation with Y** (n = 20): F-score r = 0.800, Agg-cos r = 0.791. Best OLS model (Agg-cos + layer variance + normalized specialist quality): R² = 0.796, adj-R² = 0.757, p = 1.28 × 10⁻⁷.

![Transfer Efficiency Scatter](plot_transfer_efficiency_g39.png)

---

## Repository Contents

| File | Purpose |
|---|---|
| `generate_domain_corpus_gemma4.py` | Step 1 — Q+A corpus from MMLU-Pro via Ollama |
| `profile_domains_gemma4.py` | Step 2 — 3D router activation histograms |
| `build_coe_mask.py` | Step 3 — Three-pass expert mask construction |
| `surgery_gemma4.py` | Step 4 — GGUF expert surgery |
| `bench_mmlu_pro.py` | MMLU-Pro benchmark runner |
| `bench_mmlu_pro_calc.py` | Calculator-augmented benchmark runner |
| `_run_pairwise_cross_domain_bench.py` | Near/far transfer benchmark orchestration |
| `compute_param_similarity.py` | Agg-cos similarity between mask activation profiles |
| `_compute_residual_similarity.py` | F-score set-overlap similarity between masks |
| `_analyze_transfer_correlations.py` | Pearson r / OLS regression on transfer results |
| `plot_pairwise_scatter.py` | Scatter: residual F-score vs transfer gap |
| `plot_transfer_efficiency_g39.py` | Two-panel transfer efficiency figure |
| `data/cot_think_results.pt` | CoT/reasoning activation profile (Pass 3 reference) |
| `data/param_similarity.json` | Precomputed Agg-cos similarity matrix |
| `data/residual_similarity.json` | Precomputed F-score similarity matrix |
| `mask_coe_<domain>_mmlu_merged_swapactive_K64.json` | Expert mask for each released domain (10 files) |
| `Modelfile.<domain>-mmlu-K64` | Ollama Modelfile for each domain (10 files) |
| `HF_README_coe-gemma4-mmlu_pro_batch.md` | Full methodology, benchmarks, and HF model card |

---

## Full Documentation

[HF_README_coe-gemma4-mmlu_pro_batch.md](HF_README_coe-gemma4-mmlu_pro_batch.md) contains the complete methodology writeup including: utility scoring derivation, mask construction pass details, mask similarity vs hand-curated pipeline (Jaccard 0.818–0.826), MATH-500 and AIME 2026 benchmarks, custom physics derivation benchmark, deployment context, and known limitations.

---

## License

Code and tooling: PolyForm Noncommercial 1.0.0  
Model weights: subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms)  
Commercial licensing: [LICENSE-COMMERCIAL.md](../LICENSE-COMMERCIAL.md)
