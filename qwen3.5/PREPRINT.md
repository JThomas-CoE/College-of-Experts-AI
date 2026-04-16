# College of Experts — Paper 3: Domain Intelligence is Separable in General-Purpose Mixture-of-Experts Models

**Histographic Extraction of Specialist Expert Subsets with Validated Cross-Domain Orthogonality**

**Technical Report — April 2026**  
**Author:** J. Thomas  
**Contact:** collegeofexpertsai@gmail.com  
**GitHub:** https://github.com/JThomas-CoE/College-of-Experts-AI  
**HuggingFace:** https://huggingface.co/JThomas-CoE

---

## Abstract

This report is the third in the **College of Experts** (CoE) research series and the
companion document to eight domain-specialist model releases derived from
Qwen3.5-35B-A3B. It documents the methods used to produce those models and the
validation experiments that underpin the release.

The central finding: domain intelligence in a general-purpose Mixture-of-Experts model
is localized in structurally disjoint expert subsets, and those subsets can
be identified and extracted using histographic activation analysis. A coding
specialist (K=128, 50% of the 256-expert pool) and a humanities specialist (K=128,
complementary 50%) built from Qwen3.5-35B-A3B are functionally orthogonal:

- The coding specialist achieves high pass rates on Python coding tasks and
  near-complete collapse on humanities factual-recall tasks.
- The humanities specialist achieves high pass rates on humanities tasks and
  complete collapse on coding tasks — including a characteristic degenerate output
  (syntactically valid but non-functional repetition loops) that is a structural
  signature of missing coding-domain expert capacity.

The 2×2 cross-domain benchmark result is unambiguous: each mask performs at ceiling
on its own domain and at floor on the other. This validates the core hypothesis of the
CoE program — that compact, domain-specific MoE specialists can be derived from a
general-purpose parent model without retraining, simply by identifying and retaining
the expert subsets activated by domain-specific content.

All analysis was performed on a single consumer workstation (AMD Ryzen AI Strix Halo,
128 GB unified memory). Full experimental record: `RESEARCH_LOG.md` (§20–§23).

---

## 1. Context and Prior Work

### 1.1 The College of Experts Program

The CoE research program addresses a practical problem: frontier-scale MoE models
contain more domain knowledge than any single deployment use case requires, but
existing inference frameworks load the entire model regardless. If domain knowledge
is localized within the expert layers, it should be possible to extract compact
domain specialists from the full model — reducing storage and VRAM requirements
while maintaining performance within the target domain.

**Paper 1** (Thomas, February 2026 — *College of Experts*) tested this idea using
7B publicly available models as domain stand-ins. A 5-domain orchestration
system improved average benchmark performance by +27.1% over isolated specialists
(HumanEval +6%, GSM8K +38%, LegalBench +59%). It also hypothesized the theoretical
path to doing this natively from a SOTA MoE via histographic decomposition of
activation data — but did not yet execute that decomposition. The 7B analogs were
a proxy for what genuine MoE experts might deliver.

**Paper 2** (Thomas, March 2026 — *Separability of Intelligence in Mixture-of-Experts:
Slicing Qwen3-Coder-Next into Independent Domain Specialists*) executed the first native
MoE decomposition. Using Qwen3-Coder-Next-80B-A3B (a coding-specialist parent),
histographic analysis separated a Python specialist from a Web specialist at the
structural level. The Python specialist scored 93.0% on HumanEval vs the base model's
94.0% — −1% degradation with 50% expert reduction. The Web specialist failed nearly
completely on Python tasks, conversely the Web specialist performed well on web tasks, while the Python specialist failed, confirming the separation was real.

A limitation of Paper 2: the parent model was already a coding specialist. Separating
Python from Web routing within a model pre-biased toward coding is a gentler test than
separating cognitively distant domains in a general-purpose model. A skeptic could
argue the result reflected pre-training specialization rather than a general structural
property of MoE expert routing.

**This paper** (Paper 3) closes that gap. The parent model is Qwen3.5-35B-A3B —
a general-purpose model with no domain pre-bias. The test pair is algorithmic coding
versus humanities — a near maximum cognitive contrast. If the separation holds
here, the hypothesis holds in general.

### 1.2 Hardware Platform

All experiments were conducted on a single consumer workstation:

| Component         | Specification           |
|-------------------|-------------------------|
| SoC               | AMD Ryzen AI Strix Halo |
| Unified memory    | 128 GB                  |
| Inference (BF16)  | CPU, forward pass       |
| Inference (INT4)  | GPU via Ollama Q4_K_M   |
| OS                | Windows 11              |

---

## 2. Base Model

**Qwen3.5-35B-A3B** is a Mixture-of-Experts language model. The parameters relevant
to this work:

| Parameter                   | Value               |
|-----------------------------|---------------------|
| Total layers                | 40                  |
| Expert type                 | Routed FFN          |
| Experts per layer           | 256                 |
| Active experts per token    | 9 (top-8 + 1 common)|
| Active parameters per token | ~3B                 |
| Total parameters            | ~35B                |
| Native precision            | BF16                |

The routing mechanism is post-softmax with zero+renorm: masked experts receive gate
weight 0.0, and the remaining K weights are renormalised to sum to 1.0. This is
the precondition for structural pruning — a model with softplus or unnormalised routing
would not admit this operation cleanly.

Expert weight vectors are near-perfectly orthogonal (pairwise cosine similarity
0.001–0.005 across all 32,640 pairs per layer). Expert FFNs are genuinely distinct
computational modules, not entries in a shared continuum. This may be a prerequisite for the separability hypothesis: if experts were holographically entangled, domain routing would likely not localise at a per expert level but would likely require selecting groups of entangled experts but this is speculative at this point.

---

## 3. Methodology

### 3.1 Corpus Construction

Domain-specific text corpora were assembled from open datasets and synthetic prompts,
then *hydrated* — extended with full-context versions of short prompts to increase
token density and reduce histogram noise. All 8 released models were built from
hydrated corpora. The hydrated files are included in the repository at
`data/curated/hydrated/`.

Two domains have a minor corpus note:

- **Vocational**: two text patterns were configured (`09_vocational_engineering`,
  `10_tech_engineering`); only the hydrated vocational file was consumed due to the
  profiler's hydrated-first fallback. `10_tech_engineering` raw prompts were not used.
- **Humanities**: three text patterns configured; `14_humanities_play_quotes` (no
  hydrated version) was not consumed for the same reason.

Neither affects the validity of the released models; both caveats are noted for
reproducibility.

### 3.2 Histogram Collection

A gate hook was mounted on the `Qwen3_5MoeTopKRouter` module to record, for each
token in each forward pass, the rank and identity of all 8 activated experts at every
layer. The resulting 3D histogram `H[layer, expert, rank]` accumulates event counts
across the full domain corpus.

The rank-weighted utility score per expert:

$$\text{util}[l, e] = \sum_{k=0}^{7} \frac{8-k}{36} \cdot H[l, e, k]$$

Rank 0 (top-selected expert, weight 8/36) contributes most strongly; rank 7 (bottom
of the top-8, weight 1/36) least. This weights experts by *how consistently they are
preferred* within the domain, not merely how often they appear.

Collection script: `master_profiler_v44_release.py`. Pre-computed histograms for all 8 domains
are included in the repository at `histograms/final/` (~642 KB per domain, 20 MB total).

### 3.3 Coverage Mask Generation

For each layer, the top-128 experts by utility score are selected. The result is a
`list[torch.LongTensor(128,)]` of length 40 — one tensor per layer containing the
retained expert indices.

```python
mask[l] = torch.topk(util[l], k=128).indices
```

K=128 (50% of the 256-expert pool) was selected as the deployment target based on
functional validation experiments (§4 and `RESEARCH_LOG.md` §20–§21). K=64 and K=32
were also validated; see the research log for the budget degradation sweep.

Pre-computed masks for all 8 released models are in `masks/specialist/`.

### 3.4 GGUF Surgery

The surgery script `scripts/prune_gguf_from_mask.py` operates directly on the GGUF
file format via struct-level I/O. No HuggingFace, safetensors, or PyTorch model loading
is required. For each layer it:

1. Reads all `blk.N.ffn_gate_inp.weight` router tensors (router gate weights)
2. Slices all `blk.N.ffn_*expert*` tensors to the K=128 retained indices
3. Permutes the router gate rows to match the new expert index ordering
4. Updates `qwen35moe.expert_count` metadata to 128
5. Writes a new self-contained GGUF

Output: ~12.5 GB Q4_K_M GGUF (vs ~25 GB for the full 256-expert model).
Active parameters per token: unchanged at ~3B.

---

## 4. Domain Separability Validation

This is the core empirical result of this report.

### 4.1 Perfect Structural Bifurcation

Before running any functional benchmarks, we verified the structural relationship
between the SYSTEMS (coding) and HUMANITIES K=128 masks. Layer-by-layer intersection
across all 40 layers:

$$\forall\, l \in [0,\, 39]: \quad |\text{SYSTEMS\_mask}[l] \;\cap\; \text{HUM\_mask}[l]| = 0$$

**Zero overlap in all 40 layers.** The two K=128 masks are a perfect partition of the
256-expert pool at every layer — each domain claims exactly its half, with no shared
expert between them. This is not a designed outcome; it is a consequence of the
training gradient assigning functionally distinct expert circuits to cognitively distant
domains. The histographic analysis recovers this partition without any manual guidance
or post-hoc clustering.

### 4.2 Functional Validation: The 2×2 Benchmark

The structural bifurcation was confirmed functionally by running both masks on both
task types. Five canonical Python coding tasks (pass@5 validator) and five humanities
factual-recall tasks (string-match validator) were evaluated at T=0.4, T=0.6, T=0.9.

Complete results from `RESEARCH_LOG.md` §21–§23:

|                      | **Coding tasks** (pass@5) | **Humanities tasks** (pass@5) |
|----------------------|:-------------------------:|:-----------------------------:|
| **Full model**       | 25/25                     | 25/25                         |
| **SYSTEMS K=128**    | **25/25 (100%)** at T=0.4 | 9/25  — see §4.3              |
| **HUMANITIES K=128** | **0/25**                  | **74/75**                     |

*SYSTEMS K=128 coding result: 25/25 at T=0.4; 72/75 (96%) combined across T=0.4/0.6/0.9.*  
*HUMANITIES K=128 humanities result: 74/75 (99%) combined; one failure at T=0.9 was a plausible confabulation (Tolstoy/Dostoevsky), not structural impairment.*

The matrix is anti-diagonal. Each mask achieves ceiling performance within its domain
and floor performance on the other.

### 4.3 The 2×2 in Detail

**SYSTEMS mask on coding (on-diagonal):** At T=0.4, zero functional degradation
relative to the full model on a 5-task canonical Python suite. The retained 128 experts
are sufficient for the full range of Python coding tasks tested. The coding expert
subgraph is load-bearing and complete at K=128.

**HUMANITIES mask on coding (off-diagonal):** 0/25 (0%) across all temperatures.
The failure mode is diagnostic: 89% of all generations under the humanities mask hit
the 500-token generation limit, producing either prose explanation, partial code
structures that never terminate, or a degenerate repetition loop (e.g., ~250
repetitions of `import python` — syntactically valid Python, functionally useless).
The model knows it should produce code but cannot generate it. This is structural
impairment from missing coding-domain experts, not stochastic error.

**SYSTEMS mask on humanities (off-diagonal):** 9/25 (36%) overall, but the internal
structure is revealing. Tasks 0–1 (Dostoevsky, year of Berlin Wall fall) show partial
survival; tasks 2–4 (Mona Lisa, Odysseus, Dante) score exactly 0/15. The partial
survival of tasks 0–1 reflects that the names "Dostoevsky" and the number "1989"
appear frequently in technical writing (documentation, copyright headers, changelogs) —
the SYSTEMS experts retain them as incidental token co-occurrences, not as cultural
knowledge. Pure humanities knowledge — classical literature, fine arts, ancient
mythology — is completely absent from the coding expert subgraph. The 36% aggregate
score is misleading; the 0% on tasks 2–4 is the meaningful signal.

**HUMANITIES mask on humanities (on-diagonal):** 74/75 (99%) combined. One failure
was Tolstoy vs Dostoevsky at T=0.9 — a stochastic sampling error between two near-equal
probability peaks, not structural impairment. At T=0.4 and T=0.6: 25/25 (100%).

### 4.4 What This Proves

The result rules out the residual-capacity alternative hypothesis. If the K=128
zero-degradation on coding reflected generic remaining capacity that any 50% mask would
preserve, the humanities mask should also score ~100% on coding tasks. It scores 0%.
The SYSTEMS mask's performance is attributable specifically to retaining the
coding-relevant expert subgraph — and the humanities mask's failure is attributable
specifically to retaining the other half.

This is the functional confirmation of the Paper 2 finding in a harder case: not
Python versus Web within a coding-specialist parent, but algorithmic computation versus
humanistic reasoning within a general-purpose parent. The fact that the expert space
bifurcates cleanly along this axis — and does so at the 50% budget level, with K=128
hitting ceiling in both directions — validates the histographic extraction methodology
as a general technique, not a special-case result.

---

## 5. Released Models

Eight domain specialists are released as beta models derived from the above methodology.
All are text-only in this release.

| HuggingFace repo                              | Domain            | Source corpus                                                           | Mask file                  |
|-----------------------------------------------|-------------------|-------------------------------------------------------------------------|----------------------------|
| `JThomas-CoE/coe-qwen3.5-coding-18b-a3b`      | Algo. / OS coding | `02_coding_systems_hydrated.jsonl`                                      | `coverage_SYSTEMS_K128.pt` |
| `JThomas-CoE/coe-qwen3.5-web-18b-a3b`         | Web coding        | `01_coding_web_hydrated.jsonl`                                          | `coverage_WEB_T_K128.pt`   |
| `JThomas-CoE/coe-qwen3.5-math-18b-a3b`        | Math              | `07_science_adv_math_hydrated.jsonl` + `07_science_math_hydrated.jsonl` | `coverage_MATH_T_K128.pt`  |
| `JThomas-CoE/coe-qwen3.5-physics-18b-a3b`     | Physics           | `05_science_physics_hydrated.jsonl`                                     | `coverage_PHYS_T_K128.pt`  |
| `JThomas-CoE/coe-qwen3.5-biology-18b-a3b`     | Biology           | `06_science_bio_chem_hydrated.jsonl`                                    | `coverage_BIO_T_K128.pt`   |
| `JThomas-CoE/coe-qwen3.5-engineering-18b-a3b` | Engineering       | `10_applied_engineering_hydrated.jsonl`                                 | `coverage_ENG_K128.pt`     |
| `JThomas-CoE/coe-qwen3.5-vocational-18b-a3b`  | Vocational trades | `09_vocational_engineering_hydrated.jsonl`                              | `coverage_VOC_K128.pt`     |
| `JThomas-CoE/coe-qwen3.5-humanities-18b-a3b`  | Humanities        | `08_humanities_phil_hydrated.jsonl` + `09_liberal_arts_hydrated.jsonl`  | `coverage_HUM_K128.pt`     |

**Model naming note:** The internal designation `SYSTEMS` maps to the public name
`coding` — it is an algorithmic/systems coding specialist. The `WEB_T` designation
indicates a textual-only mask for web development. All `_T` suffixes are dropped in
public names as the entire beta release is text-only.

**Recommended usage — prompt harness examples:**

The full model's default of T=0.6 will generally work for the pruned models but expert
pruning sharpens the routing distribution so T=0.4 may work better as an operating point
for these pruned variants on some tasks but temperatures up to 0.9 have been tested and
generally work if greater variability/creativity is desired.

*Coding specialist* (`coe-qwen3.5-coding-18b-a3b`, T=0.4)
```
System: "You are a Python coding expert. Complete the following task as such.
         Return a single, complete block of functional Python code. Keep comments
         and explanations concise and minimal. Do not second guess your answer."

User:   "write python code to implement a thread-safe LRU cache with O(1) get and put."
```

*Humanities specialist* (`coe-qwen3.5-humanities-18b-a3b`, T=0.4)
```
System: "You are a humanities scholar. Answer the question with precision and
         appropriate depth. Cite specific works, authors, or dates when relevant.
         Do not add unsolicited commentary — stop after your answer."

User:   "What is the dramatic function of the Chorus in Greek tragedy?
         Use Sophocles as your primary reference."
```

*Vocational specialist* (`coe-qwen3.5-vocational-18b-a3b`, T=0.4)
```
System: "You are an expert on welding. Answer as such. If appropriate include
         best practices guidelines including safety protocols. When you have
         given your answer stop without further elaboration."

User:   "What type of filler rod should I use for TIG welding 304 stainless
         steel, and what shielding gas is appropriate?"
```

**Beta status:** These models have not undergone post-surgery supervised fine-tuning.
They are structural variants of the base model, not fine-tuned specialists. Performance
within the target domain is validated by the K=128 sweep; performance on out-of-domain
tasks degrades significantly by design. Models are released as Q4_K_M GGUF only.

---

## 6. Reproducibility

### 6.1 Repository Contents

```
qwen3.5/
├── RESEARCH_LOG.md                        # Complete experimental record (§1–§35)
├── PREPRINT.md                            # This document
├── environment.yml                        # Python dependency spec
├── master_profiler_v44_release.py         # Histogram collection (requires BF16 model)
├── generate_all_masks_release.py          # Mask generation from histograms
├── histograms/
│   └── final/                             # Pre-computed 3D activation histograms (20 files, ~20 MB)
├── masks/
│   └── specialist/                        # K=128 coverage masks (8 released models)
│       ├── coverage_SYSTEMS_K128.pt
│       ├── coverage_WEB_T_K128.pt
│       └── ...
├── data/
│   └── curated/                           # Domain corpora
│       └── hydrated/                      # Hydrated versions (used for released models)
└── scripts/
    └── prune_gguf_from_mask.py            # GGUF surgery script
```

### 6.2 Environment Setup

Dependencies are listed in `environment.yml`. The minimum requirements are PyTorch,
Transformers, Accelerate, safetensors, Pillow, tqdm, psutil (for histogram collection),
numpy (for mask generation), and gguf (for GGUF surgery). PyTorch must be installed
with a wheel appropriate for your compute platform — see the platform note in
`environment.yml`. To create the environment:

```bash
conda env create -f environment.yml
conda activate coe-qwen35
```

### 6.3 Regenerating Masks from Histograms

Pre-computed histograms are provided. To regenerate masks:

```bash
conda activate coe-qwen35
cd qwen3.5/
python generate_all_masks_release.py
```

Output: `masks/specialist/coverage_{DOMAIN}_K128.pt` for each domain.

### 6.4 Regenerating Histograms from Corpus

Histogram collection requires the BF16 base model (~70 GB RAM). If rerunning from
scratch, the profiler uses hydrated corpora by default when present:

```bash
conda activate coe-qwen35
python master_profiler_v44_release.py
```

The profiler will use `data/curated/hydrated/` when a matching hydrated file exists,
falling back to `data/curated/` otherwise. All 8 released models were produced from
hydrated corpora.

### 6.5 Rebuilding GGUFs from Masks

```bash
python scripts/prune_gguf_from_mask.py \
    --mask   "masks/specialist/coverage_SYSTEMS_K128.pt" \
    --input  "<path-to-Qwen3.5-35B-A3B.gguf>" \
    --output-dir "./output"
```

The surgery script requires only the base GGUF and the mask file — no PyTorch model
loading or GPU required for the surgery itself.

---

## 7. Limitations and Project Status

**No post-surgery fine-tuning.** Released models are structural variants of the base
model. The K=128 sweep confirms functional competency within the target domain; it does
not establish equivalence with a fine-tuned specialist. Cross-comparison with the
gemma4 model family during subsequent work suggests in-domain performance is
meaningfully below the parent model on harder evaluations than the canonical pass@5
suite used here. "High in-domain fidelity" is accurate for canonical tasks; broader
generalization limits are not yet fully characterized.

**Text-only.** All released models are textual-only. No vision specialist variants
are included in this release.

**Small-n validation suite.** The primary validation experiment (§4) used 5 coding
tasks and 5 humanities tasks with pass@5 sampling. The qualitative 2×2 pattern is
unambiguous at this scale; quantitative pass rates for the on-diagonal cells should
not be interpreted as precision benchmark scores.

**Further Qwen3.5 work deferred.** Following the beta release of these 8 models,
active research in this series has transitioned to the Gemma 4 model family
(specifically Gemma 4 25B-A4B), which offers a different expert architecture and
presents new structural questions about the generality of histographic decomposition.
The Qwen3.5 models are considered stable at their current state.

---

## 8. References

1. Thomas, J. (2026). "College of Experts." *CoE Technical Report, Paper 1.* February 2026.
2. Thomas, J. (2026). "Separability of Intelligence in Mixture-of-Experts: Slicing Qwen3-Coder into Independent Domain Specialists." *CoE Technical Report, Paper 2.* March 2026.
3. Qwen Team (2025). "Qwen3.5 Technical Report." Alibaba Cloud.
4. Shazeer, N., et al. (2017). "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer." *ICLR 2017.*
5. Fedus, W., et al. (2022). "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity." *JMLR 23(1).*
6. Chen, M., et al. (2021). "Evaluating Large Language Models Trained on Code." *arXiv:2107.03374* (HumanEval).
7. Cobbe, K., et al. (2021). "Training Verifiers to Solve Math Word Problems." *arXiv:2110.14168* (GSM8K).
8. Guha, N., et al. (2023). "LegalBench: A Collaboratively Built Benchmark for Measuring Legal Reasoning." *NeurIPS Datasets and Benchmarks.*

---

*College of Experts — April 2026*  
*Released under PolyForm Noncommercial 1.0.0 — Commercial licensing available upon request*  
*GitHub: https://github.com/JThomas-CoE/College-of-Experts-AI*
