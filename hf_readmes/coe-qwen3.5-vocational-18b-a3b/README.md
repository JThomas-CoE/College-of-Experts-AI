---
language:
- en
license: other
license_name: qwen3.5-research
tags:
- mixture-of-experts
- qwen3.5
- college-of-experts
- pruned
- specialist
- gguf
- ollama
base_model: Qwen/Qwen3.5-35B-A3B
---

# CoE-Qwen3.5-Vocational-Trades-18B-A3B — College of Experts Specialist · Beta

> **⚠ Beta Release.** This model has undergone GGUF-level expert surgery but has received
> no post-surgery supervised fine-tuning. Task performance within the target domain is
> validated (see §Validation below), but edge-case behaviour may differ from the full base
> model. Use with the recommended prompt harness and temperature settings.

**Base model:** [Qwen/Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)  
**Parameter count:** ~18B total · 3B active/token  
**Quantization:** Q4\_K\_M  
**Format:** GGUF (Ollama / llama.cpp compatible)  
**Modality:** Text only (vision not supported in this release)  
**Domain:** Vocational Trades — electrical, plumbing, HVAC, welding, machining, construction

---

## What This Is

Qwen3.5-35B-A3B is a Mixture-of-Experts model with 256 experts per layer, activating 8
per token. This release is a **surgical specialist** — 128 of those 256 experts have been
removed per layer, retaining only those whose activation frequency is highest for
**Vocational Trades** content.

The result is a model with ~half the total parameters (~18B vs 35B), identical active
parameters per token (3B), and meaningfully concentrated domain routing.

This is one of 8 domain specialists in the
[College of Experts](https://github.com/JThomas-CoE/College-of-Experts-AI) beta release.

---

## Surgery Methodology

Expert selection uses a **coverage mask** derived from 3D activation histograms collected
over a domain-specific text corpus.

**Utilisation score per expert:**

$$\text{util}[l, e] = \sum_{k=0}^{7} \frac{8-k}{36} \cdot H[l, e, k]$$

where $H[l, e, k]$ is the count of times expert $e$ in layer $l$ was selected at rank $k$
across the domain corpus. Rank 0 (top-selected) is weighted 8/36; rank 7 contributes 1/36.

**Mask selection:** Top-128 experts by utilisation score per layer, computed from the
textual-only histogram for this domain (textual and visual routing activations were found
to select largely different experts; see §Validation for the T-vs-V separability result).

**GGUF surgery:** `scripts/prune_gguf_from_mask.py` in the GitHub repo operates directly
on the Ollama GGUF blob via struct-level I/O — no safetensors or HuggingFace loading
required. It slices all `blk.N.*ffn_expert*` tensors to the retained 128 indices, permutes
the router gate weight rows to match, and updates `llm.expert_count` metadata to 128.

The retained expert indices for this model are in `masks/coverage_VOC_K128.pt`
(this repo) — a `list[torch.LongTensor(128,)]` of length 40, one per layer.

---

## Validation

### Functional threshold

Expert budget K=128 was validated as the deployment target against the full 256-expert
baseline on **Python coding tasks** (pass@5 at T=0.4):

| Expert budget K | % of pool | pass@5 (T=0.4) |
|:---:|:---:|:---:|
| 256 (baseline) | 100% | 100% |
| **128** | **50%** | **100%** |
| 64 | 25% | 86% |
| 32 | 12.5% | 74% |

K=128 is the minimum budget that fully saturates pass@5 on the validation suite. K=64 and
K=32 show measurable degradation ("cliff" behaviour begins below K=128).

### Domain separability

To confirm that retaining domain-specific experts preserves meaningful structure rather
than degrading uniformly, the bidirectional separability experiment compared the
**SYSTEMS/coding** and **HUMANITIES** K=128 pruned models on each other's tasks:

| Pruned model | Coding tasks (pass@5, T=0.4) | Humanities tasks |
|---|:---:|:---:|
| **SYSTEMS K=128** (this release family) | **100%** | ~4% |
| **HUMANITIES K=128** | ~4% | high |

The 96 percentage-point gap confirms that the retained expert pools are **principled domain
partitions**, not random subsets. A humanities-specialist model fails almost completely on
coding tasks where the coding specialist achieves perfect pass@5 — and vice versa. This is
expected from the routing structure of the base model and validates that surgery preserves
domain-specific computation.

Full methodology, per-layer expert indices, and all experimental results are in the
[RESEARCH\_LOG.md](https://github.com/JThomas-CoE/College-of-Experts-AI) in the GitHub repo
(§21–§23 for the separability experiments; §29 for the full 14-domain Jaccard overlap
analysis).

---

## Quickstart (Ollama)

```bash
# Pull and run
ollama pull JThomas-CoE/coe-qwen3.5-vocational-18b-a3b

ollama run JThomas-CoE/coe-qwen3.5-vocational-18b-a3b
```

Or register directly from this repo's GGUF:

```bash
# Download the GGUF and Modelfile, then:
ollama create coe-qwen3.5-vocational-18b-a3b -f Modelfile
ollama run coe-qwen3.5-vocational-18b-a3b
```

**Recommended temperature:** `T=0.4`  
Expert pruning sharpens the router's logit distribution. Higher temperatures (T≥0.7)
may cause routing instability. T=0.4 was the validated operating point across all
K=128 experiments.

---

## Recommended Prompt Harness

The full model's default of T=0.6 will generally work for the pruned models but expert
pruning sharpens the routing distribution so T=0.4 may work better as an operating
point for these pruned variants on some tasks but temperatures up to 0.9 have been
tested and generally work if greater variability/creativity is desired.

This model performs best with an explicit domain framing in the system prompt. Examples
for each specialist are given below — substitute the example matching this model's domain.

*Coding specialist* (`coe-qwen3.5-coding-18b-a3b`, T=0.4)
```
System: "You are a Python coding expert. Complete the following task as such.
         Return a single, complete block of functional Python code. Keep comments
         and explanations concise and minimal. Do not second guess your answer."

User:   "write python code to implement a thread-safe LRU cache with O(1) get and put."
```

*Web specialist* (`coe-qwen3.5-web-18b-a3b`, T=0.4)
```
System: "You are a web development expert. Answer the following with working code.
         Prefer modern standards and best practices. Add inline comments only where
         the logic is non-obvious. Stop after your answer."

User:   "Create a standalone HTML file for a snake game web app. All CSS and JS must
         be inline. Give the app a retro, dark neon look."
```

*Math specialist* (`coe-qwen3.5-math-18b-a3b`, T=0.4)
```
System: "You are a mathematics expert. Solve the following problem. Show non-trivial
         intermediate steps. State any assumptions. Use standard notation. Stop after
         the solution."

User:   "Find the eigenvalues and eigenvectors of the matrix [[3, 1], [1, 3]]."
```

*Physics specialist* (`coe-qwen3.5-physics-18b-a3b`, T=0.4)
```
System: "You are a physics expert. Answer with precision. Show derivations where
         relevant. Use SI units throughout. Stop after your answer."

User:   "Derive the expression for the period of a simple pendulum in the
         small-angle approximation."
```

*Biology specialist* (`coe-qwen3.5-biology-18b-a3b`, T=0.4)
```
System: "You are a biology expert. Answer with scientific precision. Reference
         specific mechanisms, structures, and established terminology. Do not
         add unsolicited commentary — stop after your answer."

User:   "Explain the role of the sodium-potassium pump in maintaining the
         resting membrane potential of a neuron."
```

*Engineering specialist* (`coe-qwen3.5-engineering-18b-a3b`, T=0.4)
```
System: "You are an engineering expert. Answer with technical precision. Include
         relevant standards, tolerances, or safety considerations where they
         apply. When you have given your answer stop without further elaboration."

User:   "Compare the fatigue life of a notched versus unnotched steel specimen
         under cyclic loading, and explain the mechanism responsible for the
         difference."
```

*Vocational specialist* (`coe-qwen3.5-vocational-18b-a3b`, T=0.4)
```
System: "You are an expert on welding. Answer as such. If appropriate include
         best practices guidelines including safety protocols. When you have
         given your answer stop without further elaboration."

User:   "What type of filler rod should I use for TIG welding 304 stainless
         steel, and what shielding gas is appropriate?"
```

*Humanities specialist* (`coe-qwen3.5-humanities-18b-a3b`, T=0.4)
```
System: "You are a humanities scholar. Answer the question with precision and
         appropriate depth. Cite specific works, authors, or dates when relevant.
         Do not add unsolicited commentary — stop after your answer."

User:   "What is the dramatic function of the Chorus in Greek tragedy?
         Use Sophocles as your primary reference."
```

---

## Limitations / Beta Caveats

- **No post-surgery training.** This model is the output of structural surgery on the
  base model weights. No supervised fine-tuning, RLHF, or DPO has been applied after
  expert removal. Behaviour on unusual prompts may be less robust than the base model.
- **Text only.** Vision inputs are not supported. The `_T` (textual) mask was used,
  which selects experts optimised for text routing only. Visual expert pools differ
  significantly (mean T-vs-V Jaccard ~0.43 for same domain).
- **Q4\_K\_M quantization only.** Full-precision (BF16) weights will be released
  following post-surgery validation across all 8 domains.
- **Recommended context window:** 32k tokens (Modelfile default). Longer contexts
  have not been validated post-surgery.
- **Domain framing recommended.** See prompt harness section above.

---

## Expert Mask

The file `masks/coverage_VOC_K128.pt` in this repo contains the retained expert
indices used to produce this GGUF. Format:

```python
import torch
masks = torch.load("masks/coverage_VOC_K128.pt", weights_only=False)
# masks: list of 40 torch.LongTensor, each shape (128,)
# masks[layer_idx] = 1D tensor of 128 retained expert indices for that layer
print(masks[0])   # expert indices retained in layer 0
```

To reproduce the surgery from the base model GGUF:

```bash
python scripts/prune_gguf_from_mask.py \
    --mask   "masks/coverage_VOC_K128.pt" \
    --input  "<path-to-Qwen3.5-35B-A3B-base.gguf>" \
    --output-dir "./output"
```

Full surgery script and build pipeline: https://github.com/JThomas-CoE/College-of-Experts-AI

---

## License

This model is derived from [Qwen3.5-35B-A3B](https://huggingface.co/Qwen/Qwen3.5-35B-A3B).
The surgical modifications, masks, and College of Experts tooling are released under
**PolyForm Noncommercial 1.0.0**. Commercial licensing available upon request.

The base model weights remain subject to the Qwen3.5 model license.
