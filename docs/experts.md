# Recommended Expert Models (January 2026)

This document catalogs specialized models suitable for use as "experts" in the College of Experts architecture. Models are organized by domain and reflect the current landscape as of early 2026.

> [!IMPORTANT]
> **Disclaimer**: The models listed in this document are provided as **reasonable stand-ins for demonstration purposes**. This is not an exhaustive list, does not imply endorsement of any particular model or organization, and has not been derived from comprehensive benchmarking or testing. These recommendations are intended as starting points for experimentation. As the ecosystem evolves and true purpose-built "college of experts" models become available, this list should be updated accordingly.

## Selection Criteria

Good expert models should:
- Have **clear domain specialization** (trained or fine-tuned on domain data)
- Be available in **quantized formats** (GGUF, AWQ, GPTQ)
- Be **1B-20B parameters** (loadable on consumer hardware in <3s)
- Have **open weights** with permissive licenses

---

## Code & Programming

### Comparison Overview

| Model | Size Sweet Spot | Strengths | Best For |
|-------|-----------------|-----------|----------|
| DeepSeek-Coder-V3 | 33B | Reasoning chains, multi-file refactoring | Complex architecture, deep debugging |
| Devstral | 22B | IDE integration, MCP/tool use | Agentic workflows, tool-calling |
| Qwen3-Coder | 30B | Multilingual, explanation, Apache 2.0 | General coding, education |
| Falcon-3 Coder | 7B | Fast inference, punches above weight | Quick queries, edge/consumer hardware |
| StarCoder3 | 15B | Fill-in-middle, autocomplete | IDE completions, real-time suggestions |

### DeepSeek-Coder-V3

| Property | Value |
|----------|-------|
| Sizes | 7B, 33B, 236B |
| Quantized | GGUF via Ollama, llama.cpp |
| Strengths | Code generation, debugging, multi-language, agentic coding, excellent reasoning chains |
| Source | [DeepSeek](https://huggingface.co/deepseek-ai) |
| License | DeepSeek License (commercial OK) |
| Note | Successor to V2, significantly improved reasoning; excellent for multi-file refactoring |

**Ollama:** `ollama pull deepseek-coder-v3:33b`

### Devstral

| Property | Value |
|----------|-------|
| Sizes | 22B |
| Quantized | GGUF available |
| Strengths | IDE integration, MCP tool use, agentic coding workflows |
| Source | [Mistral](https://huggingface.co/mistralai) |
| License | Mistral Commercial |
| Note | Mistral's dedicated code model; optimized for tool-calling and structured outputs |

**Ollama:** `ollama pull devstral:22b`

### Qwen3-Coder

| Property | Value |
|----------|-------|
| Sizes | 3B, 7B, 14B, 30B, 72B |
| Strengths | Code completion, generation, 100+ languages, long context, excellent explanations |
| Source | [Qwen](https://huggingface.co/Qwen) |
| License | Apache 2.0 |
| Note | 30B offers best quality/speed balance; strong multilingual support |

**Ollama:** `ollama pull qwen3-coder:30b`

### Falcon-3 Coder

| Property | Value |
|----------|-------|
| Sizes | 7B, 40B |
| Quantized | GGUF via Ollama |
| Strengths | Fast inference, strong benchmarks at 7B size class, efficient architecture |
| Source | [TII Abu Dhabi](https://huggingface.co/tiiuae) |
| License | Falcon License (permissive) |
| Note | 7B model punches significantly above its weight; excellent for consumer hardware |

**Ollama:** `ollama pull falcon3-coder:7b`

### StarCoder3

| Property | Value |
|----------|-------|
| Sizes | 3B, 7B, 15B, 33B |
| Strengths | Code completion, fill-in-middle, 600+ languages |
| Source | [BigCode](https://huggingface.co/bigcode) |
| License | BigCode OpenRAIL-M |
| Note | Best-in-class for autocomplete; less conversational, more completion-focused |

### CodeGemma 2

| Property | Value |
|----------|-------|
| Sizes | 2B, 7B, 27B |
| Strengths | Fast inference, code completion, Google ecosystem |
| Source | [Google](https://huggingface.co/google) |
| License | Gemma License |

---


## Mathematics & Reasoning

### Qwen 3 Math

| Property | Value |
|----------|-------|
| Sizes | 3B, 7B, 32B, 72B |
| Strengths | Mathematical reasoning, competition-level problems, proofs |
| Source | [Qwen](https://huggingface.co/Qwen) |
| License | Apache 2.0 |
| Note | State-of-the-art open math model |

**Ollama:** `ollama pull qwen3-math:7b`

### DeepSeek-Math-V2

| Property | Value |
|----------|-------|
| Sizes | 7B, 67B |
| Strengths | Math competition problems, formal proofs, step-by-step reasoning |
| Source | [DeepSeek](https://huggingface.co/deepseek-ai) |
| License | DeepSeek License |

### NuminaMath-2

| Property | Value |
|----------|-------|
| Sizes | 7B, 72B |
| Strengths | IMO-level problems, chain-of-thought math |
| Source | [AI-MO](https://huggingface.co/AI-MO) |
| License | Apache 2.0 |
| Note | 2025 IMO competition-tuned |

### Mathstral-2

| Property | Value |
|----------|-------|
| Sizes | 7B |
| Strengths | Efficient math reasoning, good balance of speed/quality |
| Source | [Mistral](https://huggingface.co/mistralai) |
| License | Apache 2.0 |

---

## Medical & Healthcare

### OpenBioLLM-2

| Property | Value |
|----------|-------|
| Sizes | 8B, 70B |
| Strengths | Biomedical research, clinical reasoning, drug interactions |
| Source | [Saama](https://huggingface.co/saama) |
| License | Apache 2.0 |
| Note | Major update with 2025 medical literature |

### Med-Gemma 2

| Property | Value |
|----------|-------|
| Sizes | 2B, 9B, 27B |
| Strengths | Medical Q&A, clinical scenarios, FDA-evaluated |
| Source | [Google Health](https://huggingface.co/google) |
| License | Gemma License (medical use restricted) |

### BioMistral-2

| Property | Value |
|----------|-------|
| Sizes | 7B, 22B |
| Strengths | PubMed-trained, clinical NLP, biomedical text |
| Source | [BioMistral](https://huggingface.co/BioMistral) |
| License | Apache 2.0 |

### ClinicalCamel-2

| Property | Value |
|----------|-------|
| Sizes | 13B, 70B |
| Strengths | Clinical notes, diagnosis support, EHR integration |
| Source | [wanglab](https://huggingface.co/wanglab) |
| License | Research only |

---

## Legal

### SaulLM-2

| Property | Value |
|----------|-------|
| Sizes | 7B, 54B |
| Strengths | Legal document analysis, case law, contracts |
| Source | [Equall](https://huggingface.co/Equall) |
| License | Apache 2.0 |
| Note | Trained on updated legal corpora through 2025 |

### LegalLlama-3

| Property | Value |
|----------|-------|
| Sizes | 8B, 70B |
| Strengths | Multi-jurisdictional, legal research, precedent analysis |
| Source | [legalai](https://huggingface.co/legalai) |
| License | Llama 3 License |

---

## Science & Research

### SciGLM-2

| Property | Value |
|----------|-------|
| Sizes | 9B, 32B |
| Strengths | Scientific reasoning, paper analysis, research |
| Source | [THUDM](https://huggingface.co/THUDM) |
| License | Apache 2.0 |

### ChemLLM-2

| Property | Value |
|----------|-------|
| Sizes | 7B, 20B |
| Strengths | Chemistry, molecular analysis, reaction prediction |
| Source | [AI4Chem](https://huggingface.co/AI4Chem) |
| License | Apache 2.0 |

### Galactica-2

| Property | Value |
|----------|-------|
| Sizes | 6.7B, 30B, 120B |
| Strengths | Scientific knowledge, citations, paper understanding |
| Source | [Meta AI](https://huggingface.co/facebook) |
| License | CC-BY-NC |

---

## Creative & Writing

### Mistral-Large-3

| Property | Value |
|----------|-------|
| Sizes | 123B (quantized versions available) |
| Strengths | Creative writing, nuanced prose, storytelling |
| Source | [Mistral](https://huggingface.co/mistralai) |
| License | Mistral Commercial |
| Note | Excellent for long-form creative content |

### Command-R+ 2

| Property | Value |
|----------|-------|
| Sizes | 35B, 104B |
| Strengths | Long document writing, editing, RAG-optimized |
| Source | [Cohere](https://huggingface.co/CohereForAI) |
| License | CC-BY-NC |

### Llama 4 Scout

| Property | Value |
|----------|-------|
| Sizes | 8B, 70B, 405B |
| Strengths | General writing, instruction following, versatile |
| Source | [Meta](https://huggingface.co/meta-llama) |
| License | Llama 4 License |

---

## Multi-Modal

### LLaVA-1.7

| Property | Value |
|----------|-------|
| Sizes | 7B, 13B, 34B |
| Strengths | Image understanding, visual Q&A |
| Source | [haotian-liu](https://huggingface.co/liuhaotian) |
| License | Apache 2.0 |

### Qwen3-VL

| Property | Value |
|----------|-------|
| Sizes | 3B, 7B, 72B |
| Strengths | Vision-language, document analysis, OCR |
| Source | [Qwen](https://huggingface.co/Qwen) |
| License | Apache 2.0 |

### InternVL-3

| Property | Value |
|----------|-------|
| Sizes | 2B, 8B, 26B, 76B |
| Strengths | Multi-image reasoning, charts, diagrams |
| Source | [OpenGVLab](https://huggingface.co/OpenGVLab) |
| License | Apache 2.0 |

---

## Router Candidates (Always-Resident)

These models work well as the always-resident router due to their efficiency and general capability:

| Model | Size | Strengths | Ollama Command |
|-------|------|-----------|----------------|
| Qwen 3 | 4B | Fast, capable, excellent instruction following | `ollama pull qwen3:4b` |
| Phi-4 | 3.8B | Efficient reasoning, Microsoft optimization | `ollama pull phi4` |
| Gemma 3 | 4B | Fast inference, good classification | `ollama pull gemma3:4b` |
| Llama 4 Scout | 8B | Versatile, strong instruction following | `ollama pull llama4:8b` |
| Granite 3.1 | 3B/8B | Enterprise-ready, IBM | `ollama pull granite3.1:3b` |
| Liquid-LFM-3 | 3B | Novel architecture, efficient | (when available) |
| SmolLM-2 | 1.7B | Ultra-fast, good for simple routing | `ollama pull smollm2:1.7b` |

---

## Model Format Notes

### GGUF (Recommended)
- Native format for llama.cpp and Ollama
- Quantization levels: Q4_K_M (balanced), Q5_K_M (quality), Q8_0 (best)
- Load times: ~0.3s per GB on NVMe Gen5

### Ollama
- Easiest setup, automatic quantization
- Manages model storage and serving
- API compatible with OpenAI format

### vLLM / SGLang
- Higher throughput for production
- Better batching for multi-user scenarios
- Requires more setup

---

## Emerging Models to Watch

These are expected to be strong expert candidates in 2026:

- **Claude 4 Haiku (leaked weights)** - If/when available
- **Gemini 3 Nano open weights** - Expected Q2 2026
- **GPT-5 distilled models** - Community distillations
- **Mistral-Medium-2** - Expected release
- **DeepSeek-V4** - Following V3's success

---

## Adding New Experts

To recommend a new expert model, please include:
1. Model name and source
2. Available sizes
3. Domain specialization
4. License information
5. Quantization availability
6. Any benchmark results

Submit as a PR to this file or open an issue.

---

*Last updated: January 2026*
