# College of Experts (CoE)

> A disk-resident sparse mixture architecture for running trillion-parameter-scale AI locally on consumer hardware.

[![License: PolyForm Noncommercial](https://img.shields.io/badge/License-PolyForm_Noncommercial-purple.svg)](https://polyformproject.org/licenses/noncommercial/1.0.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

## Official Repositories

- GitHub code repository: https://github.com/JThomas-CoE/College-of-Experts-AI
- Hugging Face model hub: https://huggingface.co/JThomas-CoE

**College of Experts** is an architectural framework that decomposes the sparse expert activation patterns of large Mixture-of-Experts (MoE) models into independently-served specialist networks.

Instead of requiring 100GB+ of VRAM to load a massive model like DeepSeek-V3 or Mixtral, CoE:
1.  **Stores specialized "expert" models on NVMe storage**
2.  **Dynamically loads only the specialist needed** for the current task
3.  **Orchestrates execution** via a lightweight router and supervisor

This enables **SOTA-class performance on consumer hardware** (e.g., single GPU, 32GB RAM) by trading microsecond-level token routing for second-level task routing.

**Key results (v4 benchmark):**
- **+27.1% average improvement** over isolated models across 5 domains (Code, Math, Medical, SQL, Legal).
- **Sub-second model swapping** possible with hierarchical RAM caching.

## 📦 Pre-Built Models (INT4)

Ready-to-use quantized models optimized for DirectML (AMD/Intel/NVIDIA) are available on Hugging Face:

Publicly visible today under the Hugging Face hub:

- `JThomas-CoE/DeepSeek-R1-Distill-Qwen-7B-INT4-DML`
- `JThomas-CoE/Qwen2.5-Coder-7B-INT4-DML`
- `JThomas-CoE/Qwen2.5-Math-7B-INT4-DML`
- `JThomas-CoE/BioMistral-7B-INT4-DML`
- `JThomas-CoE/sqlcoder-7b-2-INT4-DML`
- `JThomas-CoE/law-LLM-INT4-DML`

The stable root `demo.py` path additionally expects a Nanbeige supervisor folder named `Nanbeige4.1-3B-ONNX-INT4`.

| Role | Model (INT4) | Size | Link |
|---|---|---|---|
| **Supervisor (stable demo.py path)** | Nanbeige4.1-3B-ONNX-INT4 | ~3-6GB | [Download](https://huggingface.co/JThomas-CoE/Nanbeige4.1-3B-ONNX-INT4) |
| **Supervisor (experimental / demo_v13 path)** | DeepSeek-R1-Distill-Qwen-7B | ~5GB | [Download](https://huggingface.co/JThomas-CoE/DeepSeek-R1-Distill-Qwen-7B-INT4-DML) |
| **Code** | Qwen2.5-Coder-7B | ~5GB | [Download](https://huggingface.co/JThomas-CoE/Qwen2.5-Coder-7B-INT4-DML) |
| **Math** | Qwen2.5-Math-7B | ~4GB | [Download](https://huggingface.co/JThomas-CoE/Qwen2.5-Math-7B-INT4-DML) |
| **Medical** | BioMistral-7B | ~5GB | [Download](https://huggingface.co/JThomas-CoE/BioMistral-7B-INT4-DML) |
| **SQL** | SQLCoder-7B-2 | ~5GB | [Download](https://huggingface.co/JThomas-CoE/sqlcoder-7b-2-INT4-DML) |
| **Legal** | Law-LLM | ~4GB | [Download](https://huggingface.co/JThomas-CoE/law-LLM-INT4-DML) |

## Quick Start

### Prerequisites
- Python 3.10+
- Windows (recommended for DirectML) or Linux (ROCm/CUDA)
- 16GB+ RAM (32GB+ recommended)
- NVMe SSD

### Installation

```bash
git clone https://github.com/JThomas-CoE/College-of-Experts-AI.git
cd College-of-Experts-AI
pip install -r requirements.txt
```

### Download Models

Use the included script to fetch the quantized specialist models from Hugging Face:

```bash
# Downloads the specialist models to ./models/
python scripts/download_models.py
```

Then obtain the Nanbeige supervisor for the stable root demo:

```bash
python scripts/download_nanbeige.py
```

If the Nanbeige model files are not yet downloaded locally, place the ONNX INT4 folder manually in one of the expected search directories below.

For the stable root `demo.py` path, place the Nanbeige supervisor ONNX folder at one of:

- `models/Nanbeige4.1-3B-ONNX-INT4`
- `D:/models/Nanbeige4.1-3B-ONNX-INT4`
- `C:/models/Nanbeige4.1-3B-ONNX-INT4`

If you keep models elsewhere, set `model_base_dir` in `config/demo_config.json` after first launch.

If the Nanbeige model has already been published to the `JThomas-CoE` Hugging Face hub, you can use `python scripts/download_nanbeige.py` to download it into the expected local folder.

### Template Embedding Cache

The repo includes a pre-built embedding cache for the 620 framework templates
(`config/framework_templates/embedding_cache/`).  This lets the demo start
instantly rather than spending ~5 minutes recomputing embeddings on first launch.

If you ever add or change templates the cache will be invalidated automatically
and rebuilt once, then reused on all subsequent runs.

### Run the Demo

Start the stable interactive console (`TIER1` + `TIER2`):

```bash
python demo.py
```

The stable root demo uses:

- Nanbeige as the supervisor
- process-isolated worker inference via `inference_worker.py`
- `TIER1` and `TIER2` interactive execution
- blocked interactive `TIER3` prompts, which are currently split manually

Recommended public demo flow:

1. `python demo.py` for the stable repo demo
2. `python demo_v13.py` only when you want to show the experimental multi-specialist path

### Startup Note: First-Run Embedding Delay

- The first startup that builds template embeddings can be noticeably slower.
- CoE computes embeddings for the template catalog once, then persists them to cache.
- Subsequent startups load cached vectors and should be much faster.

For the stable root `demo.py` path, cache files live under:

- `config/framework_templates/embedding_cache/template_vectors.npy`
- `config/framework_templates/embedding_cache/template_vectors_ids.json`

CPU tuning knobs for first-run embedding speed are available in:

- `config/demo_config.json`
    - `embed_cpu_batch_size` (default `96`)
    - `embed_cpu_threads` (default `12`)

### Experimental: Multi-Specialist Demo (Tier 3)

For an early preview of **Tier 3 (Multi-Specialist)** orchestration, where multiple domain experts collaborate on a single complex query:

```bash
python demo_v13.py
```

*Note: `demo_v13.py` remains the experimental multi-specialist / DeepSeek-oriented path. The stable root `demo.py` is the recommended starting point.*

## Architecture

The system uses a **Tier-Driven Pipeline** to route queries:

- **TIER 1 (Trivial):** Handled directly by the Supervisor (latency < 1s).
- **TIER 2 (Standard):** Specialist generates a draft → Supervisor synthesizes/refines (default path).
- **TIER 3 (Complex):** Experimental in `demo_v13.py`; the stable root `demo.py` currently asks users to split multi-domain prompts.

```
┌─────────────────────────────────────────────────────────────┐
│                    TIER CLASSIFIER                          │
└──────────────┬──────────────────────────────┬───────────────┘
               │                              │
      ┌────────▼────────┐            ┌────────▼────────┐
      │   SUPERVISOR    │            │   SPECIALIST    │
      │ (Always Loaded) │            │ (Loaded on Demand) │
      └────────▲────────┘            └────────┬────────┘
               │                              │
               └────────── (Synthesis) ───────┘
```

See [WHITEPAPER.md](WHITEPAPER.md) for the full architectural specification and benchmark results.

## Project Structure

```
college-of-experts/
├── benchmarks/          # Evaluation scripts and datasets
├── models/              # Quantized ONNX GenAI models
├── scripts/             # Utilities (download, quantize, setup)
├── src/                 # Core source code
│   ├── backends/        # OGA/DirectML inference engine
│   ├── gui/             # Experimental GUI
│   ├── router.py        # Semantic routing logic
│   ├── harness.py       # Orchestration layer
│   └── ...
├── demo.py              # Stable root interactive demo (TIER1/TIER2)
├── inference_worker.py  # Worker process used by demo.py
├── demo.md              # Operational spec for the stable root demo
├── SETUP.md             # detailed setup guide
└── WHITEPAPER.md        # Technical report
```

## License

**PolyForm Noncommercial License 1.0.0**

- **Free for:** Personal use, academic research, hobby projects, non-profit organizations.
- **Commercial use** requires a separate license. See [LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md) for details.

## References

- [Technical Whitepaper](WHITEPAPER.md)
- [Hugging Face Collection](https://huggingface.co/JThomas-CoE)
