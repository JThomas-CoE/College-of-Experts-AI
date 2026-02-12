# College of Experts (CoE)

> A disk-resident sparse mixture architecture for running trillion-parameter-scale AI locally on consumer hardware.

[![License: PolyForm Noncommercial](https://img.shields.io/badge/License-PolyForm_Noncommercial-purple.svg)](https://polyformproject.org/licenses/noncommercial/1.0.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

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

| Role | Model (INT4) | Size | Link |
|---|---|---|---|
| **Supervisor** | DeepSeek-R1-Distill-Qwen-7B | ~5GB | [Download](https://huggingface.co/JThomas-CoE/DeepSeek-R1-Distill-Qwen-7B-INT4-DML) |
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

Use the included script to fetch the quantized experts from Hugging Face:

```bash
# Downloads all core models to ./models/
python scripts/download_models.py
```

### Run the Demo

Start the interactive console:

```bash
python demo.py
```

## Architecture

The system uses a **Tier-Driven Pipeline** to route queries:

- **TIER 1 (Trivial):** Handled directly by the Supervisor (latency < 1s).
- **TIER 2 (Standard):** Specialist generates a draft → Supervisor synthesizes/refines (default path).
- **TIER 3 (Complex):** Multiple specialists contribute to a synthesized answer.

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
├── demo.py              # Main entry point
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
