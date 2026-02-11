# Setup Guide — College of Experts

This guide walks you through setting up the College of Experts framework on your local hardware, including environment configuration, model acquisition, quantization, and verification.

---

## 1. Hardware Requirements

### Minimum (Demo / Single Expert)

| Component | Specification |
|-----------|---------------|
| GPU | Any DirectX 12 GPU with 8+ GB VRAM |
| System RAM | 32 GB |
| Storage | 50 GB free on NVMe SSD |
| OS | Windows 10/11 or Linux |

### Recommended (Full Benchmark Suite)

| Component | Specification |
|-----------|---------------|
| GPU | AMD Radeon RX 7000+ or NVIDIA RTX 4060+ with 16+ GB VRAM |
| System RAM | 64 GB+ |
| Storage | 100 GB free on NVMe Gen4+ SSD |
| OS | Windows 11 |

### Tested Configuration

| Component | Specification |
|-----------|---------------|
| CPU | AMD Ryzen 9 with Ryzen AI NPU |
| GPU | AMD Radeon RX 8060S (DirectML) |
| VRAM | 64 GB (32 GB budget used) |
| System RAM | 128 GB |
| Storage | NVMe Gen4 SSD |

---

## 2. Environment Setup

### 2.1 Python Environment

```bash
# Create a dedicated environment (conda or venv)
python -m venv coe_env
# Windows:
coe_env\Scripts\activate
# Linux:
source coe_env/bin/activate
```

### 2.2 Install Core Dependencies

```bash
pip install -r requirements.txt
```

### 2.3 Install Inference Backend

The framework uses **ONNX Runtime GenAI** with DirectML for AMD hardware. Choose ONE of the following:

#### AMD GPU (DirectML) — Primary Path
```bash
pip install onnxruntime-genai-directml
pip install torch>=2.0.0 torch-directml>=0.2.0
```

#### NVIDIA GPU (CUDA)
```bash
pip install onnxruntime-genai
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu121
```

#### CPU Only (Slow, for testing)
```bash
pip install onnxruntime-genai
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cpu
```

### 2.4 Install Embedding Model Dependencies

```bash
# For the BGE-M3 embedding model (used by the semantic router)
pip install sentence-transformers>=2.2.0
```

---

## 3. Model Acquisition

The framework requires 6 specialist models + 1 supervisor model. Models are NOT included in the repository due to their size (~5 GB each, ~35 GB total). You must either download pre-quantized versions or build them from HuggingFace source weights.

### 3.1 Model Inventory

| Role | Source Model | HuggingFace ID | Output Directory |
|------|-------------|----------------|------------------|
| **Supervisor** | DeepSeek-R1-Distill-Qwen-7B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | `models/DeepSeek-R1-Distill-Qwen-7B-DML` |
| **Code Expert** | Qwen2.5-Coder-7B-Instruct | `Qwen/Qwen2.5-Coder-7B-Instruct` | `models/Qwen2.5-Coder-7B-DML` |
| **Math Expert** | Qwen2.5-Math-7B-Instruct | `Qwen/Qwen2.5-Math-7B-Instruct` | `models/Qwen2.5-Math-7B-DML` |
| **Medical Expert** | BioMistral-7B | `BioMistral/BioMistral-7B` | `models/BioMistral-7B-DML` |
| **SQL Expert** | sqlcoder-7b-2 | `defog/sqlcoder-7b-2` | `models/sqlcoder-7b-2-DML` |
| **Legal Expert** | law-LLM | `AdaptLLM/law-LLM` | `models/law-LLM-DML` |

### 3.2 Option A: Build from HuggingFace (Recommended)

This downloads the full-precision weights from HuggingFace and quantizes them to INT4 for your target platform.

**Prerequisites:**
- HuggingFace account (some models require accepting license agreements)
- `huggingface-cli login` (for gated models)
- ~30 GB RAM available during quantization
- ~10 GB temporary disk space per model

**Build each model:**

```bash
# 1. Supervisor (DeepSeek-R1) — MOST IMPORTANT, build first
python scripts/build_oga_model.py \
    deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    models/DeepSeek-R1-Distill-Qwen-7B-DML \
    --ep dml

# 2. Code Expert (Qwen2.5-Coder)
python scripts/build_oga_model.py \
    Qwen/Qwen2.5-Coder-7B-Instruct \
    models/Qwen2.5-Coder-7B-DML \
    --ep dml

# 3. Math Expert (Qwen2.5-Math)
python scripts/build_oga_model.py \
    Qwen/Qwen2.5-Math-7B-Instruct \
    models/Qwen2.5-Math-7B-DML \
    --ep dml

# 4. Medical Expert (BioMistral)
python scripts/build_oga_model.py \
    BioMistral/BioMistral-7B \
    models/BioMistral-7B-DML \
    --ep dml

# 5. SQL Expert (SQLCoder)
python scripts/build_oga_model.py \
    defog/sqlcoder-7b-2 \
    models/sqlcoder-7b-2-DML \
    --ep dml

# 6. Legal Expert (law-LLM)
python scripts/build_oga_model.py \
    AdaptLLM/law-LLM \
    models/law-LLM-DML \
    --ep dml
```

> **Note:** The build script includes a memory monitor and will abort if system RAM exceeds 63.5 GB. This protects against OOM crashes during quantization. If you have less than 64 GB RAM, you may need to build models one at a time with all other applications closed.

> **Note for NVIDIA users:** Replace `--ep dml` with `--ep cuda` for CUDA-optimized ONNX models.

### 3.3 Option B: Download Pre-Quantized (When Available)

Pre-quantized DML models may be available on HuggingFace Hub:

```bash
# Example (when hosted):
# huggingface-cli download college-of-experts/Qwen2.5-Coder-7B-DML --local-dir models/Qwen2.5-Coder-7B-DML
```

Check the project's HuggingFace organization page for availability.

### 3.4 Expected Directory Structure After Setup

```
models/
├── DeepSeek-R1-Distill-Qwen-7B-DML/
│   ├── model.onnx
│   ├── model.onnx.data
│   ├── genai_config.json
│   ├── tokenizer.json
│   └── ...
├── Qwen2.5-Coder-7B-DML/
│   └── ...
├── Qwen2.5-Math-7B-DML/
│   └── ...
├── BioMistral-7B-DML/
│   └── ...
├── sqlcoder-7b-2-DML/
│   └── ...
└── law-LLM-DML/
    └── ...
```

---

## 4. Knowledge Base Setup

The framework uses ChromaDB for RAG-based knowledge grounding. Seed the knowledge base with domain-specific documents:

```bash
# Seed legal and medical reference documents
python benchmarks/seed_knowledge.py
```

This populates the vector store with LegalBench and PubMedQA reference data used during ensemble synthesis.

> **Note:** The included knowledge corpus is minimal (demo-sized). For production use, you would want to curate larger domain-specific corpora (see Section 4.4 of the whitepaper).

---

## 5. Verification

### 5.1 Check Environment

```bash
python scripts/check_env.py
```

This verifies:
- Python version
- ONNX Runtime GenAI installation
- DirectML / CUDA availability
- Torch installation

### 5.2 Verify Models

Run the benchmark in dry-run mode to verify all models load correctly:

```bash
python benchmarks/run_coe_benchmark_v4.py --samples 1
```

You should see:
```
Model verification:
  ✓ python_backend: models/Qwen2.5-Coder-7B-DML
  ✓ sql_schema_architect: models/sqlcoder-7b-2-DML
  ✓ html_css_specialist: models/Qwen2.5-Coder-7B-DML
  ✓ math_expert: models/Qwen2.5-Math-7B-DML
  ✓ security_architect: models/DeepSeek-R1-Distill-Qwen-7B-DML
  ✓ legal_contracts: models/law-LLM-DML
  ✓ medical_clinical: models/BioMistral-7B-DML
  ✓ deepseek_supervisor: models/DeepSeek-R1-Distill-Qwen-7B-DML
```

### 5.3 Run Full Benchmark

```bash
# Baseline (isolated specialists, no ensemble)
python benchmarks/run_benchmark.py --samples 50

# CoE v4 (tier-driven ensemble: specialist → supervisor)
python benchmarks/run_coe_benchmark_v4.py --samples 50
```

Expected runtime: ~2-3 hours for n=50 across all 5 suites on recommended hardware.

---

## 6. VRAM Configuration

The framework uses a VRAM budget system to prevent OOM errors. Default is 32 GB.

### Adjust for Your GPU

Edit the VRAM configuration in `src/savant_pool.py` or pass via environment:

```python
# For 16 GB VRAM (load one model at a time):
CUSTOM_VRAM_CONFIG = {
    "vram_budget_mb": 16000,
    "max_expert_slots": 1,
}

# For 24 GB VRAM (specialist + supervisor concurrent):
CUSTOM_VRAM_CONFIG = {
    "vram_budget_mb": 24000,
    "max_expert_slots": 2,
}

# For 48+ GB VRAM (multiple experts concurrent):
CUSTOM_VRAM_CONFIG = {
    "vram_budget_mb": 48000,
    "max_expert_slots": 3,
}
```

See [docs/DML_MODEL_REFERENCE.md](docs/DML_MODEL_REFERENCE.md) for detailed VRAM budgets per model.

---

## 7. Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: onnxruntime_genai` | Backend not installed | `pip install onnxruntime-genai-directml` |
| OOM during model build | Not enough RAM for quantization | Close all apps, build one model at a time |
| OOM during inference | VRAM budget too high | Reduce `vram_budget_mb` in config |
| `Model not found` errors | Models not downloaded | Follow Section 3 to build models |
| Slow generation (<5 tok/s) | Running on CPU instead of GPU | Verify DirectML/CUDA is installed and detected |
| `DML not available` | Missing DirectML runtime | Install latest AMD/DirectX drivers |
| Truncated legal outputs | law-LLM has 2K context limit | Expected; supervisor compensates in ensemble |

---

## 8. Quick Start Checklist

- [ ] Python 3.10+ environment created
- [ ] `pip install -r requirements.txt`
- [ ] ONNX Runtime GenAI installed (DirectML or CUDA)
- [ ] All 6 models built/downloaded into `models/`
- [ ] `python scripts/check_env.py` passes
- [ ] `python benchmarks/run_coe_benchmark_v4.py --samples 1` verifies all models
- [ ] Knowledge base seeded (`python benchmarks/seed_knowledge.py`)

You're ready to run benchmarks or the interactive demo (`python demo.py`).

---

*See [WHITEPAPER.md](WHITEPAPER.md) for the full technical report and benchmark results.*
*See [docs/DML_MODEL_REFERENCE.md](docs/DML_MODEL_REFERENCE.md) for detailed model specifications.*
