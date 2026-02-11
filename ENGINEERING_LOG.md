# College of Experts - Engineering Log

This document serves as the chronological record of engineering decisions, technical breakthroughs, and implementation steps for the College of Experts (CoE) project. It is intended for use in future whitepapers and as a reconstruction guide.

---

## Session: High-Performance AMD Optimization (January 2026)

### **Objective**
Achieve 40-50+ tokens/second inference for 7B-14B expert models on AMD Ryzen AI hardware (Windows/DirectML), overcoming previous CPU-only bottlenecks.

### **1. Hardware Context & Constraints**
- **GPU**: AMD Radeon (Integrated Ryzen AI 8060S).
- **VRAM**: 64GB hardware-reserved (from 128GB total).
- **RAM**: ~64GB usable by OS/Applications.
- **OS**: Windows 11.
- **Backend**: ONNX Runtime with DirectML (DML) for native AMD acceleration.

### **2. Technical Breakthroughs**

#### **A. Transition to ONNX GenAI (OGA)**
- **Decision**: Abandoned `bitsandbytes` (NF4) and `llama-cpp-python` (ROCm) due to poor Windows support and compilation complexities.
- **Solution**: Shifted to `onnxruntime-genai` (OGA), which provides a high-level API for DirectML-optimized models.
- **Outcome**: Successfully leveraged the native Windows/AMD graphics stack for inference.

#### **B. The "EP DML" Fix**
- **Problem**: Initial ONNX models were running on the CPU (~14 tok/s) or crashing with "Parameter is incorrect" when DML was enabled.
- **Diagnosis**: Models built via the OGA builder for `cpu` produce a graph that is incompatible with GPU-specific optimizations/layers.
- **Solution**: Rebuilt all experts using the `-e dml` (Execution Provider: DirectML) flag. 
- **Outcome**: Inference speed jumped from 14 tok/s to **46+ tok/s**.

#### **C. Memory-Surgical Quantization**
- **Challenge**: Local 4-bit quantization of 7B models requires significant RAM, often exceeding the 64GB usable limit and causing system-wide instability.
- **Optimization**: Developed `build_oga_model.py` with:
    - **Disk Offloading**: Using `device_map="cpu"` and `offload_folder` during the export phase to use the SSD as temporary RAM extension.
    - **RAM Monitoring**: Real-time monitoring with a 63.5GB "kill-switch" to prevent system crashes.
    - **DML-Targeting**: Ensuring the final `.onnx` graph is structure-optimized for the Radeon GPU.

### **3. Implementation Steps**

1.  **Backend Development**: Created `src/backends/oga_backend.py` to wrap the OGA generation loop.
2.  **Factory Routing**: Updated `ModelFactory` to detect OGA models (via `genai_config.json`) and route to the new backend.
3.  **Expert Migration**:
    - Downloaded raw source (Safetensors/PyTorch) for BioMistral, SQLCoder, Law-LLM, and Qwen2.5-Math.
    - Successfully quantized each to 4-bit ONNX using the DML builder.
    - Validated each model with specific speed tests (`test_law_dml_speed.py`, etc.).
4.  **Profile Update**: Migrated `config/profile_savant.json` to the new `-DML` model paths.
5.  **Clean-up**: Removed redundant source folders (~100GB) and older CPU-bound ONNX models.

### **4. Current Performance Metrics**

| Specialist | Model | Backend | Speed (tokens/s) |
| :--- | :--- | :--- | :--- |
| **Python** | Qwen2.5-Coder-7B | OGA-DML | **47.6** |
| **Math** | Qwen2.5-Math-7B | OGA-DML | **46.3** |
| **Legal** | Law-LLM-7B | OGA-DML | **46.5** |
| **Medical** | BioMistral-7B | OGA-DML | **40.3** |
| **SQL** | SQLCoder-7B | OGA-DML | **38.9** |
| **Logic** | Phi-3-14B | OGA-DML | **27.9** |

### **5. Roadmap Status**
- **Completed**: Phase 1 (Baseline Infrastructure) - High-speed DML optimization and multi-suite standardized benchmarking (n=25).
- **In-Progress**: Phase 2 (Knowledge Grounding) - Connecting experts to Shared Memory and ChromaDB.
- **Next Up**: ChromaDB integration for RAG-based grounding and hallucination reduction.

---

## Session: Benchmark Standardization & LegalBench Integration (January 22, 2026)

### **Objective**
Establish a statistically robust baseline using official datasets (HumanEval, GSM8K, Spider, PubMedQA, LegalBench) to validate the "Before" state for the whitepaper.

### **1. Dataset Integration Breakthrough**
- **Spider**: Successfully parsed official `dev.json`. Implemented SQL keyword intersection grading as a structural proxy.
- **PubMedQA**: Downloaded official `pubmedqa.json`. Standardized on "yes/no/maybe" classification tasks with exact-match priority grading.
- **LegalBench**: Scraped 31 specialized legal reasoning tasks from Hugging Face. Handled `.tsv` encoding artifacts and implemented Llama-2 specific prompt templates (`<s>[INST]`).
- **HumanEval/GSM8K**: Restored full multi-sample support (n=25) to ensure results are comparable to published SOTA.

### **2. Official Baseline Results (n=25)**

| Suite | Model | Accuracy | Perf (tok/s) | Note |
| :--- | :--- | :--- | :--- | :--- |
| **HumanEval** | Qwen2.5-Coder-7B | 71.4% | 38.55 | Strong coding baseline. |
| **GSM8K** | Qwen2.5-Math-7B | 56.0% | 38.82 | Good reasoning performance. |
| **Spider** | SQLCoder-7B | 64.8% | 22.11 | High SQL structural accuracy. |
| **Medical** | BioMistral-7B | 40.0% | 13.79 | PubMedQA is a high-difficulty set. |
| **Legal** | Law-LLM-7B | 8.4% | 36.41 | Baseline "low point"; candidate for grounding. |

### **3. Key Observations**
- **Legal Gap**: The 8.4% score is a confirmed "format failure." Zero-shot evaluation causes the model to provide conversational legal analysis (e.g., "The answer is Yes because...") instead of the single-word categorical labels required by LegalBench. Additionally, prompt tag repetition in early runs indicated template sensitivity. This establishes the critical need for **Phase 2: Context Injection**, where the Shared Memory Backbone will provide the few-shot demonstrations necessary to align the specialist's output with benchmark requirements.
- **VRAM Stability**: Running n=25 across 5 models sequentially confirmed the stability of the OGA VRAM cleanup (`gc.collect()` + `torch.cuda.empty_cache()` equivalent).

---

## Critical Path Watchlist (Pending Phase 2/3)

These items are currently "staged" but not yet integrated into the benchmarking harness. They represent the core intellectual property of the CoE architecture.

| Feature | Risk of Delay | Dependency |
| :--- | :--- | :--- |
| **Shared Memory Backbone** | High | Current experts are "amnesiac". Multi-expert coordination requires `MemoryBackbone` to be injected into the OGA generation loop. |
| **Trust/Grounding (RAG)** | Critical | This is the primary AXIS for the whitepaper. Without ChromaDB grounding, the CoE is just a fast router, not a "Trustworthy" specialist. |
| **Drift Detection** | Medium | Required for Phase 3 evaluation. Needs an embedding model to quantify semantic divergence over long sessions. |

---

## Session: Council Architecture V8.0 - Executive NPU Framework (January 22, 2026)

### **Objective**
Finalize the V8 Executive Framework, transitioning from the V7 prototype to an optimized NPU-resident supervisor and Quantized GPU Savants.

### **1. The V8 Executive Core**
- **NPU Router (Executive)**: 3× Qwen3-VL:4B running as Virtual Council members on the AMD Ryzen AI NPU via `flm-serve`. 
- **Topology**: Virtual Slots 0-2 (Router/Executive) + Physical Slots 3-5 (HuggingFace OGA/DML Savants).
- **Consensus**: NPU handles zero-latency decomposition and final response synthesis via triple-vote deliberation.

### **2. Quantized Savant Layer**
- **Architecture**: Domain-specific backbones (Qwen2.5-Coder-7B-AMD, Law-LLM-DML, Phi-3-OGA) optimized for DirectML.
- **Two-Tier Identity**: The Savant provides the deep technical knowledge; the NPU Executive injects the dynamic task persona (e.g., "HIPAA Audit Specialist").
- **A2A (Google-aligned)**: Structured context sharing between the NPU and GPU layers without redundant tokenization.

### **3. Correction of V7 Regression**
- **Resolution**: All V7-labeling removed. Core files promoted to V8.
- **Optimization**: Switched from standard transformers loaders to specialized OGA/DML backends for all Savants.

*Log Entry: 2026-01-22 (V8 Milestone)*

## Session: V10 DML Reasoning Supervisor (January 24, 2026)

### **Objective**
Replace legacy NPU Supervisor with GPU-resident DeepSeek-R1-Distill.

### **Achievements**
1. **DeepSeek Integrated**: Downloaded and quantized DeepSeek-R1-Distill-Qwen-7B to Int4 DML.
2. **Performance**: 46.08 tokens/sec on GPU.
3. **V10 Architecture**: All-GPU flow. DeepSeek in Slot 0 manages decomposition/critique/synthesis. Experts in Slots 1-7 parallelize execution.
4. **Multi-Model Residency**: Confirmed simultaneous VRAM residency of Supervisor + multiple 7B experts.

*Log Entry: 2026-01-24*

## Session: Semantic Routing Feasibility (V11)

### **Objective**
Validate Vector-based routing to eliminate 'Prompt Fragility' (Rule-based routing failure).

### **Results**
- **Engine**: `BAAI/bge-m3` running on AMD GPU via `sentence_transformers`.
- **Accuracy**: Perfect domain separation.
    - HIPAA -> Legal (0.55)
    - Sepsis -> Medical (0.49)
    - Hashing -> Security (0.55)
- **Conclusion**: Semantic Routing is the correct holistic architecture for V11. Ideally implemented as a Hybrid Router (LLM Decompose + Vector Assign).

*Log Entry: 2026-01-24*
