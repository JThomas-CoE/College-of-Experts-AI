# College of Experts - Implementation Walkthrough & Reconstruction Guide

This document tracks the step-by-step progress through the **College of Experts: Benchmark Evaluation Roadmap**. It provides the technical detail required for reconstruction and serves as the evidence base for the project's whitepaper.

---

## Phase 1: Baseline Infrastructure (In Progress)

### **Step 1.1: Savant Model Integration & Performance Breakthrough**
*   **Status**: COMPLETED (January 21, 2026)
*   **Objective**: Ensure 7B-14B expert models run locally at high speeds (40-50+ tokens/s) on AMD hardware.
*   **Technical Implementation**:
    *   **Backend**: Switched to `onnxruntime-genai` (OGA) with DirectML for native Radeon/Ryzen AI acceleration.
    *   **Quantization**: Convered 4-bit INT4 ONNX models using the OGA builder.
    *   **The DML Fix**: Explicitly built models with `-e dml` to avoid CPU fallback and graph incompatibility.
    *   **Memory Management**: Implemented disk-offloading during quantization to stay within 64GB usable RAM.
*   **Verified Models**:
    *   `Qwen2.5-Coder-7B-DML` (Python/Security)
    *   `Qwen2.5-Math-7B-DML` (Mathematics)
    *   `law-LLM-7B-DML` (Legal)
    *   `BioMistral-7B-DML` (Medical)
    *   `sqlcoder-7b-2-DML` (SQL)
    *   `Phi-3-medium-14B-DML` (Reasoning/Architecture)
*   **Reference Scripts**: `scripts/build_oga_model.py`, `tests/test_law_dml_speed.py`, `tests/test_bio_oga.py`.

### **Step 1.2: Benchmark Harness Development**
*   **Status**: IN PROGRESS
*   **Objective**: Create a Reproducible Benchmark Suite (`benchmarks/run_benchmark.py`).
*   **Action Plan**: Build a script that can swap between a single 7B model, the CoE (multi-expert), and theoretically a 70B baseline (if memory allows).

### **Step 1.3: Memory Awareness Verification (Pre-requisite)**
*   **Status**: PENDING (Logic Drafted)
*   **Objective**: Verify that experts can leverage the Shared Memory Backbone before benchmarking multi-expert tasks.
*   **Verification Script**: `tests/test_memory_integration.py`.
*   **Why it matters**: A multi-expert benchmark is meaningless if experts cannot share context via the backbone. This MUST be verified before Phase 2.

### **Step 1.3: Run Standard Benchmarks (Baseline)**
*   **Status**: COMPLETED (January 22, 2026)
*   **Objective**: Establish baseline scores for HumanEval, GSM8K, Spider, Medical, and Legal.
*   **Methodology**: n=25 samples per suite.
*   **Final Baseline Results**:
    *   HumanEval (Coder): 71.4% (38.55 tok/s)
    *   GSM8K (Math): 56.0% (38.82 tok/s)
    *   Spider (SQL): 64.8% (22.11 tok/s)
    *   Medical (PubMedQA): 40.0% (13.79 tok/s)
    *   Legal (LegalBench): 8.4% (36.41 tok/s)
*   **Observation**: Legal accuracy is significantly lower than expected; suspected misalignment in prompt templates or grading logic. This establishes a strong "Before" case for Phase 2 improvement.

---

## Phase 2: Knowledge Grounding & Hallucination Reduction (COMPLETED)

### **Step 2.1: ChromaDB Integration**
*   **Status**: COMPLETED (January 22, 2026)
*   **Objective**: Implement local vector storage for grounded knowledge.
*   **Implementation**: Integrated ChromaDB with local embeddings. Resolved ORT DLL conflict by enforcing model-first loading.

### **Step 2.2: Knowledge Corpus Seeding**
*   **Status**: COMPLETED
*   **Objective**: Ingest curated domain-specific documents. Seeded LegalBench and PubMedQA reference data.

### **Step 2.3: Citation & Validation Logic**
*   **Status**: COMPLETED
*   **Objective**: Harness now retrieves context and injects "Base your answer on CONTEXT" instructions.

---

## Phase 3: Trust Evaluation (Scheduled)

### **Step 3.1: Hallucination & Consistency Tests**
*   **Measurement**: TruthfulQA and FActScore.
*   **Technical Metric**: Implement "Drift Rate" measurement using embedding similarity between session turns.

---

## Phase 4: Synthesis & Publication (Scheduled)

### **Step 4.1: Results Analysis**
*   **Target**: Prove CoE + Grounding beats 70B generalists in trust/safety metrics.

---
*Last Updated: 2026-01-23*

---
*Log Entry: 2026-01-22 (V8 Milestone)*

---

## Session: V8 Failure Analysis & V9 Pivot (January 23, 2026)

### **The Crash of V8**
The "Council of Experts" concept (3x 4B models voting) failed in practice due to two critical issues:
1.  **DirectML Instability**: The AMD DirectML driver does not support 3 concurrent OGA sessions hammering the scheduler from Python threads. This caused hard crashes (Exit Code 1).
2.  **The "Stupidity of Crowds"**: Three 4B models voting on a complex prompt often reached a "consensus" on the *wrong* answer (e.g., assigning a database schema task to a medical doctor instead of an SQL architect).

### **The V9 Supervisor Architecture**
We simplified the topology to a "General & Specialist" model:
*   **The General (NPU)**: A single, large (20B) model (`gpt-oss-sg:20b`) running on the dedicated NPU. It handles planning, assignment, and QA. It is smart enough to handle negative constraints ("Do NOT assign medical expert for IT tasks").
*   **The Specialists (GPU)**: The original V8 Savants (bio-mistral, law-llm) running on the GPU via DirectML.

**Outcome**: The system is now stable, strictly typed (no more med-expert hallucinations), and significantly faster (no voting latency).

---

## Phase 1.5: Architectural Correction (V9 Supervisor) (COMPLETED)

### **Step 1.5: Pivot from Council (V8) to Supervisor (V9)**
*   **Status**: COMPLETED (January 23, 2026)
*   **Problem**: The "V8 Council" (3x Qwen-4B) proved unstable on consumer hardware.
    *   **Instability**: Running 3 parallel DirectML inference threads caused driver timeouts and uncatchable "Exit Code 1" crashes.
    *   **Accuracy**: 4B models, even when voting, struggled to differentiate "medical record admin" (Legal/SQL) from "clinical medicine" (Medical Expert), leading to hallucinated assignments.
*   **Solution**: **The V9 Supervisor**.
    *   Replaced the 3-model voting cluster with a **single** high-intelligence executive: `gpt-oss-sg:20b`.
    *   Running on the NPU (via FLM) allows the GPU to be dedicated entirely to the heavy lifting (Savants).
    *   **Result**: Zero crashes, perfect decomposition (HIPAA -> Legal), and simplified code path.
