# College of Experts: A Disk-Resident Sparse Mixture Architecture for Local Trillion-Parameter Inference

**Design Document v0.4 (V12 Architecture)**  
**Date: January 2026**  
**Last Updated: January 27, 2026**

---

## Abstract

This document proposes the **College of Experts** (CoE) architecture—a novel approach to deploying trillion-parameter-scale AI systems on consumer hardware. Unlike traditional Mixture of Experts (MoE) models that require all expert weights in GPU memory, CoE stores the vast majority of expert parameters on high-speed local storage (NVMe SSD) and dynamically loads only the subset needed for a given task session. A lightweight router model handles scope negotiation, expert selection, and coordination, while a shared memory backbone enables inter-expert communication and continual learning without model retraining.

The key insight enabling this architecture is that **task context changes slowly** relative to storage access latency—a user engaged in agentic work (coding, research, document drafting) remains in a coherent domain for minutes to hours, not milliseconds. This temporal locality, combined with an interactive scope-clarification phase, allows expert loading latency to be hidden or productively masked.

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Background and Related Work](#2-background-and-related-work)
3. [Architecture Overview](#3-architecture-overview)
4. [System Components](#4-system-components)
   - 4.1 [Router Model](#41-router-model)
   - 4.2 [Expert Models](#42-expert-models)
   - 4.3 [Shared Memory Backbone](#43-shared-memory-backbone)
   - 4.4 [Agentic Harness](#44-agentic-harness)
   - 4.5 [Storage and Caching Layer](#45-storage-and-caching-layer)
5. [Operational Modes](#5-operational-modes)
6. [Expert Coordination Patterns](#6-expert-coordination-patterns)
7. [HRM: Hierarchical Routing Mechanism](#7-hrm-hierarchical-routing-mechanism)
8. [Continual Learning and Adaptation](#8-continual-learning-and-adaptation)
9. [Modular Evolution and Upgradeability](#9-modular-evolution-and-upgradeability)
10. [Hardware Requirements and Performance Analysis](#10-hardware-requirements-and-performance-analysis)
11. [Comparison with Existing Approaches](#11-comparison-with-existing-approaches)
12. [Phased Architecture Philosophy](#12-phased-architecture-philosophy)
13. [V12 Architecture Revision](#13-v12-architecture-revision)
    - 13.1 [Lessons from V8-V11](#131-lessons-from-v8-v11)
    - 13.2 [Grounded Expert Scope Documents](#132-grounded-expert-scope-documents)
    - 13.3 [Framework Template Library](#133-framework-template-library)
    - 13.4 [Execution Modes](#134-execution-modes)
    - 13.5 [Task Framework Execution](#135-task-framework-execution)
    - 13.6 [Mandatory Output Structure](#136-mandatory-output-structure)
    - 13.7 [Assist Protocol](#137-assist-protocol)
    - 13.8 [V12 Storage Architecture](#138-v12-storage-architecture)
    - 13.9 [Adaptive Template Router](#139-adaptive-template-router)
    - 13.10 [VRAM-Aware Scheduling](#1310-vram-aware-scheduling)
14. [Implementation Roadmap](#14-implementation-roadmap)
15. [Open Research Questions](#15-open-research-questions)
16. [Conclusion](#16-conclusion)

---

## 1. Motivation

### The Scale Problem

State-of-the-art language models have scaled well into the trillions of parameters. GPT-5 and Gemini 3 Ultra are estimated at 2-3T+ parameters; Claude 4.5 Opus demonstrates similar scale with exceptional reasoning capabilities. Open models like DeepSeek v3.2 and GLM 4.7 have achieved remarkable parity with proprietary systems. These models demonstrate emergent capabilities that smaller models lack—better reasoning, broader knowledge, more nuanced understanding.

However, running such models locally is infeasible for most users:

| Model Size | FP16 Memory | Typical Consumer GPU |
|------------|-------------|----------------------|
| 7B         | ~14 GB      | ✅ RTX 4070          |
| 70B        | ~140 GB     | ❌ Requires 2-4× A100 |
| 400B       | ~800 GB     | ❌ Enterprise only    |
| 1T+        | ~2 TB       | ❌ Datacenter only    |

### The Sparsity Opportunity

Mixture of Experts architectures (Mixtral, Switch Transformer, etc.) demonstrate that **not all parameters are needed for every token**. Typically, only 2-8 experts (out of potentially hundreds) activate per forward pass. This suggests that a trillion-parameter model might only need 10-50B parameters "hot" at any moment.

### The Temporal Locality Insight

Existing MoE implementations still load all experts into memory because they assume token-level routing—the needed expert can change every token, requiring sub-millisecond switching.

But in realistic usage patterns:
- A user asking for coding help stays in "code mode" for the entire session
- A legal document review task needs legal expertise throughout
- A creative writing session doesn't suddenly need medical knowledge

**Task context changes slowly. Storage access is fast enough to track it.**

### The User Expectation Reality

Local users already accept significant latency:
- Loading a 70B model: 30-90 seconds
- Initial IDE indexing: 15-45 seconds
- First compilation of a large project: minutes

An interactive scope-clarification phase that masks expert loading is **not a degraded experience—it's a natural conversational flow**.

---

## 2. Background and Related Work

### Mixture of Experts (MoE)

- **Switch Transformer** (Google, 2021): Demonstrated trillion-parameter models with sparse routing
- **Mixtral 8x22B** (Mistral, 2024): Open-weight MoE with improved expert utilization
- **DeepSeek v3.2 MoE** (2025): 37B active from 671B total parameters, near-GPT-5 performance
- **GLM 4.7 Sparse** (Zhipu AI, 2025): Multi-modal MoE with dynamic expert allocation
- **Gemini 3 Nano MoE** (Google, 2025): Efficient on-device MoE for mobile/edge deployment

**Limitation**: All experts must reside in memory for token-level routing.

### Memory-Augmented Architectures

- **MemGPT / Letta** (2023-2025): Hierarchical memory with main context and archival storage, now production-ready
- **Retrieval-Augmented Language Models (RLM)**: External knowledge retrieval at inference time
- **Gemini 3 Memory Layer** (2025): Native long-term memory integration in foundation models
- **Claude 4 Extended Context** (Anthropic, 2025): 1M+ token context with hierarchical attention
- **Dropstone agentic framework** (2025): effectively unlimited memory integration in agentic tasks with foundation models

**Relevance**: Provides patterns for shared memory backbone design.

### Agentic Architectures

- **ReAct / ReWOO** (2022-2024): Reasoning and acting in language models
- **LangGraph / CrewAI** (2024-2025): Stateful multi-agent workflows with role specialization
- **OpenAI Operator / Claude Computer Use** (2025): Production autonomous agents with tool use
- **DeepSeek Agent Framework** (2025): Open-source agentic orchestration

**Relevance**: Provides patterns for expert coordination and task decomposition.

### Model Offloading

- **FlexGen** (2023): Offloading to CPU/SSD for throughput-oriented batched inference
- **DeepSpeed Inference**: Tensor parallelism with NVMe offloading
- **llama.cpp mmap**: Memory-mapped model loading

**Relevance**: Demonstrates SSD-based model access is technically viable.

---

## 3. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           USER INTERFACE LAYER                              │
│              (CLI, IDE Integration, Web UI, API Endpoints)                 
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AGENTIC HARNESS                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │    Task      │  │   Multi-     │  │    Tool      │  │   Output     │     │
│  │ Decomposition│  │   Turn Mgmt  │  │ Orchestration│  │  Synthesis   │     │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘     │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ROUTER MODEL (~1-3B)                                 │
│                                                                             │
│   • Scope negotiation and clarification dialog                              │
│   • Expert selection and loading orchestration                              │
│   • Lightweight query handling (no expert needed, eg liquid AI)             │
│   • Semantic transition detection during tasks                              │
│   • Expert manifest management                                              │
└────────────────────────────────────┬────────────────────────────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
│   SHARED MEMORY      │ │  ACTIVE EXPERT POOL  │ │  STORAGE LAYER       │
│   BACKBONE           │ │                      │ │                      │
│                      │ │  [Expert A] (GPU)    │ │  ┌────────────────┐  │
│  ┌────────────────┐  │ │  [Expert B] (GPU)    │ │  │ Expert Store   │  │
│  │ Working Memory │  │ │  [Expert C] (RAM)    │ │  │ (NVMe SSD)     │  │
│  │ (current task) │  │ │                      │ │  │                │  │
│  └────────────────┘  │ │  Capacity: 2-4 hot   │ │  │ 100-1000+      │  │
│  ┌────────────────┐  │ │            8-16 warm │ │  │ specialists    │  │
│  │ Episodic Memory│  │ │                      │ │  └────────────────┘  │
│  │ (past sessions)│  │ └──────────────────────┘ │                      │
│  └────────────────┘  │                          │  ┌────────────────┐  │
│  ┌────────────────┐  │                          │  │ Memory Archive │  │
│  │Semantic Memory │  │                          │  │ (Persistent)   │  │
│  │ (learned facts)│  │                          │  └────────────────┘  │
│  └────────────────┘  │                          │                      │
└──────────────────────┘                          └──────────────────────┘
```

---

## 4. System Components

### 4.1 Router Model

The router is the always-resident coordinator—a capable language model (1-3B parameters) small enough to remain in memory permanently.

**Responsibilities:**

1. **Scope Negotiation**
   - Conduct natural language dialog to understand user intent
   - Ask clarifying questions before expensive expert loading
   - Classify task domain(s) required

2. **Expert Selection**
   - Consult expert manifest (metadata about available experts)
   - Determine optimal expert ensemble for the task
   - Issue load/prefetch commands to storage layer

3. **Trivial Query Handling**
   - Answer simple questions without expert involvement
   - Conversions, definitions, basic lookups
   - Reduces unnecessary expert loading

4. **Transition Detection**
   - Monitor conversation for domain shifts
   - Proactively prefetch likely-needed experts
   - Signal expert swap recommendations to harness

**Example Router Interaction:**

```
User: "I need help with my Python project"

Router: "I'd be happy to help with your Python project. To bring in the 
        right expertise, could you tell me:
        
        1. What kind of project is it? (web app, data science, automation, etc.)
        2. Are you starting fresh or working with existing code?
        3. What's your main goal today?"
        
        [While user responds, router begins prefetching Python Expert]

User: "It's a FastAPI backend. I'm adding authentication and need to 
      integrate with our PostgreSQL database."

Router: "Got it—FastAPI backend with auth and PostgreSQL integration.
        I'm bringing in specialists for:
        ✓ Python/FastAPI development
        ✓ Authentication patterns
        ✓ PostgreSQL integration
        
        [Loading... 3 seconds]
        
        Ready! Let's start with your authentication approach..."
```

### 4.2 Expert Models

Experts are domain-specialized language models (500M-10B parameters each) stored on disk and loaded on demand.

**Characteristics:**

| Property | Description |
|----------|-------------|
| Size Range | 500M - 10B parameters per expert |
| Specialization | Deep expertise in narrow domain |
| Storage Format | Quantized (Q4/Q8) for efficient loading |
| Interface | Standardized input/output contract |
| Independence | Can be upgraded without affecting others |

**Expert Taxonomy (Example):**

```
experts/
├── programming/
│   ├── python_expert_3b.gguf
│   ├── javascript_expert_3b.gguf
│   ├── rust_expert_2b.gguf
│   └── sql_expert_1b.gguf
├── domains/
│   ├── legal_expert_5b.gguf
│   ├── medical_expert_5b.gguf
│   ├── finance_expert_3b.gguf
│   └── scientific_expert_5b.gguf
├── capabilities/
│   ├── reasoning_expert_7b.gguf
│   ├── creative_writing_expert_3b.gguf
│   ├── code_review_expert_2b.gguf
│   └── debugging_expert_3b.gguf
└── modalities/
    ├── vision_expert_4b.gguf
    └── document_analysis_expert_3b.gguf
```

**Expert Manifest Schema:**

```json
{
  "expert_id": "python_expert_v2.1",
  "display_name": "Python Development Expert",
  "version": "2.1.0",
  "size_bytes": 3_200_000_000,
  "quantization": "Q4_K_M",
  "domains": ["python", "programming", "debugging", "testing"],
  "capabilities": ["code_generation", "code_review", "explanation"],
  "load_time_estimate_ms": 800,
  "memory_footprint_mb": 1800,
  "interface_version": "1.0",
  "dependencies": [],
  "conflict_with": []
}
```

### 4.3 Shared Memory Backbone

The memory backbone enables inter-expert communication and persistent learning without model retraining.

**Memory Tiers:**

#### Working Memory
- Current task context and state
- Active conversation history
- Inter-expert message passing
- Scratchpad for intermediate results

#### Episodic Memory
- Past session summaries
- User interaction patterns
- What approaches worked/failed
- Task completion history

#### Semantic Memory
- Learned facts and preferences
- Domain-specific knowledge base
- User-specific customizations
- Updated priors and corrections

**Memory Operations:**

```python
# Experts interact with memory through standardized interface
class MemoryBackbone:
    def read(self, key: str, memory_tier: str) -> Any
    def write(self, key: str, value: Any, memory_tier: str) -> None
    def search(self, query: str, memory_tier: str, top_k: int) -> List[Result]
    def subscribe(self, pattern: str, callback: Callable) -> None
```

**Inter-Expert Communication Example:**

```
Python Expert writes to working memory:
  {
    "finding": "User's code has SQL injection vulnerability in line 47",
    "severity": "critical",
    "suggested_fix": "Use parameterized queries",
    "requesting_expert": "security_expert"
  }

Security Expert (if loaded) reads this and provides:
  {
    "response_to": "sql_injection_finding",
    "detailed_analysis": "...",
    "additional_vulnerabilities": [...]
  }
```

### 4.4 Agentic Harness

The harness provides the orchestration layer for complex multi-step tasks.

**Components:**

1. **Task Decomposition**
   - Break complex goals into subtasks
   - Identify expert requirements per subtask
   - Build execution DAG

2. **Multi-Turn Management**
   - Maintain conversation state
   - Handle interruptions and redirections
   - Manage context window across experts

3. **Tool Orchestration**
   - Connect experts to external tools (file system, browser, APIs)
   - Manage tool permissions and sandboxing- MCP standardization now likely
   - Handle tool output integration

4. **Output Synthesis**
   - Combine multi-expert outputs coherently
   - Resolve conflicts between expert recommendations
   - Format final responses appropriately

**Harness State Machine:**

```
                 ┌──────────────┐
                 │   IDLE       │
                 └──────┬───────┘
                        │ user_input
                        ▼
                 ┌──────────────┐
         ┌───── │   SCOPING    │ ─────┐
         │      └──────────────┘      │
         │ scope_clear                │ needs_clarification
         ▼                            ▼
  ┌──────────────┐            ┌──────────────┐
  │   LOADING    │            │  CLARIFYING  │
  └──────┬───────┘            └──────┬───────┘
         │ experts_ready             │ user_response
         ▼                           │
  ┌──────────────┐                   │
  │  EXECUTING   │ ◄─────────────────┘
  └──────┬───────┘
         │
    ┌────┴────┐
    │         │
    ▼         ▼
┌────────┐ ┌────────┐
│COMPLETE│ │CONTINUE│──► back to EXECUTING
└────────┘ └────────┘
```

### 4.5 Storage and Caching Layer

**Hierarchical Cache Structure:**

| Tier | Medium | Capacity | Latency | Contents |
|------|--------|----------|---------|----------|
| L1 | GPU VRAM | 16-48 GB | <1ms | 2-4 active experts |
| L2 | System RAM | 32-64 GB | 1-10ms | 8-16 warm experts |
| L3 | NVMe SSD | 1-4 TB | 100-500ms | 100-1000 cold experts |
| L4 | Network/HDD | Unlimited | seconds | Archival/community experts |

**Loading Pipeline:**

```
1. Router identifies needed expert
2. Check L1 (GPU) → hit? dispatch immediately
3. Check L2 (RAM) → hit? promote to L1, dispatch
4. Check L3 (SSD) → initiate async load to L2
   - Meanwhile: router continues dialog OR
   - Show loading indicator
5. On load complete → promote to L1 if GPU space available
6. Apply eviction policy to demote unused experts
```

**Eviction Policy Options:**

- **LRU (Least Recently Used)**: Simple, effective for diverse workloads
- **LFU (Least Frequently Used)**: Better for repeated workflows
- **Session-Aware**: Keep experts loaded for predicted session duration
- **Dependency-Aware**: Keep experts that others frequently call

---

## 5. Operational Modes

### Single-Expert Mode
Simple queries routed to one specialist.

```
User: "Explain Python decorators"
→ Router: Python Expert sufficient
→ Load Python Expert, generate response
```

### Ensemble Mode
Multiple experts collaborate on complex queries.

```
User: "Review this FastAPI code for security issues"
→ Router: Need Python + Security + API Design experts
→ Load ensemble, each contributes analysis
→ Harness synthesizes unified report
```

### Pipeline Mode
Sequential expert processing.

```
User: "Translate this legal document and summarize key points"
→ Stage 1: Translation Expert
→ Stage 2: Legal Expert (on translated text)
→ Stage 3: Summarization Expert (on legal analysis)
```

### Agentic Mode
Extended autonomous operation with expert switching.

```
User: "Build a REST API for user management with tests"
→ Initial: Python + API Design experts
→ During implementation: Testing expert joins
→ On database work: SQL expert joins
→ On completion: Documentation expert for API docs
```

---

## 6. Expert Coordination Patterns

### Task Force Pattern
Experts work as collaborative team, sharing findings through working memory.

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Expert A   │────▶│   Working    │◀────│   Expert B   │
│              │     │   Memory     │     │              │
│  Contributes │     │              │     │  Builds on   │
│  analysis    │     │  Shared      │     │  A's work    │
└──────────────┘     │  findings    │     └──────────────┘
                     └──────────────┘
                            ▲
                            │
                     ┌──────────────┐
                     │   Expert C   │
                     │              │
                     │  Synthesizes │
                     └──────────────┘
```

### Competitive Pattern
Multiple experts propose solutions; best selected.

```
User: "How should I architect this system?"

Expert A (microservices): "Use event-driven microservices..."
Expert B (monolith-first): "Start monolithic, extract later..."
Expert C (serverless): "Consider serverless functions..."

Router: Evaluates proposals, presents options to user
        OR selects best match for stated constraints
```

### Consultative Pattern
Lead expert calls others for specific sub-problems.

```
Lead: Python Expert (owns the task)

Python Expert: "I need to optimize this database query"
               → Calls SQL Expert for query optimization
               → Receives optimized query
               → Continues with implementation

Python Expert: "Let me verify this auth implementation"
               → Calls Security Expert for review
               → Addresses flagged issues
```

---

## 7. HRM: Hierarchical Routing Mechanism

The HRM layer resides in the router/host model and provides sustained workflow orchestration. Unlike traditional MoE token-level routing, HRM operates at the **task and subtask level**, managing extended multi-step execution.

### 7.1 Core HRM Loop

```python
class HRMOrchestrator:
    """
    Hierarchical Routing Mechanism for sustained task execution.
    Lives in the router, not in individual experts.
    """
    
    def execute_task(self, user_request: str, context: dict) -> str:
        # Phase 1: Task Decomposition
        task_plan = self.decompose(user_request, context)
        
        # Phase 2: Iterative Execution with Monitoring
        results = []
        for subtask in task_plan.subtasks:
            # Select appropriate expert(s)
            experts = self.select_experts(subtask)
            
            # Execute with monitoring
            result = self.execute_with_monitoring(subtask, experts)
            
            # Evaluate result quality
            confidence = self.evaluate_confidence(result, subtask)
            
            if confidence < self.redistribution_threshold:
                # Redistribution trigger
                result = self.handle_low_confidence(subtask, result, experts)
            
            results.append(result)
            
            # Update task plan if needed (new subtasks discovered)
            task_plan = self.maybe_revise_plan(task_plan, result)
        
        # Phase 3: Synthesis
        return self.synthesize(results, user_request)
    
    def handle_low_confidence(self, subtask, result, current_experts):
        """
        Called when expert confidence drops below threshold.
        Options: add expert, switch expert, decompose further, escalate.
        """
        diagnosis = self.diagnose_difficulty(subtask, result)
        
        if diagnosis.type == "needs_additional_expertise":
            # Bring in complementary expert
            new_expert = self.select_complementary_expert(diagnosis.domain)
            return self.execute_with_ensemble([*current_experts, new_expert], subtask)
        
        elif diagnosis.type == "task_too_complex":
            # Further decompose
            sub_subtasks = self.decompose(subtask, granularity="finer")
            return [self.execute_task(s) for s in sub_subtasks]
        
        elif diagnosis.type == "out_of_scope":
            # Escalate to user
            return self.request_user_guidance(subtask, diagnosis)
        
        else:
            # Accept with caveat
            return result.with_confidence_note()
```

### 7.2 HRM State Machine

```
                     ┌────────────────────┐
                     │   TASK_RECEIVED    │
                     └─────────┬──────────┘
                               │
                               ▼
                     ┌────────────────────┐
                     │    DECOMPOSING     │
                     │                    │
                     │ Router breaks task │
                     │ into subtasks      │
                     └─────────┬──────────┘
                               │
                               ▼
                     ┌────────────────────┐
              ┌──────│    DISPATCHING     │──────┐
              │      │                    │      │
              │      │ Select expert(s)   │      │
              │      │ for next subtask   │      │
              │      └────────────────────┘      │
              │                                  │
              ▼                                  ▼
    ┌──────────────────┐              ┌──────────────────┐
    │    EXECUTING     │              │   ALL_COMPLETE   │
    │                  │              │                  │
    │ Expert working   │              │ Proceed to       │
    │ on subtask       │              │ synthesis        │
    └────────┬─────────┘              └────────┬─────────┘
             │                                  │
             ▼                                  │
    ┌──────────────────┐                        │
    │    EVALUATING    │                        │
    │                  │                        │
    │ Check confidence │                        │
    │ and quality      │                        │
    └────────┬─────────┘                        │
             │                                  │
       ┌─────┴─────┐                            │
       │           │                            │
       ▼           ▼                            │
   ┌───────┐  ┌────────────┐                    │
   │ OKAY  │  │ REDISTRIBUTE│                   │
   │       │  │             │                   │
   │ Next  │  │ Add expert  │                   │
   │subtask│  │ or decompose│                   │
   └───┬───┘  └──────┬──────┘                   │
       │             │                          │
       └─────────────┴────────▶ DISPATCHING     │
                                                │
                               ┌────────────────┘
                               ▼
                     ┌────────────────────┐
                     │   SYNTHESIZING     │
                     │                    │
                     │ Combine results    │
                     │ into final output  │
                     └─────────┬──────────┘
                               │
                               ▼
                     ┌────────────────────┐
                     │     COMPLETE       │
                     └────────────────────┘
```

### 7.3 Confidence Signals

Without true attention-mechanism access, the MVP approximates confidence through:

| Signal Type | Implementation | Example |
|-------------|----------------|---------|
| **Explicit Scoring** | Ask expert to rate confidence | "Rate your confidence 1-10" in prompt |
| **Hedge Detection** | Parse response for uncertainty language | "might", "possibly", "I'm not sure" |
| **Incompleteness** | Check for TODO/placeholder markers | "TODO:", "[needs more info]" |
| **Contradiction** | Compare with context/memory | Result contradicts earlier statements |
| **Domain Mismatch** | Expert self-reports out of scope | "This is more of a legal question..." |

```python
def evaluate_confidence(self, result: str, subtask: Subtask) -> float:
    """MVP confidence estimation without attention access."""
    
    score = 1.0
    
    # Check for explicit confidence if requested
    if explicit := extract_confidence_score(result):
        score *= explicit / 10.0
    
    # Hedge word detection
    hedge_words = ["might", "possibly", "perhaps", "I think", "not sure"]
    hedge_count = sum(1 for w in hedge_words if w.lower() in result.lower())
    score *= (1.0 - 0.1 * min(hedge_count, 5))
    
    # Incompleteness markers
    if any(marker in result for marker in ["TODO", "[TBD]", "needs clarification"]):
        score *= 0.5
    
    # Length sanity (too short might indicate difficulty)
    if len(result) < subtask.expected_min_length:
        score *= 0.7
    
    return score
```

### 7.4 MVP Simplification

For the faux expert MVP, HRM is simplified:

| Full HRM | MVP Implementation |
|----------|-------------------|
| Attention-based confidence | Prompt-based explicit scoring + hedge detection |
| Latent space redistribution signals | Text-based "need help with X" patterns |
| Sustained multi-minute execution | Iterative subtask loop with checkpoints |
| Dynamic expert loading | System prompt switching (instant) |

The MVP validates the **orchestration patterns** without requiring the advanced instrumentation that attention-based monitoring would need.

---

## 8. Continual Learning and Adaptation

The CoE architecture enables learning without retraining through memory updates and adaptive learning without wholesale model retraining by individual expert updates. When new/improved models are released, the CoE can be updated by simply replacing the models and updating the memory format even if the new models have different internal structures/paradigms due to the abstraction layer "glue".

### User Preference Learning

```
User says: "I prefer functional programming style"

→ Router writes to Semantic Memory:
  user.preferences.programming_style = "functional"

→ Future code generation:
  Experts read this preference
  Generate functional-style code by default
```

### Factual Updates

```
New information: "Python 3.14 released with new features"

→ Can be added to Semantic Memory
→ Experts read updated facts at inference time
→ If model retraining required user can be informed not to update to new version of python.
```

### Workflow Pattern Learning

```
Observed: User always runs tests after code changes

→ Episodic Memory records pattern
→ Future sessions: Harness proactively suggests/runs tests
```

### Correction Integration

```
User: "Actually, our company uses 4-space indentation, not 2"

→ Semantic Memory updated
→ All future code generation respects this
```

---

## 9. Modular Evolution and Upgradeability

### Component Independence

Each layer can be upgraded independently:

| Component | Upgrade Path | Breaking Change Risk |
|-----------|--------------|---------------------|
| Individual Expert | Replace .gguf file | Low (if interface maintained) |
| Router Model | Replace with better base model | Low (memory persists) |
| Memory Backend | Migrate data, swap implementation | Medium (data format changes) |
| Agentic Harness | Update orchestration logic | Low (experts unchanged) |
| Expert Interface | Version negotiation | Medium (coordinate changes) |

### Expert Versioning

```
experts/
├── python_expert/
│   ├── v1.0/ (deprecated)
│   ├── v2.0/ (stable)
│   └── v2.1/ (latest)
└── manifest.json → points to recommended versions
```

### A/B Testing New Experts

```
Router can:
1. Load both old and new expert versions
2. Route subset of queries to new version
3. Compare output quality
4. Gradually shift traffic to winner
```

### Community Expert Ecosystem

The standardized interface enables:
- Third-party expert development
- Domain-specific expert marketplaces
- Open-source expert sharing
- Enterprise custom expert deployment

---

## 10. Hardware Requirements and Performance Analysis

### Minimum Viable Configuration

| Component | Specification | Role |
|-----------|---------------|------|
| GPU | RTX 4060 8GB | 1-2 active experts |
| RAM | 32 GB | Router + 4-8 warm experts |
| Storage | 1 TB NVMe Gen4 | 50-100 experts |

### Recommended Configuration

| Component | Specification | Role |
|-----------|---------------|------|
| GPU | RTX 4090 24GB | 3-4 active experts |
| RAM | 64 GB | Router + 16 warm experts |
| Storage | 2 TB NVMe Gen5 | 200+ experts |

### Performance Analysis

**Expert Loading Time (NVMe Gen4, 7 GB/s):**

| Expert Size | Load Time | With Quantization (Q4) |
|-------------|-----------|------------------------|
| 1B params   | ~0.3s     | ~0.15s                 |
| 3B params   | ~0.9s     | ~0.4s                  |
| 7B params   | ~2.0s     | ~0.9s                  |
| 13B params  | ~3.7s     | ~1.7s                  |

**Effective Throughput:**

- Simple queries (router only): <500ms
- Single expert queries: 500ms-2s (including load)
- Ensemble queries: 2-5s (parallel expert loading)
- Session continuation: <500ms (experts already loaded)

---

## 11. Comparison with Existing Approaches

| Aspect | Monolithic LLM | Traditional MoE | College of Experts |
|--------|---------------|-----------------|-------------------|
| Memory Required | Full model | All experts | Active experts only |
| Max Practical Size | ~70B local | ~140B local | 1T+ local |
| Upgrade Path | Full retrain | Full retrain | Per-expert swap |
| Specialization | Generalist | Soft partitioned | Deep domain expertise |
| Cold Start | Model load time | Model load time | Scope dialog + partial load |
| Continual Learning | Fine-tuning | Fine-tuning | Memory updates |
| Hardware Cost | High | Higher | Moderate |

---

## 12. Phased Architecture Philosophy

The full College of Experts vision involves sophisticated components that require significant research and engineering. This section outlines a pragmatic implementation strategy that captures the core value proposition while deferring hard problems.

### 12.1 The Faux Expert MVP

The simplest viable implementation uses **prompt-differentiated experts**—multiple instances of the same capable base model, each configured with a specialized system prompt that activates domain-specific behavior:

```python
# All experts share the same underlying model
BASE_MODEL = "qwen2.5:7b"

EXPERT_PROMPTS = {
    "python_expert": """You are a Python specialist with deep expertise in:
        - Python 3.10+ features, type hints, async patterns
        - FastAPI, SQLAlchemy, Pydantic, pytest
        - Package management and virtual environments
        Always provide production-quality code with comprehensive error handling.
        Explain architectural decisions and trade-offs.""",
    
    "security_expert": """You are a security specialist focused on:
        - OWASP Top 10 vulnerabilities
        - Authentication/Authorization patterns  
        - Input validation, sanitization, injection prevention
        Always assume adversarial inputs. Flag potential vulnerabilities explicitly.
        Recommend defense-in-depth strategies.""",
    
    "architecture_expert": """You are a software architect focused on:
        - System design patterns and anti-patterns
        - Scalability, performance, maintainability trade-offs
        - Technical debt identification and management
        Provide multiple options with clear pros/cons analysis.
        Consider long-term evolution and team capabilities."""
}
```

**Why this works:**
- Modern instruct models have latent capabilities across many domains
- System prompts activate and focus these capabilities effectively
- The same model genuinely produces different *reasoning styles* based on persona
- Behavioral differentiation (paranoid security reviewer vs. pragmatic architect) is real and useful

**What this validates:**
- Router/HRM orchestration logic
- Expert selection and dispatch mechanisms
- Multi-expert coordination patterns
- Output synthesis strategies
- Memory integration patterns

### 12.2 Deferred Hard Problems

The MVP explicitly defers certain advanced capabilities:

| Capability | Status | Rationale |
|------------|--------|-----------|
| **Fast Latent Space Memory** | ❌ Deferred | Requires research into shared representation formats, attention-based communication protocols, and possibly custom model architectures. Text-based inter-expert communication is sufficient for MVP validation. |
| **Attention-Based Redistribution** | ❌ Deferred | Requires instrumentation to extract confidence/uncertainty signals from expert execution. Explicit confidence scoring in prompts can approximate this initially. |
| **True Domain Expert Models** | ❌ Deferred | Requires fine-tuned or distilled specialist models. Faux experts validate the architecture; true experts are drop-in upgrades. |
| **Sustained Autonomous Execution** | ⚠️ Simplified | Full HRM-style multi-minute reasoning loops deferred. Simpler iterative decomposition in the harness suffices for MVP. |

The fast latent memory problem is particularly challenging: it requires experts to share intermediate *representations* (not just text outputs), which implies compatible embedding spaces, shared key-value caches, or explicit latent communication channels. This is frontier research territory and unnecessary for validating the core architecture.

### 12.3 Expert Switching Advantage

The faux expert approach offers an unexpected advantage: **instant expert switching**.

With true domain experts (separate model weights), switching requires:
1. Evicting current expert from GPU memory
2. Loading new expert weights from storage
3. Warming up the model (first inference often slower)

With prompt-differentiated experts on a shared model:
1. Change the system prompt
2. Done

This enables rapid multi-expert consultation patterns that would be expensive with true experts:

```
User: "Review this FastAPI endpoint"

Router thinks: "Need Python + Security + API Design perspectives"

# With faux experts: ~100ms total switching
python_review = python_expert.analyze(code)      # System prompt swap
security_review = security_expert.analyze(code)   # System prompt swap  
api_review = api_design_expert.analyze(code)     # System prompt swap
synthesized = router.synthesize([python_review, security_review, api_review])

# With true experts: ~3-9 seconds loading overhead per switch
```

This makes the MVP potentially *more responsive* for certain workloads than a naive true-expert implementation.

### 12.4 The Domain Expert Efficiency Hypothesis

While the MVP uses faux experts, the long-term vision anticipates **true domain experts** with dramatically better efficiency curves.

**The Core Insight:**

A general-purpose 200B parameter model must allocate capacity across all domains—coding, medicine, law, creative writing, mathematics, etc. Most of these parameters are "dormant" for any given query.

A purpose-trained 4B domain expert can potentially match or exceed the domain-specific performance of a 200B generalist by:
- Dedicating 100% of parameters to domain knowledge
- Optimizing tokenization for domain vocabulary
- Training exclusively on domain-relevant data
- Developing domain-specific reasoning patterns

**Theoretical Efficiency:**

| Model Type | Parameters | Domain Performance | Efficiency Ratio |
|------------|------------|-------------------|------------------|
| General 200B | 200B | Baseline | 1× |
| Domain Expert 4B | 4B | ~Equal to 200B in-domain | 50× more efficient |
| Domain Expert 7B | 7B | Exceeds 200B in-domain | 28× more efficient |

This is the **sparse expertise at scale** vision: a "1 trillion parameter equivalent" system composed of 100-200 domain experts at 4-10B each, with only 1-3 active at any moment.

**Evidence for this hypothesis:**
- Code-specialized models (DeepSeek-Coder, CodeLlama) outperform larger generalists on coding benchmarks
- Math-specialized models show similar gains
- Multilingual models with language-specific components outperform uniform architectures
- Mechanistic interpretability research suggests models develop semi-specialized circuits that could be isolated

**MVP Role:**

The faux expert MVP validates the *orchestration* question: can a router-based architecture effectively dispatch to, coordinate, and synthesize outputs from multiple specialists?

If yes, the efficiency gains from true domain experts become compelling economic motivation for their development.

### 12.5 Evolutionary Path

```
Phase 0 (Faux Expert MVP)
    │
    │  Validates: Orchestration, routing, memory patterns
    │  Uses: Same model + different system prompts
    │
    ▼
Phase 1 (Mixed Reality)
    │
    │  Introduces: 1-2 true domain experts for critical domains
    │  Compares: Faux vs. true expert output quality
    │  Quantifies: Actual efficiency gains
    │
    ▼
Phase 2 (Selective Specialization)  
    │
    │  Expands: True experts for high-value domains
    │  Retains: Faux experts for long-tail domains
    │  Optimizes: Loading/caching strategies
    │
    ▼
Phase 3 (Full College)
    │
    │  Implements: Fast latent memory (if beneficial)
    │  Supports: Community expert contributions
    │  Enables: Custom enterprise expert development
```

---

## 13. V12 Architecture Revision

Following extensive testing of V8 through V11 implementations, this section documents the architectural lessons learned and the V12 design that addresses identified regressions while preserving successful innovations.

### 13.1 Lessons from V8-V11

#### Version History Summary

| Version | Architecture | What Worked | What Regressed |
|---------|--------------|-------------|----------------|
| **V7** | NPU Router (3× serial council) + GPU Experts | Temperature-diverse decomposition | Serial execution too slow (~10s) |
| **V8** | NPU Council + GPU Savants + A2A Crosstalk | ✅ Best benchmark results (47 tok/s) | Council voting overhead |
| **V9** | NPU Supervisor + Self-Review Loop | ✅ Persona-swap review pattern | NPU latency bottleneck |
| **V10** | All-GPU DeepSeek Supervisor | Eliminated NPU dependency | Plan-level-only review |
| **V11** | DeepSeek + BGE-M3 Semantic Router | ✅ Embedding-based routing | Over-engineered synthesis, keyword-trap misrouting |

#### Key Failure: The HIPAA Query

A test query "Write a Python script to securely analyze medical records from a HIPAA compliant database and generate a Delaware-jurisdiction legal disclaimer" exposed critical flaws:

1. **Keyword-Trap Misrouting**: "medical records" triggered medical expert assignment despite no clinical expertise being needed
2. **LLM Synthesis Failures**: Final output contained syntax errors, missing templates, incorrect legal information
3. **Domain vs Task Confusion**: System conflated "data about domain X" with "expertise in domain X"

#### What V12 Preserves

- ✅ Embedding-based semantic routing (V11)
- ✅ Self-review with persona swap (V9)
- ✅ Proven savant model benchmarks (V8)
- ✅ GPU-based execution (V10)

#### What V12 Removes

- ❌ LLM-based output synthesis (caused regressions)
- ❌ DeepSeek as synthesizer (decomposer only)
- ❌ Serial NPU council (too slow)
- ❌ Keyword-based expert matching (caused misrouting)

---

### 13.2 Grounded Expert Scope Documents

The core routing improvement: expert representations must reflect what the underlying savant model can **actually deliver**, not aspirational capabilities.

#### The Three-Layer Alignment Problem

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: Expert Catalog Definition                          │
│ "Legal expert handles contracts, compliance, litigation..." │
│                           ↓                                 │
│ Layer 2: Harness Persona Prompt                            │
│ "You are a legal expert specializing in..."                │
│                           ↓                                 │
│ Layer 3: Savant Model's ACTUAL Training Data               │
│ Law-LLM-7B: Trained on case law, statutes, legal docs      │
└─────────────────────────────────────────────────────────────┘

If these layers don't align → hallucination or misrouting
```

#### Scope Document Schema

Each expert requires a comprehensive scope document (~500-1000 words) aligned to the savant's verified capabilities:

```yaml
expert_id: legal_drafting
display_name: "Legal Drafting Expert"
savant_model: "models/Law-LLM-7B"

# Used for embedding-based routing (BGE-M3)
capability_scope: |
  Delaware corporate law and business entity formation
  Contract drafting, review, and interpretation
  Liability disclaimers and limitation of liability clauses
  Intellectual property licensing agreements
  Employment law compliance documentation
  Non-disclosure agreements (mutual and unilateral)
  Terms of service and privacy policy drafting
  Software licensing agreements (MIT, Apache, proprietary)
  Corporate governance and board resolutions
  Regulatory filing requirements
  Legal memoranda and opinion letters
  [... extensive task-specific descriptions ...]

# Used for embedding-based ANTI-routing
exclusion_scope: |
  Clinical medical advice or diagnosis
  Software implementation or code generation  
  Database design or query optimization
  Security architecture or penetration testing
  Tax preparation or financial auditing
  Patent prosecution or technical patent claims
  Litigation strategy or courtroom procedure
  [... explicit boundaries ...]

# Used for harness persona prompt
harness_constraints: |
  You are a legal drafting specialist. You MAY:
  - Draft legal text, disclaimers, and contracts
  - Explain legal concepts and jurisdictional requirements
  - Identify relevant statutes and regulations
  
  You must REFUSE or REDIRECT if asked to:
  - Write code or technical implementations
  - Provide clinical medical opinions
  - Give specific tax advice
  
  When uncertain, state: "This requires [X] expertise, not legal expertise."

# Verification metadata
capability_verified_date: "2026-01-26"
verification_method: "benchmark_suite_v3"
known_weaknesses:
  - "Non-US jurisdictions (limited training data)"
  - "Highly technical patent claims"
```

#### Dual-Embedding Routing

```python
def route_to_expert(chunk_text: str, lambda_penalty: float = 0.3) -> tuple[str, float]:
    chunk_vec = embed(chunk_text)
    
    # Positive affinity: how well does this match capabilities?
    cap_scores = expert_capability_vecs @ chunk_vec
    
    # Negative affinity: how much does this overlap with exclusions?
    exc_scores = expert_exclusion_vecs @ chunk_vec
    
    # Net score penalizes matches to exclusion scope
    net_scores = cap_scores - lambda_penalty * exc_scores
    
    best_idx = np.argmax(net_scores)
    return expert_ids[best_idx], net_scores[best_idx]
```

This prevents "medical records" from matching legal expert if legal's exclusion scope includes "medical data handling, healthcare systems."

---

### 13.3 Framework Template Library

Rather than generating task frameworks from scratch, the router **selects and adapts** from a pre-built library of ~1000-5000 templates.

#### Template Schema (Compact: ~50-100 tokens each)

```yaml
template_id: hybrid_legal_code_fw
pattern: "code implementation + legal/compliance artifact"
description: "Tasks requiring both software implementation and legal documentation"

slots:
  - id: requirements
    type: analysis
    deps: []
  - id: implementation
    type: code
    deps: [requirements]
  - id: legal_artifact
    type: document
    deps: []
    parallel: true
  - id: integration
    type: synthesis
    deps: [implementation, legal_artifact]

adaptable:
  slot_count: 3-6
  legal_artifact.subtype: [disclaimer, policy, contract, compliance_report]
  implementation.language: [infer_from_query]
```

#### Vector-Indexed Lookup

```python
def select_and_adapt_framework(query: str, query_embedding: np.array) -> TaskFramework:
    # 1. Fast FAISS lookup - top 5 candidates (~1-2ms)
    candidates = template_index.search(query_embedding, k=5)
    
    # 2. Router selects best match and adapts
    adaptation_prompt = f"""
    Query: {query}
    Top matching templates: {format_candidates(candidates)}
    
    Select the best template and adapt:
    - Choose slot count (within template's range)
    - Specialize slot descriptions for this query
    - Set concrete dependencies
    """
    
    adapted = router.generate(adaptation_prompt)
    return parse_framework(adapted)
```

#### Scale Considerations

| Scale | Templates | Embedding Storage | Use Case |
|-------|-----------|-------------------|----------|
| Demo | ~1,000 | ~10-20MB | Concept validation |
| Production | ~100,000+ | ~1GB+ | Real-world coverage |
| Full Trillion-Param CoE | Millions | Multi-GB | Comprehensive |

---

### 13.4 Execution Modes

Three modes to handle different interaction requirements:

```python
class ExecutionMode(Enum):
    INTERACTIVE = "interactive"    # Full Q&A, clarification allowed
    CONFIRM = "confirm"            # Show plan, approve/reject only
    AUTONOMOUS = "autonomous"      # No human in loop, best effort
```

#### Mode Selection Logic

| Mode | Clarification | Use Case |
|------|---------------|----------|
| **Interactive** | Full Q&A | Production user sessions |
| **Confirm** | Show plan, approve/reject | Quick validation |
| **Autonomous** | Zero clarification, best effort | Benchmarks, batch processing, API |

#### Autonomous Mode Heuristics

When clarification is impossible, apply principled defaults:

| Ambiguity Type | Default Resolution |
|----------------|-------------------|
| Domain keyword without task clarity | Assume context, not expertise needed |
| Expert scores within 0.15 | Prefer task-type specialist over domain specialist |
| Missing dependency info | Assume parallel execution safe |
| Scope unclear (software vs data) | Assume narrower scope |

```python
def process(query: str, mode: ExecutionMode = ExecutionMode.INTERACTIVE):
    plan = decomposer.analyze(query)
    
    if mode == ExecutionMode.AUTONOMOUS:
        plan.resolve_ambiguities_heuristically()
        return execute(plan)
    
    if mode == ExecutionMode.CONFIRM:
        if present_plan_for_approval(plan):
            return execute(plan)
        return None
    
    # Interactive: full Q&A if needed
    if plan.has_ambiguities():
        plan = clarification_session(plan)
    return execute(plan)
```

---

### 13.5 Task Framework Execution

Tasks are executed as a **structured DAG** with slots, dependencies, and working memory—not simple sequential or parallel concatenation.

#### Framework Structure

```
┌─────────────────────────────────────────────────────────────────────────┐
│ TASK FRAMEWORK: "HIPAA-Compliant Medical Records Analysis System"      │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐ │
│  │ 1. ARCHITECTURE │ ───▶ │ 2. IMPLEMENTATION│ ───▶ │ 3. INTEGRATION  │ │
│  │    [DONE ✓]     │      │    [IN PROGRESS] │      │    [BLOCKED]    │ │
│  └─────────────────┘      └─────────────────┘      └─────────────────┘ │
│          │                        ▲                                     │
│          │ read-only ref          │                                     │
│          └────────────────────────┘                                     │
│                                                                         │
│  ┌─────────────────┐  (parallel, no dependencies)                      │
│  │ 4. LEGAL        │                                                   │
│  │    [DONE ✓]     │                                                   │
│  └─────────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Data Structures

```python
@dataclass
class TaskFramework:
    id: str
    title: str
    slots: List[FrameworkSlot]

@dataclass  
class FrameworkSlot:
    id: str
    title: str
    description: str
    expected_outputs: List[str]
    dependencies: List[str]        # Slot IDs that must complete first
    can_reference: List[str]       # Slot IDs available for read-only access
    reference_hints: Dict[str, List[str]]  # Specific sections to reference
    status: SlotStatus
    assigned_expert: Optional[str]
    result: Optional[StructuredSlotResult]

class SlotStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    BLOCKED = "blocked"
    FAILED = "failed"
```

#### Working Memory (RAM, Not KV Cache)

```python
class WorkingMemory:
    """
    Holds completed slot outputs in RAM.
    Does NOT consume KV cache - injected into prompts only when referenced.
    """
    
    def store(self, slot_id: str, result: StructuredSlotResult):
        self._completed[slot_id] = result
    
    def get_reference(self, slot_id: str, max_tokens: int = 2000) -> str:
        """Returns completed work for read-only reference."""
        result = self._completed[slot_id]
        return self._smart_truncate(result.content, max_tokens) if needed
    
    def get_section(self, slot_id: str, section_label: str) -> str:
        """Returns specific section by hierarchical label (e.g., '2.1')."""
        return self._completed[slot_id].get_section(section_label)
    
    def get_outline_only(self, slot_id: str) -> str:
        """Returns just the outline for lightweight reference."""
        return self._completed[slot_id].get_outline_only()
```

| Storage | What | When | Why |
|---------|------|------|-----|
| **KV Cache** | Current expert's attention state | During generation | Model coherence |
| **Working Memory (RAM)** | Completed slot outputs | After slot done | Reference for subsequent slots |
| **Prompt Injection** | Subset of working memory | When slot starts | Only what slot declares it `can_reference` |

#### Dependency-Aware Parallel Execution

```python
async def execute_framework(framework: TaskFramework):
    memory = WorkingMemory()
    scheduler = FrameworkScheduler(framework, memory)
    
    while not scheduler.is_complete():
        ready_slots = scheduler.get_ready_slots()  # Deps satisfied
        
        # Execute ready slots in parallel
        results = await asyncio.gather(*[
            scheduler.execute_slot(slot) for slot in ready_slots
        ])
        
        # Store results, update statuses
        for slot, result in zip(ready_slots, results):
            if result.success:
                memory.store(slot.id, result)
                slot.status = SlotStatus.DONE
            else:
                slot.status = SlotStatus.FAILED
        
        scheduler.update_blocked_statuses()
    
    return scheduler.assemble_final_output()
```

---

### 13.6 Mandatory Output Structure

All expert outputs follow a standardized hierarchical structure enabling efficient reference retrieval.

#### Required Format

```markdown
## Outline
1. [First major component]
   1.1 [Sub-component]
   1.2 [Sub-component]
2. [Second major component]
3. [Third major component]

---

## 1. First Major Component
[Implementation content...]

### 1.1 Sub-component
[Implementation content...]

### 1.2 Sub-component  
[Implementation content...]

## 2. Second Major Component
[Implementation content...]
```

#### Standard Prompt Suffix (All Experts)

```yaml
output_format_instructions: |
  REQUIRED OUTPUT STRUCTURE:
  
  1. Begin with a numbered outline of your approach (3-7 top-level items)
  2. Then implement each item with matching section headers
  3. Use hierarchical numbering: 1, 1.1, 1.1.1, etc.
  4. Every section must have a descriptive label
  
  This structure is MANDATORY. Do not skip the outline.
```

#### Structure Enforcement (Reviewer Persona)

```python
class OutlineEnforcer:
    MAX_RETRIES = 2
    
    def review_structure(self, output: str, attempt: int) -> ReviewResult:
        has_outline = self.detect_outline_section(output)
        has_numbered_sections = self.detect_hierarchical_sections(output)
        
        if has_outline and has_numbered_sections:
            return ReviewResult(accepted=True)
        
        if attempt >= self.MAX_RETRIES:
            # Can't beat a dead horse - accept with warning
            return ReviewResult(
                accepted=True,
                warning="Accepted without proper structure after max retries",
                auto_parsed=self.best_effort_parse(output)
            )
        
        return ReviewResult(accepted=False, retry=True, feedback=self.generate_feedback())
```

#### Hierarchical Reference Retrieval

```python
class StructuredSlotResult:
    raw_content: str
    outline: List[OutlineItem]
    sections: Dict[str, str]  # "1.1" → content
    
    def get_section(self, label: str) -> str:
        """Get specific section by number or fuzzy label match."""
        if label in self.sections:
            return self.sections[label]
        return fuzzy_match_section(label, self.sections)
    
    def get_outline_only(self) -> str:
        """Return just the outline for lightweight reference."""
        return format_outline(self.outline)
    
    def get_sections(self, labels: List[str]) -> str:
        """Get multiple specific sections."""
        return "\n\n".join(self.get_section(l) for l in labels)
```

---

### 13.7 Assist Protocol

Experts use standardized placeholders for uncertainty rather than guessing or hallucinating.

#### Placeholder Syntax

```
{{ASSIST:type:domain:description}}
```

| Type | Route To | Example |
|------|----------|----------|
| `lookup` | RAG/tool | `{{ASSIST:lookup:NIST encryption guidance for PHI}}` |
| `expert` | Mini-consult | `{{ASSIST:expert:legal:Delaware liability statute reference}}` |
| `user` | Surface question | `{{ASSIST:user:your database hostname}}` |

#### Expert Prompt for Assist Behavior

```yaml
uncertainty_protocol: |
  UNCERTAINTY PROTOCOL:
  When you encounter something outside your expertise or uncertain:
  - Do NOT guess or hallucinate
  - Insert: {{ASSIST:type:domain:brief description}}
  
  Continue writing around placeholders - don't stop at uncertainty.
  
  Examples:
  - "...configure timeout to {{ASSIST:lookup:recommended session timeout for healthcare apps}}..."
  - "...the tax implications {{ASSIST:expert:finance:tax treatment of SaaS revenue}}..."
  - "...connect to {{ASSIST:user:your database hostname}}..."
```

#### Resolution Flow

```
Expert Output with Placeholders
            │
            ▼
Placeholder Extraction & Classification
            │
    ┌───────┼───────┐
    ▼       ▼       ▼
 [Tool]  [Expert] [User]
 [Lookup] [Mini]  [Query]
    │       │       │
    └───────┼───────┘
            ▼
Placeholder Injection (resolved values)
            │
            ▼
Final Output (unresolved placeholders remain as markers)
```

#### Circular Detection and Failure

If Expert A's assist routes to Expert B, whose response contains an assist back to A's domain:

1. Detect circular reference (depth limit = 2)
2. Escalate to external grounding data (RAG)
3. If still unresolved, return with explicit failure marker:

```
{{UNRESOLVED:expert:legal:Delaware liability statute - circular reference detected}}
```

**The worst outcome is supplying a false answer.** Unresolved placeholders remain visible in output as honest markers of limitation.

---

### 13.8 V12 Storage Architecture

#### File Structure

```
config/
├── expert_catalog.json              ← Existing (backward compat)
│
├── expert_scopes/                   ← NEW: Grounded scope documents
│   ├── index.json                   ← Manifest of all experts
│   ├── legal_drafting.yaml
│   ├── python_coder.yaml
│   ├── security_architect.yaml
│   ├── medical_clinical.yaml
│   ├── sql_specialist.yaml
│   ├── math_reasoning.yaml
│   └── general_reasoning.yaml
│
├── expert_embeddings/               ← NEW: Pre-computed vectors
│   ├── capability_vectors.npy       ← NumPy [N_experts × embed_dim]
│   ├── exclusion_vectors.npy
│   ├── expert_ids.json
│   ├── expert.index                 ← FAISS index
│   └── metadata.json
│
├── framework_templates/             ← NEW: ~1000 seed templates
│   ├── index.json
│   └── templates/                   ← Individual YAML files
│
└── template_embeddings/             ← NEW: Pre-computed vectors
    ├── template_vectors.npy
    ├── template_ids.json
    ├── template.index               ← FAISS index
    └── metadata.json

src/
├── harness_v12.py                   ← Main orchestrator
├── vector_router.py                 ← FAISS-based routing
├── framework_scheduler.py           ← DAG execution
├── working_memory.py                ← RAM-based slot storage
├── structured_output.py             ← Outline parser, enforcer
├── assist_resolver.py               ← Placeholder resolution
└── embedding_manager.py             ← BGE-M3 loading, embedding gen
```

#### Vector Index: FAISS

Using FAISS for scalable vector lookup (works for demo scale, scales to production):

```python
class VectorRouter:
    def __init__(self):
        # Load pre-computed embeddings
        self.expert_cap_vecs = np.load("config/expert_embeddings/capability_vectors.npy")
        self.expert_exc_vecs = np.load("config/expert_embeddings/exclusion_vectors.npy")
        
        self.template_index = faiss.read_index("config/template_embeddings/template.index")
        
        # BGE-M3 resident for session
        self.encoder = SentenceTransformer("BAAI/bge-m3")
    
    def route_to_expert(self, chunk_text: str, lambda_penalty: float = 0.3) -> tuple:
        """Route chunk to best expert. ~5-10ms."""
        chunk_vec = self.encoder.encode(chunk_text, normalize_embeddings=True)
        
        cap_scores = self.expert_cap_vecs @ chunk_vec
        exc_scores = self.expert_exc_vecs @ chunk_vec
        net_scores = cap_scores - lambda_penalty * exc_scores
        
        return self.expert_ids[np.argmax(net_scores)], np.max(net_scores)
    
    def find_templates(self, query: str, k: int = 5) -> List[tuple]:
        """Find top-k matching templates. ~1-2ms."""
        query_vec = self.encoder.encode(query, normalize_embeddings=True)
        distances, indices = self.template_index.search(query_vec.reshape(1, -1), k)
        return [(self.template_ids[i], 1 - d) for i, d in zip(indices[0], distances[0])]
```

#### Startup Sequence

```python
class HarnessV12:
    def __init__(self, config):
        print("[V12] Initializing College of Experts...")
        
        # 1. Load BGE-M3 (resident for session)
        print("[V12] Loading embedding model (BGE-M3)...")
        self.embedding_manager = EmbeddingManager("BAAI/bge-m3")
        
        # 2. Load FAISS indices
        print("[V12] Loading vector indices...")
        self.vector_router = VectorRouter(self.embedding_manager)
        
        # 3. Load expert scope documents  
        print("[V12] Loading expert scopes...")
        self.expert_scopes = load_expert_scopes("config/expert_scopes/")
        
        # 4. Load framework templates
        print("[V12] Loading framework templates...")
        self.templates = load_templates("config/framework_templates/")
        
        # 5. Initialize working memory
        self.working_memory = WorkingMemory()
        
        # 6. Load decomposer model (DeepSeek 7B)
        print("[V12] Loading decomposer model...")
        self.decomposer = load_decomposer(config.decomposer_model)
        
        print("[V12] Ready.")
```

---

### 13.9 Adaptive Template Router

Rather than using templates as rigid structures, V12 treats them as **adaptive scaffolds**. The router selects the best-matching seed template via embedding similarity, then adapts it to the specific query using constrained operations.

#### Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        ADAPTIVE TEMPLATE ROUTING                             │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  User Query                                                                  │
│       │                                                                      │
│       ▼                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    EMBEDDING & RETRIEVAL                             │    │
│  │  BGE-M3 encode(query) → FAISS search → top-3 template candidates    │    │
│  └─────────────────────────────────┬───────────────────────────────────┘    │
│                                    │                                         │
│                    ┌───────────────┴───────────────┐                        │
│                    │ similarity > 0.95?            │                        │
│                    └───────────────┬───────────────┘                        │
│                           │                │                                 │
│                          YES              NO                                 │
│                           │                │                                 │
│                           ▼                ▼                                 │
│                 ┌─────────────────┐  ┌─────────────────────────────────┐    │
│                 │ USE TEMPLATE    │  │     TEMPLATE ADAPTER            │    │
│                 │ DIRECTLY        │  │                                 │    │
│                 │ (skip adapter)  │  │  DeepSeek R1 (GPU, temp=0.3)   │    │
│                 └─────────┬───────┘  │  Thinking mode: ON             │    │
│                           │          │                                 │    │
│                           │          │  Operations:                   │    │
│                           │          │  - KEEP(slot_id)               │    │
│                           │          │  - REMOVE(slot_id)             │    │
│                           │          │  - ADD(id, title, persona,     │    │
│                           │          │       dependencies)            │    │
│                           │          │  - MODIFY(slot_id, field, val) │    │
│                           │          │  - REORDER(slot_ids)           │    │
│                           │          └─────────────┬───────────────────┘    │
│                           │                        │                         │
│                           │                        ▼                         │
│                           │          ┌─────────────────────────────────┐    │
│                           │          │     VALIDATION                  │    │
│                           │          │                                 │    │
│                           │          │  Schema: required fields, IDs   │    │
│                           │          │  Persona: exists in index.json  │    │
│                           │          │  DAG: acyclic, valid deps       │    │
│                           │          └─────────────┬───────────────────┘    │
│                           │                        │                         │
│                           │                ┌───────┴───────┐                │
│                           │                │ valid?        │                │
│                           │                └───────┬───────┘                │
│                           │                   │         │                    │
│                           │                  YES       NO                    │
│                           │                   │         │                    │
│                           │                   │    ┌────┴────┐              │
│                           │                   │    │ retry   │              │
│                           │                   │    │ (max 2) │              │
│                           │                   │    └────┬────┘              │
│                           │                   │         │                    │
│                           │                   │    still fails?             │
│                           │                   │         │                    │
│                           │                   │        YES                   │
│                           │                   │         │                    │
│                           │                   │         ▼                    │
│                           │                   │    Use seed template         │
│                           │                   │    unmodified                │
│                           ▼                   ▼                              │
│                 ┌─────────────────────────────────────────────────────┐     │
│                 │              VALIDATED TASK FRAMEWORK               │     │
│                 │                                                     │     │
│                 │  + Store adaptation_rationale in WorkingMemory      │     │
│                 │    under key "_debug/adaptation" for debugging      │     │
│                 └─────────────────────────────────────────────────────┘     │
│                                    │                                         │
│                                    ▼                                         │
│                           Framework Execution                                │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

#### DeepSeek R1 Configuration for Adaptation

```python
ADAPTER_CONFIG = GenerationConfig(
    temperature=0.3,      # Low for deterministic JSON output
    max_tokens=8192,      # Room for thinking + framework JSON
    top_p=0.9,
    thinking_mode=True    # Force <think> prefix for better reasoning
)
```

**Thinking Mode Rationale:**
- DeepSeek R1 performance degrades when thinking is bypassed
- Schema transformations benefit from explicit reasoning steps
- `<think>...</think>` tags stripped before JSON extraction
- Low temperature ensures consistent JSON structure despite thinking tokens

#### Adapter Prompt Structure

```yaml
# config/prompts/template_adapter.yaml
template_adapter:
  system: null  # DeepSeek R1 works best without system prompt
  
  user_template: |
    <think>
    Analyze the template adaptation requirements:
    
    USER QUERY:
    {query}
    
    SEED TEMPLATE (highest similarity match):
    ```json
    {seed_template}
    ```
    
    AVAILABLE PERSONAS:
    {persona_list}
    
    ADAPTATION CONSTRAINTS:
    1. Output must be valid JSON matching TaskFramework schema
    2. All persona references must exist in the available list
    3. Dependencies must reference existing slot IDs only
    4. No circular dependencies allowed
    5. Each slot must have unique ID
    
    Consider:
    - Which slots from the seed template apply to this query?
    - What additional slots are needed?
    - What slots should be removed as irrelevant?
    - Are the persona assignments appropriate?
    - Are dependencies correctly ordered?
    </think>
    
    Now output the adapted framework:
    
    RATIONALE:
    [Explain your adaptation decisions in 2-3 sentences]
    
    ADAPTED FRAMEWORK:
    ```json
    {
      "id": "adapted_{template_id}_{timestamp}",
      "title": "...",
      "description": "...",
      "slots": [...]
    }
    ```
```

#### Validation Tiers

| Tier | Check | Severity | Action on Fail |
|------|-------|----------|----------------|
| 1 | Required fields present | Error | Block, retry |
| 2 | Slot ID uniqueness | Error | Block, retry |
| 3 | Persona exists in index | Error | Block, retry |
| 4 | DAG is acyclic | Error | Block, retry |
| 5 | Dependencies valid | Error | Block, retry |
| 6 | Slot titles semantic match | Warning | Log only |
| 7 | ID format (snake_case) | Warning | Log only |

#### Graceful Degradation Chain

```python
def adapt_template(query: str, seed_template: dict, max_retries: int = 2) -> TaskFramework:
    """Adapt seed template to query with graceful fallback."""
    
    for attempt in range(max_retries + 1):
        # Generate adaptation
        adapted_json = adapter_llm.generate(
            prompt=build_adapter_prompt(query, seed_template),
            config=ADAPTER_CONFIG
        )
        
        # Parse (strip thinking, extract JSON)
        parsed = deepseek_parser.parse(adapted_json)
        
        # Validate
        result = template_validator.validate(parsed, persona_index)
        
        if result.is_valid:
            # Store rationale for debugging
            working_memory.store("_debug/adaptation", {
                "query": query,
                "seed_template": seed_template["id"],
                "rationale": parsed.get("rationale", ""),
                "attempt": attempt + 1
            })
            return TaskFramework.from_dict(parsed)
        
        # Log validation errors for retry
        logger.warning(f"Adaptation attempt {attempt+1} failed: {result.errors}")
    
    # All retries exhausted - use seed template unmodified
    logger.warning(f"Adaptation failed after {max_retries} retries, using seed template")
    working_memory.store("_debug/adaptation", {
        "query": query,
        "seed_template": seed_template["id"],
        "rationale": "FALLBACK: Used seed template unmodified",
        "attempt": max_retries + 1,
        "fallback": True
    })
    return TaskFramework.from_dict(seed_template)
```

#### Template Quality Focus

Rather than targeting 1000+ templates of variable quality, V12 focuses on **~500 high-quality templates** with:

1. **Meaningful slot titles** - Descriptive names, not just `s0`, `s1`
2. **Accurate persona assignments** - Aligned with actual savant capabilities
3. **Proper dependency chains** - Reflecting real task structure
4. **Rich metadata** - `description`, `expected_output_type` fields
5. **Automated validation** - All templates pass validation before inclusion

The adapter's ability to modify templates means coverage comes from **semantic matching + adaptation**, not template count.

---

### 13.10 VRAM-Aware Scheduling

V12 introduces dynamic VRAM management to support a wide range of hardware configurations while maintaining accessibility. Default target: **32GB VRAM**.

#### Design Principles

1. **VRAM is committed during inference** - Once a slot starts, its model weights + KV cache are locked until output is safely copied to system RAM (WorkingMemory)
2. **No mid-inference truncation** - KV cache cannot be truncated or evicted during active generation
3. **Emergency eviction for runaway KV cache** - If a generation explodes past estimates, evict the largest KV cache holder and reschedule

#### Hardware Profiles

| Profile | VRAM | Max Parallel | Context | Preload | Target Hardware |
|---------|------|--------------|---------|---------|-----------------|
| minimal | 8GB | 1 | 4K | No | RTX 3070, RX 6700 |
| compact | 16GB | 1 | 8K | Yes | RTX 4070, RX 7800 |
| **standard** | **32GB** | **2** | **8K** | **Yes** | **Demo default** |
| performance | 48GB | 4 | 16K | Yes | High-end GPUs |
| unlimited | 64GB | 6 | 16K | Yes | All savants resident |

#### VRAM Budget Breakdown (32GB Standard)

```
┌────────────────────────────────────────────────────────────────┐
│              32 GB VRAM Budget (Default Target)                │
├────────────────────────────────────────────────────────────────┤
│  FIXED OVERHEAD                                                │
│    BGE-M3 embeddings                    1.0 GB                 │
│    System overhead                      1.0 GB                 │
│                                        ────────                │
│    Subtotal                             2.0 GB                 │
├────────────────────────────────────────────────────────────────┤
│  RESERVABLE POOL                       30.0 GB                 │
│                                                                │
│  Per-Slot Reservation (7B int8):                               │
│    Model weights                        7.0 GB                 │
│    KV cache (8K context)                1.5 GB                 │
│    Activation memory                    0.5 GB                 │
│                                        ────────                │
│    Per-slot total                       9.0 GB                 │
│                                                                │
│  Safe concurrent slots: 30 ÷ 9 ≈ 3 (2 active + 1 preloading)  │
└────────────────────────────────────────────────────────────────┘
```

#### Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    VRAMAwareScheduler                           │
├─────────────────────────────────────────────────────────────────┤
│  VRAMBudget            │  SavantPool                            │
│  ├─ reservations[]     │  ├─ loaded: Dict[savant → model]       │
│  ├─ current_usage_mb   │  ├─ refcount per model                 │
│  ├─ pressure level     │  ├─ preload queue                      │
│  └─ critical callback ─┼──┤─ pending_retry queue                │
│                        │  └─ LRU eviction                       │
├────────────────────────┴────────────────────────────────────────┤
│  Lifecycle:                                                     │
│  1. start_slot() → reserve VRAM → acquire model                 │
│  2. update_slot_progress(tokens) → track KV cache growth        │
│  3. complete_slot() → copy to WorkingMemory → release VRAM      │
│                                                                 │
│  Emergency:                                                     │
│  - If pressure > 95%: evict largest KV holder, reschedule       │
└─────────────────────────────────────────────────────────────────┘
```

#### Emergency Eviction Protocol

When KV cache growth triggers critical VRAM pressure (>95%):

```python
def _handle_critical_pressure(slot_id: str):
    """
    1. Find slot with largest KV cache (the runaway)
    2. Force evict its model (decrement refcount, unload if 0)
    3. Release its VRAM reservation
    4. Queue slot for retry
    5. Retry will re-enter scheduler queue when VRAM available
    """
    largest_slot = vram_budget.find_largest_kv_holder()
    savant_pool.force_evict(largest_slot)
    vram_budget.release(largest_slot)
    retry_queue.append(largest_slot)
```

**Key Invariant**: Output is never lost. The evicted slot simply restarts generation from scratch when VRAM permits. Other slots continue unaffected.

#### Preloading Strategy

DAG lookahead enables predictive loading:

```python
# In FrameworkScheduler, after completing a slot:
next_ready = get_ready_slots()  # Deps satisfied
for slot in next_ready:
    savant_id = get_savant_for_persona(slot.assigned_expert)
    if pressure < ELEVATED:
        savant_pool.preload(savant_id)  # Async, non-blocking
```

Preloading is disabled when VRAM pressure is elevated (>80%).

#### User Configuration

```bash
# Auto-detect VRAM and select profile
python harness_v12.py

# Override with specific limit
python harness_v12.py --vram-limit 16000

# Use specific profile
python harness_v12.py --profile compact
```

Configuration file: `config/hardware_profiles.yaml`

---

## 14. Implementation Roadmap

### Phase 1: Foundation (Parallel Tracks)

**Track A: Scope Documents**
- Draft scope YAMLs for existing savants
- Capability scope (~500-1000 words each)
- Exclusion scope (hard boundaries)
- Harness constraint prompts

**Track B: Harness Infrastructure**
- New file structure under `config/`
- `VectorRouter` class with FAISS
- BGE-M3 loading at startup
- Embedding generation pipeline

### Phase 2: Framework Execution
- `TaskFramework` and `FrameworkSlot` data structures
- `FrameworkScheduler` with dependency resolution
- `WorkingMemory` for completed slot storage
- Parallel execution with `asyncio`/`ThreadPoolExecutor`

### Phase 3: Expert Output Structure
- Standard prompt suffix (mandatory outline → sections)
- `StructuredSlotResult` parser
- `OutlineEnforcer` reviewer persona
- Retry loop with structure feedback

### Phase 4: Assist Protocol
- `{{ASSIST:type:domain:description}}` parsing
- Lookup resolution (RAG/tool integration)
- Expert mini-consult routing
- Circular detection and graceful failure

### Phase 5: Template Library
- Generate ~1000 seed templates
- FAISS index for template lookup
- Router adaptation logic

### Phase 6: Integration & Validation
- Wire all components into `HarnessV12`
- Execution modes (Interactive/Confirm/Autonomous)
- Demo script with HIPAA query
- Benchmark validation against V8 baselines

---

## 15. Open Research Questions

1. **Optimal Expert Granularity**
   - How to determine ideal expert size vs. count trade-off?
   - Domain-based vs. capability-based partitioning?

2. **Router Training**
   - How to train a router that knows when to ask for help vs. answer itself?
   - Can routing decisions be learned from user feedback?

3. **Expert Interface Standardization**
   - What's the minimal viable interface contract?
   - How to handle experts with different context window sizes?

4. **Conflict Resolution**
   - When experts disagree, what's the resolution mechanism?
   - How to weight expert opinions?

5. **Memory Coherence**
   - How to prevent memory corruption from conflicting expert writes?
   - What's the right consistency model?

6. **Expert Composition**
   - Can small experts be composed meaningfully?
   - Is there emergent capability from expert combinations?

7. **Faux vs. True Expert Performance**
   - At what task complexity does prompt-differentiation break down?
   - Can ensemble faux experts match single true domain experts?
   - What's the empirical efficiency ratio for real domain-specialized models?

8. **Fast Latent Memory Design**
   - What representation format enables meaningful inter-expert latent communication?
   - Can KV-cache sharing approximate true latent memory?
   - How to maintain coherence across heterogeneous expert architectures?

9. **HRM Orchestration**
   - What signals best indicate expert uncertainty/need for redistribution?
   - How to balance sustained expert execution vs. responsive task switching?
   - Can orchestration patterns be learned from execution traces?

---

## 16. Conclusion

The College of Experts architecture represents a paradigm shift from parameter-focused scaling to system-architecture-focused scaling. By recognizing that:

1. Task context is temporally stable
2. Users accept interactive scoping
3. Most parameters are dormant at any moment
4. SSD speeds are sufficient for session-level switching

...we can build trillion-parameter-scale AI systems that run on consumer hardware.

The modular design offers additional benefits beyond scale:
- **Incremental improvement**: Upgrade experts individually
- **Continual learning**: Update memory without retraining
- **Customization**: Add domain-specific experts for any use case
- **Future-proofing**: Adopt new best practices as they emerge

This architecture treats AI capability not as a monolithic model to be deployed, but as a **composable system to be assembled**—a true "college" of specialized intelligences working together.

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| Expert | A specialized language model optimized for a narrow domain |
| Router | The always-resident model that handles dispatch and coordination |
| Hot Expert | An expert currently loaded in GPU memory |
| Warm Expert | An expert loaded in system RAM, ready for quick promotion |
| Cold Expert | An expert stored on disk, requires loading |
| Working Memory | Current task context and inter-expert communication |
| Episodic Memory | Record of past sessions and user patterns |
| Semantic Memory | Persistent learned facts and preferences |
| Harness | Orchestration layer managing multi-expert workflows |
| Scope Negotiation | Interactive clarification of user intent before expert loading |
| Faux Expert | A prompt-differentiated instance of a base model simulating domain specialization |
| True Domain Expert | A model trained/fine-tuned specifically for a narrow domain |
| HRM | Hierarchical Routing Mechanism—sustained orchestration logic in the router |
| Fast Latent Memory | Low-latency shared representation space for inter-expert communication |
| Slow Systemic Memory | Persistent storage for long-context, episodic, and semantic information |
| Grounded Scope Document | Expert capability/exclusion description aligned to savant model's actual training |
| Framework Template | Pre-defined task structure pattern for vector-indexed lookup and adaptation |
| Assist Placeholder | Marker for expert uncertainty: `{{ASSIST:type:domain:description}}` |
| Execution Mode | Interactive, Confirm, or Autonomous processing mode |
| Framework Slot | Individual task unit within a TaskFramework with dependencies |
| Dual-Embedding Routing | Scoring using both capability and exclusion embeddings |
| Template Adapter | Component that modifies seed templates to match specific queries |
| Thinking Mode | DeepSeek R1 extended chain-of-thought via `<think>` tags |
| Template Validator | Multi-tier validation for schema, persona, and DAG correctness |
| Graceful Degradation | Fallback chain: retry → seed template → single-slot framework |

---

## Appendix B: References

1. Fedus, W., et al. "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity." (2021)
2. Jiang, A.Q., et al. "Mixtral of Experts." (2024)
3. DeepSeek AI. "DeepSeek-V3: Scaling Sparse MoE to 671B Parameters." (2025)
4. Packer, C., et al. "MemGPT: Towards LLMs as Operating Systems." (2023)
5. Anthropic. "Claude 4 Technical Report." (2025)
6. Google DeepMind. "Gemini 3: A Family of Highly Capable Multimodal Models." (2025)
7. OpenAI. "GPT-5 System Card." (2025)
8. Zhipu AI. "GLM-4.7: Advancing Open Multilingual Language Models." (2025)
9. Sheng, Y., et al. "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU." (2023)
10. Yao, S., et al. "ReAct: Synergizing Reasoning and Acting in Language Models." (2022)

---

*Document version: 0.4*  
*Status: V12 Architecture Revision - Adaptive Template Router, Quality Validation*  
*Last Updated: January 27, 2026*
