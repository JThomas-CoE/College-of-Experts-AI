# College of Experts: A Disk-Resident Sparse Mixture Architecture for Local Trillion-Parameter Inference

**Design Document v0.1**  
**Date: January 2026**

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
7. [Continual Learning and Adaptation](#7-continual-learning-and-adaptation)
8. [Modular Evolution and Upgradeability](#8-modular-evolution-and-upgradeability)
9. [Hardware Requirements and Performance Analysis](#9-hardware-requirements-and-performance-analysis)
10. [Comparison with Existing Approaches](#10-comparison-with-existing-approaches)
11. [Implementation Roadmap](#11-implementation-roadmap)
12. [Open Research Questions](#12-open-research-questions)
13. [Conclusion](#13-conclusion)

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

## 7. Continual Learning and Adaptation

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

## 8. Modular Evolution and Upgradeability

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

## 9. Hardware Requirements and Performance Analysis

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

## 10. Comparison with Existing Approaches

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

## 11. Implementation Roadmap

### Phase 1: Proof of Concept (MVP)
- Simple router (Qwen 3:4b, liquied 2.5, Phi 4, granite)
- 3-5 domain experts (coding, writing, analysis)
- File-based memory (JSON/SQLite)
- Basic CLI interface
- Manual expert selection

### Phase 2: Core System
- Full router with scope negotiation
- Expert manifest and auto-selection
- Hierarchical caching (RAM + SSD)
- Structured memory backbone
- Simple agentic harness

### Phase 3: Production System
- Robust multi-expert coordination
- Advanced memory (vector search, temporal)
- Plugin architecture for new experts
- IDE/API integrations
- Performance optimizations

### Phase 4: Ecosystem
- Expert marketplace/registry
- Community expert contributions
- Fine-tuning pipeline for custom experts
- Enterprise deployment tools

---

## 12. Open Research Questions

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

---

## 13. Conclusion

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

*Document version: 0.1*  
*Status: Initial Draft*  
*Feedback welcome*
