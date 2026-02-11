# College of Experts (CoE)

> A disk-resident sparse mixture architecture for running trillion-parameter-scale AI locally

[![License: PolyForm Noncommercial](https://img.shields.io/badge/License-PolyForm_Noncommercial-purple.svg)](https://polyformproject.org/licenses/noncommercial/1.0.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

## Overview

**College of Experts** is an architectural approach to deploying massively scaled AI systems on consumer hardware. Instead of loading entire trillion-parameter models into memory, CoE:

1. **Stores specialized "expert" models on SSD** (NVMe)
2. **Dynamically loads only the experts needed** for the current task
3. **Uses a lightweight router** for scope negotiation and expert dispatch
4. **Shares memory across experts** for continuity and learning

The key insight: **task context changes slowly**. A user working on code stays in "code mode" for minutes to hours—plenty of time to load specialized experts from fast storage.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    USER INTERFACE                               │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    AGENTIC HARNESS                              │
│         (Task decomposition, tool orchestration)                │
└──────────────────────────┬──────────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ROUTER (~1-3B)                               │
│    • Scope negotiation    • Expert selection                    │
│    • Lightweight queries  • Transition detection                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
         ┌─────────────────┼─────────────────┐
         ▼                 ▼                 ▼
┌─────────────┐   ┌─────────────────┐   ┌─────────────┐
│   SHARED    │   │  EXPERT POOL    │   │   STORAGE   │
│   MEMORY    │   │  (GPU/RAM)      │   │   (SSD)     │
│             │   │                 │   │             │
│ • Working   │   │ [Code Expert]   │   │ 100+ cold   │
│ • Episodic  │   │ [Math Expert]   │   │ experts     │
│ • Semantic  │   │ [...]           │   │             │
└─────────────┘   └─────────────────┘   └─────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.ai/) installed and running
- NVMe SSD recommended (SATA SSD works, slower loading)
- 16GB+ RAM, GPU optional but recommended

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/college-of-experts.git
cd college-of-experts
pip install -r requirements.txt
```

### Download Expert Models

```bash
# Router model (always in memory)
ollama pull qwen3:4b

# Demo experts
ollama pull deepseek-coder-v3:33b
ollama pull qwen3-math:7b
ollama pull llama4:8b
```

### Run the Demo

```bash
python demo.py
```

## Project Structure

```
college-of-experts/
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── requirements.txt
├── docs/
│   ├── design.md              # Full design document (white paper)
│   └── experts.md             # Recommended expert models
├── src/
│   ├── __init__.py
│   ├── router.py              # Router implementation
│   ├── expert_loader.py       # Dynamic expert loading
│   ├── memory_backbone.py     # Shared memory layer
│   └── harness.py             # Agentic orchestration
├── experts/
│   └── manifests/             # Expert metadata
├── demo.py                    # Interactive demo
└── tests/
    └── test_router.py
```

## Recommended Expert Models

See [docs/experts.md](docs/experts.md) for a curated list of specialized models suitable for use as experts.

| Domain | Model | Size | Notes |
|--------|-------|------|-------|
| **Code** | DeepSeek-Coder-V3 | 7B/33B/236B | Excellent code generation, agentic coding |
| **Math** | Qwen3-Math | 3B-72B | Strong mathematical reasoning |
| **General** | Qwen3 | 4B-72B | Good general router/fallback |
| **Medical** | OpenBioLLM-2 | 8B/70B | Biomedical domain |
| **Legal** | SaulLM-2 | 7B/54B | Legal document analysis |

## Design Philosophy

### MVP: Faux Experts First

The initial implementation uses **prompt-differentiated experts**—the same capable base model with specialized system prompts that activate domain-specific behavior:

```python
# Same model, different personas
experts = {
    "python": {"model": "qwen2.5:7b", "prompt": "You are a Python specialist..."},
    "security": {"model": "qwen2.5:7b", "prompt": "You are a security expert..."},
    "architect": {"model": "qwen2.5:7b", "prompt": "You are a software architect..."}
}
```

**Why start here:**
- Instant expert switching (no model loading)
- Validates orchestration architecture
- Modern instruct models have latent multi-domain capabilities
- True specialized experts become drop-in upgrades

### The Domain Expert Efficiency Hypothesis

Long-term, CoE anticipates **true domain experts** with dramatic efficiency gains:

| Model | Parameters | Domain Performance |
|-------|------------|-------------------|
| General 200B | 200B | Baseline |
| Domain Expert 4B | 4B | ≈Equal to 200B *in-domain* |

A 4B Python specialist could match a 200B generalist at coding—by dedicating 100% of parameters to that domain. This is the **sparse expertise at scale** vision.

### Why Not Traditional MoE?

Traditional Mixture of Experts loads all experts into memory for token-level routing. This limits practical size to available GPU memory.

CoE routes at the **task level**, not token level. Since task context is stable for extended periods, we can load experts from fast SSD storage without impacting user experience.

### The Scope Negotiation Pattern

Instead of guessing which experts are needed, the router conducts a brief dialog:

```
User: "Help with my project"
Router: "What kind of project? (code, writing, analysis...)"
User: "Python backend with database"
Router: [Loads Python + SQL experts while providing loading feedback]
```

This masks loading latency and ensures the right experts are ready.

### HRM: Router as Orchestrator

The Router houses **Hierarchical Routing Mechanism** (HRM) logic for sustained task execution:
- Decomposes complex tasks into expert-appropriate subtasks
- Monitors execution and redistributes on uncertainty
- Synthesizes multi-expert outputs coherently

### Shared Memory Backbone

All experts read from and write to a common memory layer:
- **Working Memory**: Current task context
- **Episodic Memory**: Past sessions, what worked
- **Semantic Memory**: Learned facts, user preferences

This enables continual learning without model retraining.

> **Note:** Fast latent-space memory for inter-expert communication is deferred to future phases. MVP uses text-based communication.

## Roadmap

### Phase 0: Faux Expert MVP (Current)
- [x] Design document
- [ ] Faux expert system prompts (5-7 personas)
- [ ] Simple HRM orchestration in router
- [ ] Text-based inter-expert communication
- [ ] Basic CLI demo

### Phase 1: Core System
- [ ] Multiple model backend support
- [ ] Expert manifest and auto-selection
- [ ] Simple shared memory (SQLite)
- [ ] Hierarchical caching (RAM + SSD)

### Phase 2+: Future
- [ ] True domain expert integration
- [ ] Fast latent memory (research)
- [ ] Memory backbone with vector search
- [ ] IDE integration

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. We welcome:
- Expert model recommendations
- Memory layer implementations
- Router improvements
- Documentation and examples

## License

MIT License - see [LICENSE](LICENSE) for details.

## References

- [Design Document](docs/design.md) - Full architectural specification
- [Switch Transformers](https://arxiv.org/abs/2101.03961) - Foundational MoE work
- [MemGPT](https://github.com/cpacker/MemGPT) - Memory layer inspiration
- [Ollama](https://ollama.ai/) - Local model serving

---

*"A college of specialized intelligences working together."*
