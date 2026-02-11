# V13 Memory Architecture - 3 Tier + Future Tier 0

## Original Vision Mapping

| Tier | Purpose | Current Implementation | Status |
|------|---------|------------------------|--------|
| **Tier 0** | Model-to-model latent space sharing | Shared latent tensors during inference | FUTURE |
| **Tier 1** | Working Memory - finished tasks, read-only downstream | `src/working_memory.py` | ✅ Implemented |
| **Tier 2** | Local Knowledge Base - ground truth "encyclopedia" | `src/knowledge_layer.py` - LocalKnowledgeBase | ✅ Implemented |
| **Tier 3** | Online Resources - frequently updated curated material | `src/knowledge_layer.py` - Web search enabled | ✅ Implemented |

## Current Implementation Analysis

### Tier 1: Working Memory ✅
**File**: [`src/working_memory.py`](src/working_memory.py)

```python
# Current features match your vision:
- RAM-based storage (fast access)
- Stores StructuredSlotResult (finished task outputs)
- Hierarchical reference (outline, sections, full)
- Read-only access for downstream slots
- Signature extraction for code dependencies
```

**Used in demo_v13.py**:
```python
memory = WorkingMemory()
memory.store(slot_id, expert_id, resolved_content)  # Store finished slot
dep_sigs = memory.get_all_signatures(slot.dependencies)  # Read-only ref
```

### Tier 2: Local Knowledge Base ✅
**File**: [`src/knowledge_layer.py`](src/knowledge_layer.py) - `LocalKnowledgeBase` class

```python
# Ground truth storage:
- SQLite/FAISS based
- Chunked document storage
- Semantic search via embeddings
- Embeddings cached for fast retrieval
- Categories for organization
```

**Initialization**:
```python
knowledge_retriever = KnowledgeRetriever(
    embedding_fn=embedding_manager.encode,
    knowledge_base_dir="data/knowledge",
    enable_web_search=False  # Tier 3 off
)

# Retrieve from Tier 2
results = await knowledge_retriever.retrieve(
    query="SQL best practices",
    use_local=True  # Tier 2
)
```

### Tier 3: Online Resources ✅
**File**: [`src/knowledge_layer.py`](src/knowledge_layer.py) - web search integration

```python
# Currently supports:
- Web search (via search API)
- Retrieved content processed same as Tier 2
- Can be enabled/disabled per query
```

**Usage**:
```python
knowledge_retriever = KnowledgeRetriever(
    enable_web_search=True  # Enable Tier 3
)

results = await knowledge_retriever.retrieve(
    query="latest Python 3.12 features",
    use_web=True  # Tier 3
)
```

### Tier 0: Latent Space Sharing 🔮 FUTURE
**File**: Not yet implemented

```python
# Vision: Model-to-model tensor sharing
- Shared KV-cache segments
- Cross-attention between experts
- Latent space communication protocol
- Requires: Custom model architecture or framework support
```

**Current blocker**: Standard OGA/transformers don't support cross-model KV sharing. Would need:
- Custom inference engine
- Standardized latent representation format
- Memory coherence protocols

## Integration in demo_v13.py

All three tiers are already integrated:

```python
# Tier 1: Working Memory
memory = WorkingMemory()
knowledge_retriever.set_working_memory(memory)  # Link Tier 1

# Tier 2 & 3 via KnowledgeRetriever
knowledge_results = await knowledge_retriever.retrieve(
    query=search_query,
    use_memory=True,   # Tier 1
    use_local=True,    # Tier 2
    use_web=False      # Tier 3 (disabled for now)
)

# Format for prompt injection
knowledge_context = knowledge_retriever.format_context(knowledge_results)
```

## Architecture Flow

```
┌─────────────────────────────────────────────────────────────┐
│ EXPERT EXECUTION                                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐     ┌──────────────────────────────┐  │
│  │ Working Memory  │────▶│ Downstream Slot References   │  │
│  │ (Tier 1 - RAM)  │     │ (Read-only)                  │  │
│  └─────────────────┘     └──────────────────────────────┘  │
│           │                                                 │
│           │ Save completed outputs                          │
│           ▼                                                 │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Knowledge Retriever                                │    │
│  │ ┌─────────────┐  ┌──────────┐  ┌──────────────┐   │    │
│  │ │ Tier 1: WM  │  │Tier 2: KB│  │Tier 3: Web   │   │    │
│  │ │  (slots)    │  │(ground)  │  │(curated)     │   │    │
│  │ └─────────────┘  └──────────┘  └──────────────┘   │    │
│  └────────────────────────────────────────────────────┘    │
│                           │                                 │
│                           ▼                                 │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Injected into prompt as knowledge_context          │    │
│  └────────────────────────────────────────────────────┘    │
│                           │                                 │
│                           ▼                                 │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Expert Generation                                   │    │
│  └────────────────────────────────────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Recommendations

### Keep Current Implementation ✅
- **Tier 1** (`working_memory.py`) - Perfect match for your vision
- **Tier 2 & 3** (`knowledge_layer.py`) - Already implemented, matches vision

### Future: Tier 0 Architecture

When ready to implement model-to-model latent sharing:

```python
# Proposed Tier 0 module: src/latent_exchange.py
class LatentExchange:
    '''Shared latent space for model-to-model communication'''
    
    def publish_latent(self, slot_id: str, kv_cache_segment: Tensor):
        '''Publish a KV-cache segment for other experts to reference'''
        pass
    
    def subscribe_latent(self, slot_id: str, expert_id: str) -> Tensor:
        '''Reference another expert's latent output'''
        pass
    
    def format_latent_for_prompt(self, latent: Tensor) -> str:
        '''Convert latent representation to text for standard models'''
        # Fallback for models that don't support latent input
        pass
```

### Memory Integration Points

1. **Tier 1 ↔ Tier 2**: Save completed slot summaries to LocalKnowledgeBase
2. **Tier 2 persistence**: Keep ground truth in versioned chunks
3. **Tier 3 freshness**: Curated web sources with update timestamps

## Summary

Your 3-tier architecture is **already implemented** in V13:

| Tier | Module | Usage in demo_v13 |
|------|--------|-------------------|
| 1 | `src/working_memory.py` | DAG slot reference |
| 2 | `src/knowledge_layer.py` - LocalKnowledgeBase | Ground truth lookup |
| 3 | `src/knowledge_layer.py` - Web search | Recent/current info |

Tier 0 (latent sharing) is the only piece not yet implemented, and it requires significant research into cross-model KV-cache sharing protocols.
