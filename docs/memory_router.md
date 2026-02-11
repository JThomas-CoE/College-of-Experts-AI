# Memory Vector Router

## Overview

The `MemoryVectorRouter` implements **memory-first routing** - checking WorkingMemory for similar past queries before routing to expert generation. This avoids redundant LLM calls when the same or similar queries are processed repeatedly.

## Why This Matters

In the 3-tier memory architecture:
- **Tier 1 (WorkingMemory)** stores completed slot outputs
- Without memory routing, every query goes to an expert (expensive)
- With memory routing, similar queries return cached results instantly

## How It Works

```
Incoming Query
       │
       ▼
┌──────────────────┐
│  Vector Encode   │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ Search Working   │◄── Compare against all stored slots
│ Memory (Tier 1)  │    using cosine similarity
└────────┬─────────┘
         │
    ┌────┴────┐
    │         │
Similar?    Not Similar?
 (>0.85)    (<0.85)
    │         │
    ▼         ▼
┌────────┐  ┌──────────────┐
│ Return │  │ Route to     │
│ Cached │  │ Expert via   │
│ Result │  │ VectorRouter │
└────────┘  └──────────────┘
```

## Usage

```python
from src.memory_router import MemoryVectorRouter, create_memory_router

# Create router
memory_router = create_memory_router(
    working_memory=working_memory,      # Tier 1 storage
    vector_router=vector_router,         # Fallback expert router
    embedding_manager=embedding_manager, # Shared embeddings
    config={
        "similarity_threshold": 0.85,    # Cache hit threshold
        "min_content_length": 100        # Minimum content to cache
    }
)

# Route a query
result = memory_router.route(
    query="How do I implement OAuth?",
    context="Authentication requirements"
)

# Result structure:
{
    "source": "memory" | "expert",
    "content": "...",           # If memory hit
    "expert_id": "security_architect",  # If expert route
    "confidence": 0.92,
    "match_info": {             # If memory hit
        "slot_id": "auth_slot_1",
        "similarity": 0.92,
        "is_exact": False
    }
}
```

## Integration in demo_v13.py

The memory router is used in the slot execution flow:

```python
# Before executing a slot
route_result = memory_router.route(slot.description, context=slot.title)

if route_result["source"] == "memory":
    # Cache hit - skip expert generation
    memory.store(slot_id, route_result["expert_id"], route_result["content"])
    return (slot_id, True)

# Cache miss - proceed with expert
expert_id = route_result["expert_id"]
# ... generate with expert ...

# After successful generation
memory.store(slot_id, expert_id, resolved_content)
memory_router.index_slot(slot_id, resolved_content)  # Index for future lookups
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `similarity_threshold` | 0.85 | Minimum cosine similarity for cache hit |
| `min_content_length` | 100 | Minimum content length to consider for caching |
| `context` | None | Optional context to include in embedding |

## Statistics

```python
stats = memory_router.get_stats()
# {
#     "total_queries": 42,
#     "memory_hits": 15,
#     "memory_misses": 27,
#     "expert_routes": 27,
#     "memory_hit_rate": 0.357,
#     "cached_embeddings": 12
# }
```

## Comparison to Other Routers

| Router | Purpose | Routes To |
|--------|---------|-----------|
| `SemanticRouter` | Expert selection via capability | Expert |
| `VectorRouter` | Expert selection via dual-embedding | Expert |
| `MemoryVectorRouter` | Avoid redundant generation | WorkingMemory OR Expert |

## Architecture Integration

The memory router sits at the intersection of routing and memory:

```
┌─────────────────────────────────────────────────────────────┐
│                    QUERY FLOW                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   User Query ──┬──► MemoryVectorRouter ──┬──► WorkingMemory │
│                │   (Memory First Check)  │   (Tier 1 - RAM) │
│                │                         │                  │
│                │                         └──► VectorRouter  │
│                │                             (Tier 2/3 KB)  │
│                │                                    │        │
│                │                                    ▼        │
│                │                              Expert Gen     │
│                │                                    │        │
│                │                                    ▼        │
│                └─────────────────────────────► Store Result  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Future Enhancements

1. **Tier 2 Integration**: Search LocalKnowledgeBase before expert routing
2. **Hierarchical Matching**: Match outline → sections → full content
3. **Temporal Decay**: Reduce similarity for older cached results
4. **Partial Matching**: Combine sections from multiple cached results

## Performance Impact

- **Memory Hit**: ~10ms (embedding comparison only)
- **Memory Miss**: ~10s-60s (expert generation)
- **Expected Hit Rate**: 20-40% for repetitive workflows
- **VRAM Savings**: Avoids loading expert models for cached results
