# Deprecated Router Implementations

This directory contains deprecated router implementations that are no longer used in the current demo (demo_v13.py).

## Deprecated Files

### `router.py`
- **Status:** DEPRECATED as of 2026-02-01
- **Reason:** Replaced by `memory_router.py` + `vector_router.py` architecture
- **Last Used In:** demo.py, test_router.py
- **Migration Path:** Use `MemoryVectorRouter` from `src.memory_router`

### `router_v8.py`
- **Status:** DEPRECATED as of 2026-02-01
- **Reason:** Superseded by v9 and then by modular routing
- **Last Used In:** None (orphaned)
- **Migration Path:** Delete or archive

### `router_v9.py`
- **Status:** DEPRECATED as of 2026-02-01
- **Reason:** Superseded by modular routing architecture
- **Last Used In:** None (orphaned)
- **Migration Path:** Delete or archive

### `semantic_router.py`
- **Status:** DEPRECATED as of 2026-02-01
- **Reason:** Functionality merged into `vector_router.py`
- **Last Used In:** demo_v12_e2e.py
- **Migration Path:** Use `VectorRouter` from `src.vector_router`

## Current Routing Architecture (demo_v13.py)

The current system uses a **layered routing approach**:

```
Query
  ↓
MemoryVectorRouter (checks WorkingMemory first)
  ↓ (cache miss)
VectorRouter (FAISS-based dual-embedding routing)
  ↓
Expert Assignment
```

### Active Router Files:
- ✅ `memory_router.py` - Memory-first routing with cache
- ✅ `vector_router.py` - FAISS-based similarity search

### Key Classes:
- `MemoryVectorRouter` - Primary router with memory awareness
- `VectorRouter` - Underlying vector similarity engine
- `DualEmbeddingRouter` (in demo_v13.py) - Capability/exclusion scoring

## Cleanup Plan

1. **Phase 1 (DONE):** Move deprecated files to `src/deprecated/`
2. **Phase 2:** Add deprecation warnings to files
3. **Phase 3:** Update tests to use new routers
4. **Phase 4:** Archive or delete after 1 release cycle

## Questions?

Contact: See ENGINEERING_LOG.md for architecture decisions
