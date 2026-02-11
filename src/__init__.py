"""
College of Experts - A disk-resident sparse mixture architecture

This package provides the core components for running a College of Experts
system locally, including:
- Router: Scope negotiation and expert dispatch
- ExpertLoader: Dynamic loading and caching of expert models
- Backends: Pluggable model serving (Transformers, Ollama, Hybrid)
- MemoryBackbone: Shared memory for inter-expert communication (SQLite or Memvid)
- Harness: Agentic orchestration layer
- SharedLatentSpace: Experimental shared KV cache for multi-expert coordination

V8 Components:
- ExpertCatalog: Expert definitions and capability matching
- ExpertSlotManager: Dynamic expert loading with VRAM management
- CrosstalkBus: Expert-to-expert communication (A2A)
- EpisodicMemory: Committed vs WIP work tracking
"""

__version__ = "0.8.0"

from .router import Router, RouterConfig, ExpertRecommendation
from .expert_loader import ExpertLoader, ExpertLoaderConfig, Expert
from .memory_backbone import MemoryBackbone, MemoryConfig, MemoryEntry
from .harness import Harness, HarnessConfig, Session

# Backend system
from .backends import (
    BackendType,
    ModelInfo,
    GenerationConfig,
    create_backend
)

# Shared latent space (experimental)
from .shared_latent import (
    SharedLatentSpace,
    SharedLatentConfig,
    SharedContextMode,
    SharedLatentExperiment
)

# V8 Components
from .expert_catalog import ExpertCatalog, ExpertDefinition, load_catalog
from .expert_slots_v8 import ExpertSlotManager, ExpertInstance
from .crosstalk_v8 import CrosstalkBus, CrosstalkMessage, MessageType, get_crosstalk_bus
from .episodic_memory import EpisodicMemory, MemoryItem, MemoryStatus, create_memory
from .router_v8 import CouncilRouter, TaskChunk, CompletionStatus, CheckpointResult
from .council_v8 import CouncilMode, CouncilResult, CouncilResponse

# Memvid memory backend (optional, preferred)
try:
    from .memvid_memory import (
        MemvidMemoryBackbone,
        MemvidConfig,
        create_memory_backbone
    )
    MEMVID_AVAILABLE = True
except ImportError:
    MEMVID_AVAILABLE = False
    MemvidMemoryBackbone = None
    MemvidConfig = None
    create_memory_backbone = None

__all__ = [
    # Core classes
    "Router",
    "ExpertLoader", 
    "MemoryBackbone",
    "MemvidMemoryBackbone",
    "Harness",
    # Backend system
    "BackendType",
    "ModelInfo",
    "GenerationConfig",
    "create_backend",
    # Shared latent space (experimental)
    "SharedLatentSpace",
    "SharedLatentConfig",
    "SharedContextMode",
    "SharedLatentExperiment",
    # V7 Components
    "ExpertCatalog",
    "ExpertDefinition",
    "load_catalog",
    "ExpertSlotManager",
    "ExpertInstance",
    "CrosstalkBus",
    "CrosstalkMessage",
    "MessageType",
    "get_crosstalk_bus",
    "EpisodicMemory",
    "MemoryItem",
    "MemoryStatus",
    "create_memory",
    "CouncilRouter",
    "TaskChunk",
    "CompletionStatus",
    "CheckpointResult",
    "CouncilMode",
    "CouncilResult",
    "CouncilResponse",
    # Config classes
    "RouterConfig",
    "ExpertLoaderConfig",
    "MemoryConfig",
    "MemvidConfig",
    "HarnessConfig",
    # Data classes
    "ExpertRecommendation",
    "Expert",
    "MemoryEntry",
    "Session",
    # Factory functions
    "create_memory_backbone",
    # Feature flags
    "MEMVID_AVAILABLE",
]
