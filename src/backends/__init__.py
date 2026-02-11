"""
Backends package for College of Experts.

Provides pluggable model serving backends:
- TransformersBackend: Pure Python with HuggingFace/DirectML
- OllamaBackend: Ollama API for optimized inference  
- HybridBackend: Ollama for hot model, Transformers for others
"""

from .base import (
    ModelBackend,
    BaseBackend,
    BackendType,
    ModelInfo,
    GenerationConfig,
    create_backend
)

from .transformers_backend import TransformersBackend

__all__ = [
    "ModelBackend",
    "BaseBackend", 
    "BackendType",
    "ModelInfo",
    "GenerationConfig",
    "create_backend",
    "TransformersBackend",
]

# Lazy imports for optional backends
def get_transformers_backend():
    return TransformersBackend

def get_ollama_backend():
    from .ollama_backend import OllamaBackend
    return OllamaBackend

def get_hybrid_backend():
    from .hybrid_backend import HybridBackend
    return HybridBackend

