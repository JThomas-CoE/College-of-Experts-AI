"""
Model Backend Protocol - Abstract interface for model serving.

This module defines the contract that all model backends must implement,
enabling hardware-agnostic model serving across different frameworks:
- TransformersBackend: Pure Python with HuggingFace/DirectML
- OllamaBackend: Ollama API for optimized inference
- HybridBackend: Ollama for hot model, Transformers for others
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Protocol, runtime_checkable
from enum import Enum


class BackendType(Enum):
    """Available backend types."""
    TRANSFORMERS = "transformers"
    OLLAMA = "ollama"
    HYBRID = "hybrid"


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    model_id: str
    model_path: str
    backend_type: BackendType
    is_loaded: bool = False
    memory_bytes: Optional[int] = None
    device: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    top_k: int = 40
    stop_sequences: List[str] = field(default_factory=list)
    stream: bool = False


@runtime_checkable
class ModelBackend(Protocol):
    """
    Protocol defining the interface for model backends.
    
    All backends must implement these methods to be compatible
    with the College of Experts system.
    """
    
    def load_model(
        self,
        model_id: str,
        model_path: str,
        device: Optional[str] = None,
        **kwargs
    ) -> ModelInfo:
        """
        Load a model into memory.
        
        Args:
            model_id: Unique identifier for this model instance
            model_path: Path to model (HF repo, local path, or Ollama model name)
            device: Target device (cuda, cpu, directml, auto)
            **kwargs: Backend-specific options
            
        Returns:
            ModelInfo with details about the loaded model
        """
        ...
    
    def unload_model(self, model_id: str) -> bool:
        """
        Unload a model from memory.
        
        Args:
            model_id: ID of the model to unload
            
        Returns:
            True if model was unloaded, False if not found
        """
        ...
    
    def is_loaded(self, model_id: str) -> bool:
        """Check if a model is currently loaded."""
        ...
    
    def list_loaded(self) -> List[str]:
        """List all currently loaded model IDs."""
        ...
    
    def generate(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a response from a loaded model.
        
        Args:
            model_id: ID of the model to use
            messages: Chat messages in OpenAI format [{"role": "user", "content": "..."}]
            config: Generation configuration
            system_prompt: Optional system prompt to prepend
            
        Returns:
            Generated text response
        """
        ...
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get information about a loaded model."""
        ...


class BaseBackend(ABC):
    """
    Abstract base class for model backends.
    
    Provides common functionality and enforces the ModelBackend protocol.
    """
    
    def __init__(self, device: str = "auto"):
        """
        Initialize the backend.
        
        Args:
            device: Default device for model loading (cuda, cpu, directml, auto)
        """
        self.default_device = device
        self._models: Dict[str, ModelInfo] = {}
    
    @property
    @abstractmethod
    def backend_type(self) -> BackendType:
        """Return the backend type."""
        ...
    
    @abstractmethod
    def load_model(
        self,
        model_id: str,
        model_path: str,
        device: Optional[str] = None,
        **kwargs
    ) -> ModelInfo:
        """Load a model - implementation required."""
        ...
    
    @abstractmethod
    def unload_model(self, model_id: str) -> bool:
        """Unload a model - implementation required."""
        ...
    
    @abstractmethod
    def generate(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate text - implementation required."""
        ...
    
    def is_loaded(self, model_id: str) -> bool:
        """Check if model is loaded."""
        return model_id in self._models and self._models[model_id].is_loaded
    
    def list_loaded(self) -> List[str]:
        """List loaded model IDs."""
        return [mid for mid, info in self._models.items() if info.is_loaded]
    
    def get_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """Get model info."""
        return self._models.get(model_id)
    
    def _resolve_device(self, device: Optional[str]) -> str:
        """Resolve device string to actual device."""
        if device is None:
            device = self.default_device
        
        if device == "auto":
            return self._detect_best_device()
        return device
    
    def _detect_best_device(self) -> str:
        """Detect the best available device."""
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            # Check for DirectML (Windows AMD)
            try:
                import torch_directml
                return "privateuseone"  # DirectML device string
            except ImportError:
                pass
        except ImportError:
            pass
        return "cpu"


def create_backend(
    backend_type: BackendType = BackendType.TRANSFORMERS,
    device: str = "auto",
    **kwargs
) -> BaseBackend:
    """
    Factory function to create a backend instance.
    
    Args:
        backend_type: Type of backend to create
        device: Default device for models
        **kwargs: Backend-specific configuration
        
    Returns:
        Configured backend instance
    """
    if backend_type == BackendType.TRANSFORMERS:
        from .transformers_backend import TransformersBackend
        return TransformersBackend(device=device, **kwargs)
    
    elif backend_type == BackendType.OLLAMA:
        from .ollama_backend import OllamaBackend
        return OllamaBackend(**kwargs)
    
    elif backend_type == BackendType.HYBRID:
        from .hybrid_backend import HybridBackend
        return HybridBackend(device=device, **kwargs)
    
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
