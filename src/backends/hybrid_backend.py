"""
Hybrid Backend - Combines Ollama (hot) + Transformers (warm).

This backend uses Ollama for the active "hot" model for fastest generation,
while keeping additional models loaded via Transformers for quick switching.

Features:
- Best of both worlds: Ollama speed + multi-model flexibility
- Hot model through Ollama (optimized, quantized)
- Warm models through Transformers (ready, slower)
- Automatic promotion/demotion between tiers
"""

from typing import Optional, Dict, List, Any

from .base import BaseBackend, BackendType, ModelInfo, GenerationConfig
from .transformers_backend import TransformersBackend
from .ollama_backend import OllamaBackend


class HybridBackend(BaseBackend):
    """
    Hybrid model backend combining Ollama and Transformers.
    
    Strategy:
    - One "hot" model runs through Ollama for fastest generation
    - Other models load through Transformers, ready for quick activation
    - When switching hot models, demote old hot to warm (Transformers)
    """
    
    def __init__(
        self,
        device: str = "auto",
        ollama_host: str = "http://localhost:11434",
        max_warm_models: int = 5,
        **transformers_kwargs
    ):
        """
        Initialize hybrid backend.
        
        Args:
            device: Device for Transformers backend
            ollama_host: Ollama server URL
            max_warm_models: Maximum models in Transformers tier
            **transformers_kwargs: Additional args for TransformersBackend
        """
        super().__init__(device=device)
        
        # Initialize both backends
        self._ollama = OllamaBackend(host=ollama_host)
        self._transformers = TransformersBackend(device=device, **transformers_kwargs)
        
        self.max_warm_models = max_warm_models
        
        # Track which model is hot (in Ollama)
        self._hot_model_id: Optional[str] = None
        
        # Map model_id to which backend holds it
        self._model_backends: Dict[str, str] = {}  # model_id -> "ollama" | "transformers"
        
        # Store original model paths for backend switching
        self._model_paths: Dict[str, str] = {}
        
        # Ollama model name mapping (model_id -> ollama_name)
        self._ollama_names: Dict[str, str] = {}
    
    @property
    def backend_type(self) -> BackendType:
        return BackendType.HYBRID
    
    def is_ollama_available(self) -> bool:
        """Check if Ollama is running."""
        return self._ollama.is_ollama_running()
    
    def load_model(
        self,
        model_id: str,
        model_path: str,
        device: Optional[str] = None,
        as_hot: bool = False,
        ollama_name: Optional[str] = None,
        **kwargs
    ) -> ModelInfo:
        """
        Load a model into the hybrid system.
        
        Args:
            model_id: Unique identifier
            model_path: HuggingFace path or local path
            device: Device for Transformers (ignored for Ollama)
            as_hot: Load as hot model (via Ollama)
            ollama_name: Ollama model name (if different from model_path)
            **kwargs: Additional loading options
        """
        self._model_paths[model_id] = model_path
        
        if as_hot and self.is_ollama_available():
            return self._load_as_hot(model_id, model_path, ollama_name)
        else:
            return self._load_as_warm(model_id, model_path, device, **kwargs)
    
    def _load_as_hot(
        self,
        model_id: str,
        model_path: str,
        ollama_name: Optional[str] = None
    ) -> ModelInfo:
        """Load model as hot (Ollama)."""
        # If there's already a hot model, demote it
        if self._hot_model_id and self._hot_model_id != model_id:
            self._demote_hot_to_warm()
        
        # Use ollama_name if provided, else model_path
        ollama_model = ollama_name or model_path
        self._ollama_names[model_id] = ollama_model
        
        info = self._ollama.load_model(model_id, ollama_model)
        
        self._hot_model_id = model_id
        self._model_backends[model_id] = "ollama"
        self._models[model_id] = info
        
        return info
    
    def _load_as_warm(
        self,
        model_id: str,
        model_path: str,
        device: Optional[str] = None,
        **kwargs
    ) -> ModelInfo:
        """Load model as warm (Transformers)."""
        # Evict oldest warm if at capacity
        warm_count = sum(1 for b in self._model_backends.values() if b == "transformers")
        if warm_count >= self.max_warm_models:
            self._evict_oldest_warm()
        
        info = self._transformers.load_model(model_id, model_path, device, **kwargs)
        
        self._model_backends[model_id] = "transformers"
        self._models[model_id] = info
        
        return info
    
    def _demote_hot_to_warm(self):
        """Demote current hot model to warm tier."""
        if not self._hot_model_id:
            return
        
        model_id = self._hot_model_id
        model_path = self._model_paths.get(model_id)
        
        if model_path:
            print(f"Demoting {model_id} from hot to warm...")
            # Unload from Ollama (just removes our reference)
            self._ollama.unload_model(model_id)
            # Load into Transformers
            self._transformers.load_model(model_id, model_path)
            self._model_backends[model_id] = "transformers"
        
        self._hot_model_id = None
    
    def _evict_oldest_warm(self):
        """Evict oldest warm model to make room."""
        # Find oldest transformers model
        for model_id, backend in list(self._model_backends.items()):
            if backend == "transformers":
                self._transformers.unload_model(model_id)
                del self._model_backends[model_id]
                print(f"Evicted warm model: {model_id}")
                break
    
    def promote_to_hot(self, model_id: str) -> bool:
        """
        Promote a warm model to hot tier.
        
        Args:
            model_id: ID of model to promote
            
        Returns:
            True if successful
        """
        if model_id not in self._model_backends:
            return False
        
        if self._model_backends[model_id] == "ollama":
            return True  # Already hot
        
        if not self.is_ollama_available():
            print("Ollama not available, cannot promote")
            return False
        
        # Get Ollama name
        ollama_name = self._ollama_names.get(model_id)
        if not ollama_name:
            print(f"No Ollama model name for {model_id}, cannot promote")
            return False
        
        # Demote current hot
        self._demote_hot_to_warm()
        
        # Unload from transformers
        self._transformers.unload_model(model_id)
        
        # Load as hot
        self._load_as_hot(model_id, self._model_paths[model_id], ollama_name)
        
        return True
    
    def unload_model(self, model_id: str) -> bool:
        """Unload model from whichever backend has it."""
        if model_id not in self._model_backends:
            return False
        
        backend = self._model_backends[model_id]
        
        if backend == "ollama":
            self._ollama.unload_model(model_id)
            self._hot_model_id = None
        else:
            self._transformers.unload_model(model_id)
        
        del self._model_backends[model_id]
        if model_id in self._models:
            self._models[model_id].is_loaded = False
        
        return True
    
    def generate(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate using appropriate backend."""
        if model_id not in self._model_backends:
            raise ValueError(f"Model {model_id} not loaded")
        
        backend = self._model_backends[model_id]
        
        if backend == "ollama":
            return self._ollama.generate(model_id, messages, config, system_prompt)
        else:
            return self._transformers.generate(model_id, messages, config, system_prompt)
    
    def get_hot_model(self) -> Optional[str]:
        """Get ID of current hot model."""
        return self._hot_model_id
    
    def get_warm_models(self) -> List[str]:
        """Get IDs of warm models."""
        return [
            mid for mid, backend in self._model_backends.items()
            if backend == "transformers"
        ]
    
    def get_status(self) -> Dict[str, Any]:
        """Get hybrid backend status."""
        return {
            "hot_model": self._hot_model_id,
            "warm_models": self.get_warm_models(),
            "ollama_available": self.is_ollama_available(),
            "ollama_models": self._ollama.list_ollama_models() if self.is_ollama_available() else [],
            "transformers_memory": self._transformers.get_memory_usage()
        }


if __name__ == "__main__":
    # Quick test
    backend = HybridBackend()
    print(f"Ollama available: {backend.is_ollama_available()}")
    print("HybridBackend initialized successfully")
