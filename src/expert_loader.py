"""
Expert Loader - Handles dynamic loading and unloading of expert models.

This module manages the expert lifecycle with pluggable backends:
1. Loading experts via configurable backend (Transformers, Ollama, Hybrid)
2. Caching loaded experts in memory tiers
3. Evicting unused experts
4. Expert health checking
"""

import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from collections import OrderedDict
from pathlib import Path
import json

from .backends.base import (
    BackendType,
    GenerationConfig,
    create_backend,
    BaseBackend
)


@dataclass
class Expert:
    """Represents a loaded expert model."""
    expert_id: str
    model_name: str
    display_name: str
    domains: List[str]
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    backend_type: Optional[BackendType] = None
    
    def touch(self):
        """Mark the expert as recently used."""
        self.last_used = time.time()
        self.use_count += 1


@dataclass 
class ExpertLoaderConfig:
    """Configuration for the expert loader."""
    max_hot_experts: int = 3  # Max experts in "hot" tier (GPU/active)
    max_warm_experts: int = 8  # Max experts in "warm" tier (RAM)
    manifest_path: Optional[Path] = None
    preload_experts: List[str] = field(default_factory=list)
    
    # Backend configuration
    backend_type: BackendType = BackendType.TRANSFORMERS
    device: str = "auto"
    
    # Ollama-specific
    ollama_host: str = "http://localhost:11434"
    
    # Transformers-specific
    torch_dtype: str = "float16"
    use_flash_attention: bool = False


class ExpertLoader:
    """
    Manages dynamic loading and caching of expert models.
    
    Uses a pluggable backend architecture supporting:
    - TransformersBackend: Pure Python/DirectML for any hardware
    - OllamaBackend: Optimized quantized inference
    - HybridBackend: Ollama for hot + Transformers for warm
    
    Uses a tiered caching strategy:
    - Hot: Currently active experts (limited by GPU memory)
    - Warm: Recently used experts (kept available)
    - Cold: All other experts (on disk, need to be loaded)
    """
    
    def __init__(self, config: Optional[ExpertLoaderConfig] = None):
        self.config = config or ExpertLoaderConfig()
        self.manifest = self._load_manifest()
        
        # Hot cache: OrderedDict for LRU eviction
        self._hot_cache: OrderedDict[str, Expert] = OrderedDict()
        self._warm_cache: OrderedDict[str, Expert] = OrderedDict()
        
        # Initialize backend
        self._backend = self._create_backend()
        
        # Preload specified experts
        for expert_id in self.config.preload_experts:
            self.load(expert_id)
    
    def _create_backend(self) -> BaseBackend:
        """Create the model backend based on configuration."""
        if self.config.backend_type == BackendType.OLLAMA:
            return create_backend(
                BackendType.OLLAMA,
                host=self.config.ollama_host
            )
        elif self.config.backend_type == BackendType.HYBRID:
            return create_backend(
                BackendType.HYBRID,
                device=self.config.device,
                ollama_host=self.config.ollama_host,
                torch_dtype=self.config.torch_dtype
            )
        else:  # Default to TRANSFORMERS
            return create_backend(
                BackendType.TRANSFORMERS,
                device=self.config.device,
                torch_dtype=self.config.torch_dtype,
                use_flash_attention=self.config.use_flash_attention
            )
    
    def _load_manifest(self) -> dict:
        """Load expert manifest from file or use defaults."""
        if self.config.manifest_path and self.config.manifest_path.exists():
            with open(self.config.manifest_path) as f:
                return json.load(f)
        
        # Default manifest - uses Nemotron Nano 30B for all experts via Ollama
        # Each expert has domain-specific system prompts that shape the model's behavior
        # When using Ollama backend, ollama_name is used; for Transformers, model is used
        default_model = "nemotron-3-nano:30b"  # Primary model for Ollama
        
        return {
            "code_python": {
                "display_name": "Python Expert",
                "model": "LiquidAI/LFM2.5-1.2B-Instruct",
                "ollama_name": default_model,
                "domains": ["python", "programming", "fastapi", "django", "async"],
                "system_prompt": """You are an expert Python developer with deep expertise in:
- Python 3.10+ features: pattern matching, type hints, dataclasses, async/await
- Web frameworks: FastAPI, Django, Flask
- Testing: pytest, hypothesis
- Package management: pip, poetry, uv

Write production-quality code with type hints. Explain non-obvious decisions. Consider edge cases."""
            },
            "code_general": {
                "display_name": "General Coding Expert",
                "model": "LiquidAI/LFM2.5-1.2B-Instruct",
                "ollama_name": default_model,
                "domains": ["programming", "code", "software", "javascript", "rust", "go"],
                "system_prompt": """You are an expert software developer proficient in multiple languages:
- JavaScript/TypeScript, React, Node.js
- Rust, Go, C++
- Design patterns, SOLID principles
- Code review and refactoring

Provide clean, well-structured code with explanations of design choices."""
            },
            "sql_expert": {
                "display_name": "Database & SQL Expert",
                "model": "LiquidAI/LFM2.5-1.2B-Instruct",
                "ollama_name": default_model,
                "domains": ["sql", "database", "postgresql", "mysql", "optimization"],
                "system_prompt": """You are a database expert specializing in:
- SQL dialects: PostgreSQL, MySQL, SQLite
- Query optimization and EXPLAIN analysis
- Schema design and normalization
- Transactions, isolation levels, indexing

Provide optimized queries and explain performance implications."""
            },
            "security_expert": {
                "display_name": "Security Expert",
                "model": "LiquidAI/LFM2.5-1.2B-Instruct",
                "ollama_name": default_model,
                "domains": ["security", "authentication", "authorization", "owasp", "vulnerabilities"],
                "system_prompt": """You are an application security expert focused on:
- OWASP Top 10 vulnerabilities
- Authentication: OAuth2, JWT, session management
- Input validation and output encoding
- Secure coding practices

Flag security issues with severity ratings. Be thorough and cautious."""
            },
            "architecture_expert": {
                "display_name": "Architecture Expert",
                "model": "LiquidAI/LFM2.5-1.2B-Instruct",
                "ollama_name": default_model,
                "domains": ["architecture", "design", "microservices", "scaling", "api"],
                "system_prompt": """You are a software architecture expert specializing in:
- Architectural patterns: microservices, monolith, event-driven
- API design: REST, GraphQL, gRPC
- Scalability and distributed systems
- Trade-off analysis

Present multiple options with pros/cons. There are no perfect architectures."""
            },
            "general": {
                "display_name": "General Assistant",
                "model": "LiquidAI/LFM2.5-1.2B-Instruct",
                "ollama_name": default_model,
                "domains": ["general", "research", "analysis", "writing"],
                "system_prompt": "You are a knowledgeable general assistant. Provide helpful, accurate information."
            }
        }
    
    def get_expert_info(self, expert_id: str) -> Optional[dict]:
        """Get metadata about an expert without loading it."""
        return self.manifest.get(expert_id)
    
    def list_available(self) -> List[str]:
        """List all available expert IDs."""
        return list(self.manifest.keys())
    
    def list_loaded(self) -> List[str]:
        """List currently loaded (hot) expert IDs."""
        return list(self._hot_cache.keys())
    
    def is_loaded(self, expert_id: str) -> bool:
        """Check if an expert is currently loaded (hot or warm)."""
        return expert_id in self._hot_cache or expert_id in self._warm_cache
    
    def load(self, expert_id: str, as_hot: bool = True) -> Expert:
        """
        Load an expert model, moving it to the hot cache.
        
        Args:
            expert_id: ID of the expert to load
            as_hot: If True, load as hot (for hybrid backend)
            
        Returns:
            The loaded Expert object
            
        Raises:
            ValueError: If expert_id not in manifest
        """
        # Check if already hot
        if expert_id in self._hot_cache:
            expert = self._hot_cache[expert_id]
            expert.touch()
            # Move to end (most recently used)
            self._hot_cache.move_to_end(expert_id)
            return expert
        
        # Check warm cache
        if expert_id in self._warm_cache:
            expert = self._warm_cache.pop(expert_id)
            expert.touch()
            self._promote_to_hot(expert)
            return expert
        
        # Need to load from cold storage
        if expert_id not in self.manifest:
            raise ValueError(f"Unknown expert: {expert_id}")
        
        info = self.manifest[expert_id]
        
        # Get model path based on backend type
        if self.config.backend_type == BackendType.OLLAMA:
            model_path = info.get("ollama_name", info["model"])
        else:
            model_path = info["model"]
        
        # Load via backend
        try:
            if self.config.backend_type == BackendType.HYBRID:
                # For hybrid, set as_hot flag
                self._backend.load_model(
                    expert_id,
                    model_path,
                    as_hot=as_hot,
                    ollama_name=info.get("ollama_name")
                )
            else:
                self._backend.load_model(expert_id, model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load expert {expert_id}: {e}")
        
        # Create expert object
        expert = Expert(
            expert_id=expert_id,
            model_name=model_path,
            display_name=info.get("display_name", expert_id),
            domains=info.get("domains", []),
            backend_type=self.config.backend_type
        )
        
        self._promote_to_hot(expert)
        return expert
    
    def _promote_to_hot(self, expert: Expert):
        """Move an expert to the hot cache, evicting if necessary."""
        # Evict from hot if at capacity
        while len(self._hot_cache) >= self.config.max_hot_experts:
            # Evict least recently used (first item)
            evicted_id, evicted = self._hot_cache.popitem(last=False)
            self._demote_to_warm(evicted)
        
        self._hot_cache[expert.expert_id] = expert
    
    def _demote_to_warm(self, expert: Expert):
        """Move an expert from hot to warm cache."""
        # Evict from warm if at capacity
        while len(self._warm_cache) >= self.config.max_warm_experts:
            evicted_id, evicted = self._warm_cache.popitem(last=False)
            # Fully unload
            self._backend.unload_model(evicted_id)
        
        self._warm_cache[expert.expert_id] = expert
    
    def unload(self, expert_id: str):
        """Explicitly unload an expert from all caches."""
        self._hot_cache.pop(expert_id, None)
        self._warm_cache.pop(expert_id, None)
        self._backend.unload_model(expert_id)
    
    def query(
        self,
        expert_id: str,
        messages: List[dict],
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """
        Query a loaded expert.
        
        Args:
            expert_id: ID of the expert to query
            messages: Chat messages in OpenAI format
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            The expert's response text
        """
        expert = self.load(expert_id)  # Ensures loaded
        
        # Get system prompt from manifest
        info = self.manifest.get(expert_id, {})
        system_prompt = info.get("system_prompt", "You are a helpful assistant.")
        
        # Create generation config
        config = GenerationConfig(
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        response = self._backend.generate(
            expert_id,
            messages,
            config=config,
            system_prompt=system_prompt
        )
        
        expert.touch()
        return response
    
    def get_cache_stats(self) -> dict:
        """Get statistics about cache state."""
        return {
            "hot_count": len(self._hot_cache),
            "warm_count": len(self._warm_cache),
            "hot_experts": list(self._hot_cache.keys()),
            "warm_experts": list(self._warm_cache.keys()),
            "hot_capacity": self.config.max_hot_experts,
            "warm_capacity": self.config.max_warm_experts,
            "backend_type": self.config.backend_type.value
        }
    
    @property
    def backend(self) -> BaseBackend:
        """Access the underlying backend."""
        return self._backend


if __name__ == "__main__":
    # Quick test with Transformers backend
    print("Testing ExpertLoader with Transformers backend...")
    
    config = ExpertLoaderConfig(
        backend_type=BackendType.TRANSFORMERS,
        device="auto"
    )
    
    loader = ExpertLoader(config)
    print(f"Available experts: {loader.list_available()}")
    print(f"Backend: {loader.config.backend_type.value}")
    print(f"Cache stats: {loader.get_cache_stats()}")
