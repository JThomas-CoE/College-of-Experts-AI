"""
Shared Latent Space - Experimental KV Cache Sharing for Multi-Expert Coordination.

This module provides an EXPERIMENTAL implementation of shared latent space
between multiple expert instances using KV cache prefixing.

Philosophy:
    This is a "such as" implementation - useful enough to learn from,
    explicitly NOT optimal or final. The goal is validated learning.

Features:
    - Shared context computed once, reused by all experts
    - Experts "see" shared context in attention without context window cost
    - Enable/disable for A/B testing
    - Workspace state tracking

Usage:
    from src.shared_latent import SharedLatentSpace, SharedLatentConfig
    
    config = SharedLatentConfig(enabled=True)
    shared = SharedLatentSpace(model, tokenizer, config)
    shared.set_shared_context("Current task: Build auth system...")
    
    response = shared.generate_with_shared_context(
        expert_id="python_expert",
        expert_prompt=PYTHON_PROMPT,
        user_message="How should I handle JWT refresh?"
    )
"""

import torch
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SharedContextMode(Enum):
    """How shared context is provided to experts."""
    DISABLED = "disabled"           # No sharing
    KV_CACHE_PREFIX = "kv_cache"    # Shared KV cache (latent)
    TEXT_PREAMBLE = "text"          # Text injected into prompt (baseline)


@dataclass
class SharedLatentConfig:
    """Configuration for shared latent space."""
    
    # Master enable/disable
    enabled: bool = True
    
    # Mode of sharing
    mode: SharedContextMode = SharedContextMode.KV_CACHE_PREFIX
    
    # Maximum shared context length (tokens)
    max_shared_tokens: int = 1024
    
    # Auto-update: recompute shared cache after each expert turn
    auto_update: bool = False
    
    # Fallback to text mode if KV cache fails
    fallback_to_text: bool = True
    
    # Debug: log shared context operations
    debug: bool = False


@dataclass
class SharedContextCache:
    """
    Stores a precomputed KV cache for shared context.
    All experts can extend from this common base.
    """
    past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]]
    sequence_length: int
    context_text: str
    token_count: int
    device: str


@dataclass
class ExpertState:
    """Tracks state of an individual expert."""
    expert_id: str
    status: str = "idle"  # idle, working, waiting, done
    current_task: Optional[str] = None
    confidence: float = 1.0
    last_output_summary: Optional[str] = None


class SharedLatentSpace:
    """
    Experimental shared latent space using KV cache prefixing.
    
    Provides:
    1. Shared context that all experts can attend to
    2. Expert state tracking for coordination
    3. Enable/disable for A/B testing
    
    This is explicitly a "useful approximation" for learning,
    not an optimal final implementation.
    """
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Optional[SharedLatentConfig] = None
    ):
        """
        Initialize the shared latent space.
        
        Args:
            model: The loaded transformer model (same for all experts)
            tokenizer: The tokenizer
            config: Configuration options
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or SharedLatentConfig()
        
        # Shared context cache
        self.shared_cache: Optional[SharedContextCache] = None
        
        # Expert state tracking
        self.expert_states: Dict[str, ExpertState] = {}
        
        # Metrics for analysis
        self.metrics = {
            "cache_computations": 0,
            "generations_with_shared": 0,
            "generations_without_shared": 0,
            "fallbacks_to_text": 0
        }
        
        if self.config.debug:
            logger.setLevel(logging.DEBUG)
    
    @property
    def is_enabled(self) -> bool:
        """Check if shared latent space is enabled."""
        return self.config.enabled
    
    def enable(self):
        """Enable shared latent space."""
        self.config.enabled = True
        logger.info("SharedLatentSpace enabled")
    
    def disable(self):
        """Disable shared latent space."""
        self.config.enabled = False
        logger.info("SharedLatentSpace disabled")
    
    def set_mode(self, mode: SharedContextMode):
        """Change the sharing mode."""
        self.config.mode = mode
        logger.info(f"SharedLatentSpace mode set to: {mode.value}")
    
    # =========================================================================
    # CONTEXT MANAGEMENT
    # =========================================================================
    
    def set_shared_context(self, context: str) -> bool:
        """
        Compute and store KV cache for shared context.
        
        Args:
            context: The shared context text
            
        Returns:
            True if cache was computed successfully
        """
        if not self.config.enabled:
            logger.debug("SharedLatentSpace disabled, skipping cache computation")
            return False
        
        if self.config.mode == SharedContextMode.DISABLED:
            return False
        
        if self.config.mode == SharedContextMode.TEXT_PREAMBLE:
            # Just store the text, no cache computation
            self.shared_cache = SharedContextCache(
                past_key_values=None,
                sequence_length=0,
                context_text=context,
                token_count=len(self.tokenizer.encode(context)),
                device="cpu"
            )
            return True
        
        # KV Cache mode
        try:
            # Tokenize
            inputs = self.tokenizer(
                context,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.max_shared_tokens
            )
            
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Compute KV cache
            with torch.no_grad():
                outputs = self.model(
                    **inputs,
                    use_cache=True,
                    return_dict=True
                )
            
            self.shared_cache = SharedContextCache(
                past_key_values=outputs.past_key_values,
                sequence_length=inputs["input_ids"].shape[1],
                context_text=context,
                token_count=inputs["input_ids"].shape[1],
                device=str(device)
            )
            
            self.metrics["cache_computations"] += 1
            logger.debug(f"Computed shared cache: {self.shared_cache.token_count} tokens")
            
            return True
            
        except Exception as e:
            logger.warning(f"Failed to compute KV cache: {e}")
            if self.config.fallback_to_text:
                logger.info("Falling back to text preamble mode")
                self.config.mode = SharedContextMode.TEXT_PREAMBLE
                self.metrics["fallbacks_to_text"] += 1
                return self.set_shared_context(context)
            return False
    
    def update_shared_context(self, additional_context: str):
        """
        Append to shared context and recompute cache.
        
        Args:
            additional_context: Text to append
        """
        if self.shared_cache is None:
            self.set_shared_context(additional_context)
        else:
            new_context = self.shared_cache.context_text + "\n" + additional_context
            self.set_shared_context(new_context)
    
    def get_workspace_summary(self) -> str:
        """
        Generate a text summary of current workspace state.
        Used for shared context updates.
        """
        lines = ["[WORKSPACE STATE]"]
        for expert_id, state in self.expert_states.items():
            status_line = f"- {expert_id}: {state.status}"
            if state.current_task:
                status_line += f" ({state.current_task})"
            if state.confidence < 0.7:
                status_line += f" [confidence: {state.confidence:.1f}]"
            lines.append(status_line)
        return "\n".join(lines)
    
    # =========================================================================
    # EXPERT STATE TRACKING
    # =========================================================================
    
    def register_expert(self, expert_id: str):
        """Register an expert in the workspace."""
        if expert_id not in self.expert_states:
            self.expert_states[expert_id] = ExpertState(expert_id=expert_id)
    
    def update_expert_state(
        self,
        expert_id: str,
        status: Optional[str] = None,
        current_task: Optional[str] = None,
        confidence: Optional[float] = None,
        last_output_summary: Optional[str] = None
    ):
        """Update an expert's state."""
        if expert_id not in self.expert_states:
            self.register_expert(expert_id)
        
        state = self.expert_states[expert_id]
        if status is not None:
            state.status = status
        if current_task is not None:
            state.current_task = current_task
        if confidence is not None:
            state.confidence = confidence
        if last_output_summary is not None:
            state.last_output_summary = last_output_summary
        
        # Auto-update shared context if enabled
        if self.config.auto_update and self.shared_cache is not None:
            self.update_shared_context(self.get_workspace_summary())
    
    # =========================================================================
    # GENERATION
    # =========================================================================
    
    def get_shared_prefix(self) -> Optional[str]:
        """
        Get text prefix for text mode.
        Returns None if in KV cache mode or disabled.
        """
        if not self.config.enabled or self.shared_cache is None:
            return None
        
        if self.config.mode == SharedContextMode.TEXT_PREAMBLE:
            return self.shared_cache.context_text
        
        return None
    
    def get_past_key_values(self) -> Optional[Tuple]:
        """
        Get cached KV values for KV cache mode.
        Returns None if in text mode or disabled.
        """
        if not self.config.enabled or self.shared_cache is None:
            return None
        
        if self.config.mode == SharedContextMode.KV_CACHE_PREFIX:
            return self.shared_cache.past_key_values
        
        return None
    
    def get_cloned_past_key_values(self) -> Optional[Tuple]:
        """
        Get a DEEP COPY of cached KV values.
        Crucial for parallel experts to avoid race conditions.
        """
        if not self.config.enabled or self.shared_cache is None:
            return None
            
        if self.config.mode == SharedContextMode.KV_CACHE_PREFIX:
            pkv = self.shared_cache.past_key_values
            if pkv is None:
                return None
            # Deep clone the tuple of tuples of tensors
            return tuple(tuple(t.clone() for t in layer) for layer in pkv)
            
        return None
    
    def prepare_generation_inputs(
        self,
        expert_id: str,
        expert_prompt: str,
        user_message: str
    ) -> Dict[str, Any]:
        """
        Prepare inputs for generation, incorporating shared context.
        
        Returns a dict with:
        - input_text: The text to tokenize
        - past_key_values: KV cache to use (or None)
        - mode: Which mode was used
        """
        result = {
            "mode": "none",
            "past_key_values": None,
            "input_text": f"{expert_prompt}\n\nUser: {user_message}\nAssistant:"
        }
        
        if not self.config.enabled:
            self.metrics["generations_without_shared"] += 1
            return result
        
        if self.config.mode == SharedContextMode.KV_CACHE_PREFIX:
            if self.shared_cache and self.shared_cache.past_key_values:
                result["mode"] = "kv_cache"
                result["past_key_values"] = self.shared_cache.past_key_values
                self.metrics["generations_with_shared"] += 1
                logger.debug(f"Using shared KV cache for {expert_id}")
        
        elif self.config.mode == SharedContextMode.TEXT_PREAMBLE:
            if self.shared_cache and self.shared_cache.context_text:
                result["mode"] = "text"
                result["input_text"] = (
                    f"{self.shared_cache.context_text}\n\n"
                    f"{expert_prompt}\n\nUser: {user_message}\nAssistant:"
                )
                self.metrics["generations_with_shared"] += 1
                logger.debug(f"Using text preamble for {expert_id}")
        
        return result
    
    def generate_with_shared_context(
        self,
        expert_id: str,
        expert_prompt: str,
        user_message: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """
        Generate from an expert using shared context.
        
        This is the main generation method that incorporates
        the shared latent space (if enabled).
        
        Args:
            expert_id: ID of the expert generating
            expert_prompt: The expert's persona prompt
            user_message: The user's input
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation kwargs
            
        Returns:
            Generated text response
        """
        # Update expert state
        self.update_expert_state(expert_id, status="working")
        
        # Prepare inputs
        prep = self.prepare_generation_inputs(expert_id, expert_prompt, user_message)
        
        # Tokenize
        inputs = self.tokenizer(
            prep["input_text"],
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            **kwargs
        }
        
        if prep["past_key_values"] is not None:
            gen_kwargs["past_key_values"] = prep["past_key_values"]
        
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)
        
        # Decode
        input_length = inputs["input_ids"].shape[1]
        response = self.tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        )
        
        # Update expert state
        self.update_expert_state(
            expert_id,
            status="done",
            last_output_summary=response[:100] + "..." if len(response) > 100 else response
        )
        
        return response.strip()
    
    # =========================================================================
    # METRICS & DEBUGGING
    # =========================================================================
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics about shared space usage."""
        return {
            **self.metrics,
            "enabled": self.config.enabled,
            "mode": self.config.mode.value,
            "has_cache": self.shared_cache is not None,
            "cache_tokens": self.shared_cache.token_count if self.shared_cache else 0,
            "registered_experts": list(self.expert_states.keys())
        }
    
    def reset_metrics(self):
        """Reset usage metrics."""
        self.metrics = {
            "cache_computations": 0,
            "generations_with_shared": 0,
            "generations_without_shared": 0,
            "fallbacks_to_text": 0
        }
    
    def clear(self):
        """Clear all cached state."""
        self.shared_cache = None
        self.expert_states.clear()
        self.reset_metrics()


# =============================================================================
# INTEGRATION WITH TRANSFORMERS BACKEND
# =============================================================================

def create_shared_latent_space(
    backend,
    model_id: str,
    config: Optional[SharedLatentConfig] = None
) -> SharedLatentSpace:
    """
    Create a SharedLatentSpace from a TransformersBackend.
    
    Args:
        backend: The TransformersBackend instance
        model_id: ID of the loaded model to use
        config: Optional configuration
        
    Returns:
        Configured SharedLatentSpace
    """
    if model_id not in backend._loaded_models:
        raise ValueError(f"Model {model_id} not loaded in backend")
    
    model = backend._loaded_models[model_id]
    tokenizer = backend._tokenizers[model_id]
    
    return SharedLatentSpace(model, tokenizer, config)


# =============================================================================
# TESTING UTILITIES
# =============================================================================

class SharedLatentExperiment:
    """
    Utility for running A/B experiments with shared latent space.
    
    Allows comparing generation quality/behavior with and without
    shared context at various settings.
    """
    
    def __init__(self, shared_space: SharedLatentSpace):
        self.shared_space = shared_space
        self.results: List[Dict] = []
    
    def run_comparison(
        self,
        expert_id: str,
        expert_prompt: str,
        user_message: str,
        shared_context: str
    ) -> Dict[str, Any]:
        """
        Run the same generation with and without shared context.
        
        Returns comparison data for analysis.
        """
        # Set shared context
        self.shared_space.set_shared_context(shared_context)
        
        # Run with shared context
        self.shared_space.enable()
        response_with = self.shared_space.generate_with_shared_context(
            expert_id, expert_prompt, user_message
        )
        
        # Run without shared context
        self.shared_space.disable()
        response_without = self.shared_space.generate_with_shared_context(
            expert_id, expert_prompt, user_message
        )
        
        # Re-enable
        self.shared_space.enable()
        
        result = {
            "expert_id": expert_id,
            "user_message": user_message,
            "shared_context": shared_context[:200] + "...",
            "response_with_shared": response_with,
            "response_without_shared": response_without,
            "length_with": len(response_with),
            "length_without": len(response_without)
        }
        
        self.results.append(result)
        return result
    
    def get_summary(self) -> Dict[str, Any]:
        """Summarize experiment results."""
        if not self.results:
            return {"total_comparisons": 0}
        
        return {
            "total_comparisons": len(self.results),
            "avg_length_with_shared": sum(r["length_with"] for r in self.results) / len(self.results),
            "avg_length_without_shared": sum(r["length_without"] for r in self.results) / len(self.results)
        }
