"""
Transformers Backend - Pure Python model serving via HuggingFace.

This backend uses HuggingFace transformers for model loading and inference,
supporting various devices including DirectML for AMD GPUs on Windows.

Features:
- Multiple models can be loaded simultaneously
- Automatic device selection (CUDA, DirectML, CPU)
- Memory-efficient loading with device_map="auto"
- Compatible with most HuggingFace models
"""

import gc
from typing import Optional, Dict, List, Any
from pathlib import Path

from .base import BaseBackend, BackendType, ModelInfo, GenerationConfig

# Lazy imports to avoid loading torch until needed
_torch = None
_transformers = None


def _get_torch():
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _get_transformers():
    global _transformers
    if _transformers is None:
        import transformers
        _transformers = transformers
    return _transformers


class TransformersBackend(BaseBackend):
    """
    Model backend using HuggingFace Transformers.
    
    Supports CUDA, DirectML (AMD on Windows), and CPU inference.
    Can hold multiple models in memory simultaneously.
    """
    
    def __init__(
        self,
        device: str = "auto",
        torch_dtype: str = "float16",
        use_flash_attention: bool = False,
        max_memory: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the Transformers backend.
        
        Args:
            device: Target device (cuda, cpu, directml, auto)
            torch_dtype: Model dtype (float16, bfloat16, float32)
            use_flash_attention: Enable flash attention if available
            max_memory: Per-device memory limits for device_map
        """
        super().__init__(device=device)
        
        self.torch_dtype = torch_dtype
        self.use_flash_attention = use_flash_attention
        self.max_memory = max_memory
        
        # Model and tokenizer storage
        self._loaded_models: Dict[str, Any] = {}
        self._tokenizers: Dict[str, Any] = {}
    
    @property
    def backend_type(self) -> BackendType:
        return BackendType.TRANSFORMERS
    
    def _get_torch_dtype(self):
        """Convert string dtype to torch dtype."""
        torch = _get_torch()
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.torch_dtype, torch.float16)
    
    def _detect_best_device(self) -> str:
        """Detect best available device with DirectML support."""
        torch = _get_torch()
        
        # Check CUDA first
        if torch.cuda.is_available():
            return "cuda"
        
        # Check DirectML (Windows AMD)
        try:
            import torch_directml
            return "privateuseone"
        except ImportError:
            pass
        
        return "cpu"
    
    def load_model(
        self,
        model_id: str,
        model_path: str,
        device: Optional[str] = None,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs
    ) -> ModelInfo:
        """
        Load a model from HuggingFace Hub or local path.
        
        Args:
            model_id: Unique identifier for this model instance
            model_path: HuggingFace model path or local directory
            device: Target device (None for auto)
            load_in_8bit: Use 8-bit quantization (requires bitsandbytes)
            load_in_4bit: Use 4-bit quantization (requires bitsandbytes)
            **kwargs: Additional args passed to from_pretrained
        """
        if model_id in self._loaded_models:
            return self._models[model_id]
        
        torch = _get_torch()
        transformers = _get_transformers()
        
        device = self._resolve_device(device)
        print(f"Loading {model_path} on {device}...")
        
        # Load tokenizer
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        # Prepare model loading args
        model_kwargs = {
            "trust_remote_code": True,
            "dtype": self._get_torch_dtype(),  # Note: 'dtype' instead of deprecated 'torch_dtype'
        }
        
        # Handle quantization
        if load_in_8bit or load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                if load_in_4bit:
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=self._get_torch_dtype()
                    )
                elif load_in_8bit:
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True
                    )
            except ImportError:
                print("Warning: bitsandbytes not installed, loading without quantization")
        
        # Handle device mapping
        if device == "cpu":
            model_kwargs["device_map"] = "cpu"
        elif device in ("cuda", "privateuseone"):
            if self.max_memory:
                model_kwargs["device_map"] = "auto"
                model_kwargs["max_memory"] = self.max_memory
            else:
                model_kwargs["device_map"] = "auto"
        
        # Flash attention
        if self.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"
        
        model_kwargs.update(kwargs)
        
        # Load model
        try:
            model = transformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                **model_kwargs
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load model {model_path}: {e}")
        
        # Set pad token if needed
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Store model and tokenizer
        self._loaded_models[model_id] = model
        self._tokenizers[model_id] = tokenizer
        
        # Create model info
        memory_bytes = None
        try:
            memory_bytes = sum(
                p.numel() * p.element_size() 
                for p in model.parameters()
            )
        except Exception:
            pass
        
        info = ModelInfo(
            model_id=model_id,
            model_path=model_path,
            backend_type=self.backend_type,
            is_loaded=True,
            memory_bytes=memory_bytes,
            device=device,
            metadata={"dtype": self.torch_dtype}
        )
        self._models[model_id] = info
        
        print(f"Loaded {model_id} ({memory_bytes / 1e9:.1f}GB)" if memory_bytes else f"Loaded {model_id}")
        return info
    
    def unload_model(self, model_id: str) -> bool:
        """Unload model and free memory."""
        if model_id not in self._loaded_models:
            return False
        
        torch = _get_torch()
        
        # Delete model and tokenizer
        del self._loaded_models[model_id]
        del self._tokenizers[model_id]
        
        if model_id in self._models:
            self._models[model_id].is_loaded = False
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"Unloaded {model_id}")
        return True
    
    def generate(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate a response from the model.
        
        Args:
            model_id: ID of loaded model
            messages: Chat messages [{"role": "user", "content": "..."}]
            config: Generation configuration
            system_prompt: Optional system prompt
        """
        if model_id not in self._loaded_models:
            raise ValueError(f"Model {model_id} not loaded")
        
        torch = _get_torch()
        
        model = self._loaded_models[model_id]
        tokenizer = self._tokenizers[model_id]
        config = config or GenerationConfig()
        
        # Build messages with system prompt
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)
        
        # Apply chat template if available
        try:
            prompt = tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=True
            )
        except Exception:
            # Fallback for models without chat template
            prompt = self._format_messages_fallback(full_messages)
        
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Move to model device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                top_k=config.top_k,
                do_sample=config.temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode response (only new tokens)
        input_length = inputs["input_ids"].shape[1]
        response = tokenizer.decode(
            outputs[0][input_length:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def _format_messages_fallback(self, messages: List[Dict[str, str]]) -> str:
        """Simple fallback message formatting."""
        parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                parts.append(f"System: {content}\n")
            elif role == "user":
                parts.append(f"User: {content}\n")
            elif role == "assistant":
                parts.append(f"Assistant: {content}\n")
        parts.append("Assistant:")
        return "".join(parts)
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in GB."""
        torch = _get_torch()
        
        usage = {"models_loaded": len(self._loaded_models)}
        
        if torch.cuda.is_available():
            usage["cuda_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            usage["cuda_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
        
        # Check for DirectML (AMD GPUs on Windows)
        try:
            import torch_directml
            usage["dml_available"] = True
            usage["dml_warning"] = "Memory tracking unavailable for DML - using reservation estimates"
        except ImportError:
            pass
        
        return usage
    
    # =========================================================================
    # SHARED LATENT SPACE INTEGRATION
    # =========================================================================
    
    def create_shared_latent_space(
        self,
        model_id: str,
        enabled: bool = True,
        mode: str = "kv_cache"
    ):
        """
        Create a SharedLatentSpace for multi-expert coordination.
        
        This is an EXPERIMENTAL feature for exploring shared context
        between multiple expert instances using the same base model.
        
        Args:
            model_id: ID of the loaded model to use
            enabled: Whether shared space is enabled initially
            mode: "kv_cache" for latent sharing, "text" for text preamble baseline
            
        Returns:
            SharedLatentSpace instance
            
        Example:
            backend = TransformersBackend()
            backend.load_model("expert_base", "LiquidAI/LFM2.5-1.2B-Instruct")
            
            shared = backend.create_shared_latent_space("expert_base", enabled=True)
            shared.set_shared_context("Current task: Build auth system...")
            
            # Generate with shared context
            response = shared.generate_with_shared_context(
                expert_id="python_expert",
                expert_prompt=PYTHON_PROMPT,
                user_message="How should I handle JWT?"
            )
        """
        if model_id not in self._loaded_models:
            raise ValueError(f"Model {model_id} not loaded. Load it first with load_model()")
        
        # Lazy import to avoid circular dependencies
        from ..shared_latent import SharedLatentSpace, SharedLatentConfig, SharedContextMode
        
        mode_enum = SharedContextMode.KV_CACHE_PREFIX if mode == "kv_cache" else SharedContextMode.TEXT_PREAMBLE
        config = SharedLatentConfig(enabled=enabled, mode=mode_enum)
        
        return SharedLatentSpace(
            model=self._loaded_models[model_id],
            tokenizer=self._tokenizers[model_id],
            config=config
        )
    
    def generate_with_shared_context(
        self,
        model_id: str,
        shared_space,  # SharedLatentSpace
        expert_id: str,
        messages: List[Dict[str, str]],
        system_prompt: Optional[str] = None,
        config: Optional[GenerationConfig] = None
    ) -> str:
        """
        Generate using the standard interface but with shared latent context.
        
        This integrates SharedLatentSpace with the normal backend API.
        
        Args:
            model_id: ID of the model (should match shared_space's model)
            shared_space: The SharedLatentSpace instance
            expert_id: ID of the expert for state tracking
            messages: Chat messages
            system_prompt: Expert system prompt (persona)
            config: Generation configuration
        """
        if model_id not in self._loaded_models:
            raise ValueError(f"Model {model_id} not loaded")
        
        config = config or GenerationConfig()
        
        # Extract user message from messages
        user_messages = [m["content"] for m in messages if m.get("role") == "user"]
        user_message = user_messages[-1] if user_messages else ""
        
        # Use shared space for generation
        return shared_space.generate_with_shared_context(
            expert_id=expert_id,
            expert_prompt=system_prompt or "",
            user_message=user_message,
            max_new_tokens=config.max_tokens,
            temperature=config.temperature
        )


if __name__ == "__main__":
    # Quick test
    backend = TransformersBackend(device="auto")
    print(f"Best device: {backend._detect_best_device()}")
    print("TransformersBackend initialized successfully")
    print("\nShared Latent Space Support: Available")
    print("  Use backend.create_shared_latent_space(model_id) to create")

