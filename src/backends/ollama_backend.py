"""
Ollama Backend - Model serving via Ollama API.

This backend uses the Ollama API for optimized inference with
quantized models (GGUF format via llama.cpp).

Features:
- Fast inference with quantized models
- Simple setup (just ollama pull)
- Good for single-model hot inference
- Lower memory footprint with quantization
- Auto-start Ollama server if not running
"""

import subprocess
import time
import platform
from typing import Optional, Dict, List, Any

from .base import BaseBackend, BackendType, ModelInfo, GenerationConfig

# Lazy import
_ollama = None


def _get_ollama():
    global _ollama
    if _ollama is None:
        import ollama
        _ollama = ollama
    return _ollama


class OllamaBackend(BaseBackend):
    """
    Model backend using Ollama API.
    
    Uses llama.cpp under the hood for optimized inference.
    Best for single hot model with fastest generation speed.
    """
    
    def __init__(
        self,
        host: str = "http://localhost:11434",
        auto_pull: bool = False,
        auto_start: bool = True
    ):
        """
        Initialize Ollama backend.
        
        Args:
            host: Ollama server URL
            auto_pull: Automatically pull models if not found
            auto_start: Automatically start Ollama server if not running
        """
        super().__init__(device="ollama")
        self.host = host
        self.auto_pull = auto_pull
        self.auto_start = auto_start
        
        # Track which models we've "loaded" (verified available)
        self._available_models: Dict[str, str] = {}  # model_id -> ollama_model_name
        
        # Auto-start Ollama if enabled
        if self.auto_start:
            self._ensure_ollama_running()
    
    def _ensure_ollama_running(self, max_retries: int = 3, retry_delay: float = 2.0) -> bool:
        """
        Ensure Ollama server is running, starting it if necessary.
        
        Args:
            max_retries: Number of times to retry connection after starting
            retry_delay: Seconds to wait between retries
            
        Returns:
            True if Ollama is running, False otherwise
        """
        # Check if already running
        if self.is_ollama_running():
            return True
        
        print("Ollama not running, attempting to start...")
        
        # Start Ollama based on platform
        try:
            if platform.system() == "Windows":
                # Windows: Start in background with minimized window
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            else:
                # Linux/Mac: Start in background
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True
                )
        except FileNotFoundError:
            print("ERROR: 'ollama' command not found. Please install Ollama.")
            return False
        except Exception as e:
            print(f"ERROR: Failed to start Ollama: {e}")
            return False
        
        # Wait for Ollama to be ready
        for attempt in range(max_retries):
            time.sleep(retry_delay)
            if self.is_ollama_running():
                print(f"Ollama started successfully (attempt {attempt + 1})")
                return True
            print(f"Waiting for Ollama to start (attempt {attempt + 1}/{max_retries})...")
        
        print("ERROR: Ollama failed to start after retries")
        return False
    
    @property
    def backend_type(self) -> BackendType:
        return BackendType.OLLAMA
    
    def _check_model_available(self, model_name: str) -> bool:
        """Check if model is available in Ollama."""
        ollama = _get_ollama()
        try:
            ollama.show(model_name)
            return True
        except Exception:
            return False
    
    def _pull_model(self, model_name: str):
        """Pull model from Ollama registry."""
        ollama = _get_ollama()
        print(f"Pulling {model_name}...")
        ollama.pull(model_name)
        print(f"Pulled {model_name}")
    
    def load_model(
        self,
        model_id: str,
        model_path: str,
        device: Optional[str] = None,
        **kwargs
    ) -> ModelInfo:
        """
        Verify model is available in Ollama.
        
        For Ollama, model_path is the Ollama model name (e.g., "qwen3:4b")
        """
        if model_id in self._available_models:
            return self._models[model_id]
        
        ollama_model = model_path
        
        # Check if model exists
        if not self._check_model_available(ollama_model):
            if self.auto_pull:
                self._pull_model(ollama_model)
            else:
                raise RuntimeError(
                    f"Model {ollama_model} not found in Ollama. "
                    f"Run: ollama pull {ollama_model}"
                )
        
        self._available_models[model_id] = ollama_model
        
        # Get model info
        ollama = _get_ollama()
        try:
            model_info = ollama.show(ollama_model)
            size_bytes = model_info.get("size", 0)
        except Exception:
            size_bytes = None
        
        info = ModelInfo(
            model_id=model_id,
            model_path=ollama_model,
            backend_type=self.backend_type,
            is_loaded=True,
            memory_bytes=size_bytes,
            device="ollama",
            metadata={"ollama_model": ollama_model}
        )
        self._models[model_id] = info
        
        print(f"Ollama model ready: {model_id} -> {ollama_model}")
        return info
    
    def unload_model(self, model_id: str) -> bool:
        """
        Mark model as unloaded.
        
        Note: Ollama manages its own model loading/unloading.
        This just removes our reference.
        """
        if model_id not in self._available_models:
            return False
        
        del self._available_models[model_id]
        if model_id in self._models:
            self._models[model_id].is_loaded = False
        
        print(f"Ollama model unlinked: {model_id}")
        return True
    
    def generate(
        self,
        model_id: str,
        messages: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate using Ollama API."""
        if model_id not in self._available_models:
            raise ValueError(f"Model {model_id} not loaded")
        
        ollama = _get_ollama()
        ollama_model = self._available_models[model_id]
        config = config or GenerationConfig()
        
        # Build messages with system prompt
        full_messages = []
        if system_prompt:
            full_messages.append({"role": "system", "content": system_prompt})
        full_messages.extend(messages)
        
        # Call Ollama
        response = ollama.chat(
            model=ollama_model,
            messages=full_messages,
            options={
                "temperature": config.temperature,
                "num_predict": config.max_tokens,
                "top_p": config.top_p,
                "top_k": config.top_k,
            }
        )
        
        return response["message"]["content"]
    
    def list_ollama_models(self) -> List[str]:
        """List all models available in Ollama."""
        ollama = _get_ollama()
        try:
            models = ollama.list()
            return [m["name"] for m in models.get("models", [])]
        except Exception:
            return []
    
    def is_ollama_running(self) -> bool:
        """Check if Ollama server is running."""
        try:
            self.list_ollama_models()
            return True
        except Exception:
            return False


if __name__ == "__main__":
    # Quick test
    backend = OllamaBackend()
    
    if backend.is_ollama_running():
        print("Ollama is running")
        print(f"Available models: {backend.list_ollama_models()}")
    else:
        print("Ollama is not running")
