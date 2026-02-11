import os
import time
import onnxruntime_genai as og
from typing import List, Optional

class OGABackend:
    """
    High-performance backend for ONNX GenAI models using DirectML.
    Optimized for AMD Radeon GPUs on Windows.
    """
    
    # VRAM-aware context limits
    DEFAULT_MAX_CONTEXT = 3072
    VRAM_PRESSURE_LIMITS = {
        "critical": 1024,
        "high": 2048,
        "elevated": 2560,
        "normal": 3072
    }
    
    def __init__(self, model_path: str, vram_budget_mb: int = 32000):
        print(f"[OGABackend] Initializing model from {model_path}...")
        self.model_path = model_path
        
        # Load model and tokenizer
        # OGA automatically detects genai_config.json and sets up the session
        start_time = time.time()
        self.model = og.Model(model_path)
        self.tokenizer = og.Tokenizer(self.model)
        self.tokenizer_stream = self.tokenizer.create_stream()
        self.is_api_backend = True # NEW: Signal that it handles internal formatting
        self.vram_budget_mb = vram_budget_mb
        self.current_vram_usage_mb = 0
        print(f"[OGABackend] Model loaded in {time.time() - start_time:.2f}s")
    
    def set_vram_budget(self, vram_budget_mb: int):
        """Update VRAM budget for dynamic context limiting."""
        self.vram_budget_mb = vram_budget_mb
    
    def update_vram_usage(self, usage_mb: int):
        """Update current VRAM usage estimate."""
        self.current_vram_usage_mb = usage_mb
    
    def _get_safe_context_length(self, input_tokens: int, max_tokens: int) -> int:
        """Get VRAM-safe context length based on current usage."""
        if self.vram_budget_mb <= 0:
            return min(len(input_tokens) + max_tokens, self.DEFAULT_MAX_CONTEXT)
        
        usage_ratio = self.current_vram_usage_mb / self.vram_budget_mb
        
        if usage_ratio >= 0.95:
            max_total = self.VRAM_PRESSURE_LIMITS["critical"]
        elif usage_ratio >= 0.90:
            max_total = self.VRAM_PRESSURE_LIMITS["high"]
        elif usage_ratio >= 0.80:
            max_total = self.VRAM_PRESSURE_LIMITS["elevated"]
        else:
            max_total = self.VRAM_PRESSURE_LIMITS["normal"]
        
        return min(max_total, len(input_tokens) + max_tokens)

    def format_messages(self, messages: List[dict]) -> str:
        """Simple ChatML formatter for OGA models (Qwen/Phi)."""
        prompt = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    def generate(
        self, 
        messages: Optional[List[dict]] = None,
        prompt: Optional[str] = None,
        model_id: str = "default",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        V7 Compatible Generation Interface.
        """
        # 1. Resolve prompt vs messages
        if messages is None:
            if prompt:
                # If prompt is a string, use it
                messages = [{"role": "user", "content": prompt}]
            else:
                return ""

        if isinstance(messages, str):
            final_prompt = messages
        else:
            # If system_prompt provided separately, ensure it's at the start
            if system_prompt and not any(m.get("role") == "system" for m in messages):
                messages = [{"role": "system", "content": system_prompt}] + messages
            final_prompt = self.format_messages(messages)

        # 2. Tokenize
        input_tokens = self.tokenizer.encode(final_prompt)
        
        # 3. Setup Generator
        params = og.GeneratorParams(self.model)
        
        # V11 VRAM Safety: Cap total context length at 3072 to prevent KV cache explosion
        # V12 VRAM Safety: Use dynamic context limiting based on VRAM pressure
        total_len = self._get_safe_context_length(input_tokens, max_tokens)
        
        params.set_search_options(
            max_length=total_len,
            temperature=temperature,
            do_sample=True if temperature > 0 else False
        )
        generator = og.Generator(self.model, params)
        generator.append_tokens(input_tokens)
        
        # 4. Loop
        output_text = ""
        try:
            while not generator.is_done():
                generator.generate_next_token()
                new_token = generator.get_next_tokens()[0]
                chunk = self.tokenizer_stream.decode(new_token)
                output_text += chunk
        except Exception as e:
            print(f"[OGABackend] Error during generation: {e}")
        finally:
            # Force immediate resource release
            del generator
            del params
            
        return output_text.strip()

    def __call__(self, prompt: str, **kwargs):
        """Standard interface for compatibility."""
        # For legacy calls
        messages = [{"role": "user", "content": prompt}]
        return self.generate("default", messages, **kwargs)
