"""
FLM Backend - Uses AMD FLM serve for NPU inference.

FLM (FastFlowLM) runs models on AMD Ryzen AI NPU via an OpenAI-compatible API.
This backend calls the FLM server for router/council operations.
"""

import os
import base64
import requests
from typing import List, Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class FLMConfig:
    """Configuration for FLM backend."""
    host: str = "localhost"
    port: int = 52625
    model: str = "qwen3vl-it:4b"
    timeout: int = 600


class FLMBackend:
    """
    Backend for FLM (FastFlowLM) NPU inference server.
    Provides OpenAI-compatible API for text and vision-language models.
    """
    
    def __init__(self, config: FLMConfig = None):
        self.config = config or FLMConfig()
        self.base_url = f"http://{self.config.host}:{self.config.port}"
        self.device = "npu"
        
        print(f"[FLMBackend] Connecting to {self.base_url}")
        print(f"[FLMBackend] Target Model: {self.config.model}")
        
        # Verify and potentially start server/model
        import time
        connected = self._ensure_model_running()
        
        if not connected:
            print(f"[FLMBackend] WARNING: Could not confirm {self.config.model} is running.")
            print(f"[FLMBackend] Manual action may be required: flm serve {self.config.model}")
        
        # Mark as API backend (vs HuggingFace model)
        self.is_api_backend = True

    def _ensure_model_running(self) -> bool:
        """Verify model is running, attempt to start if not."""
        import subprocess
        import time
        import sys
        
        server_url = f"{self.base_url}/v1/models"
        print(f"[FLMBackend] Checking NPU Server at {server_url}...")

        # 1. Quick Check: Is it already running?
        try:
            resp = requests.get(server_url, timeout=2)
            if resp.status_code == 200:
                print(f"[FLMBackend] Server is ACTIVE. ({resp.json().get('data', [{}])[0].get('id', 'unknown')})")
                return True
        except requests.RequestException:
            # Not running, proceed to start
            pass

        # 2. Start Server
        print(f"[FLMBackend] Server not found. Starting: flm serve {self.config.model}")
        
        try:
            # Use CREATE_NEW_CONSOLE for Windows to ensure it lives if this script dies
            # and verify environment is passed
            flags = subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
            
            # We assume 'flm' is in the PATH. If this fails, user env is wrong.
            subprocess.Popen(
                ["flm", "serve", self.config.model],
                creationflags=flags,
                env=os.environ
            )
        except Exception as e:
            print(f"[FLMBackend] CRITICAL ERROR starting flm: {e}")
            return False

        # 3. Wait for Startup (up to 45s)
        print("[FLMBackend] Waiting for server initialization...")
        for i in range(45):
            time.sleep(1)
            try:
                resp = requests.get(server_url, timeout=1)
                if resp.status_code == 200:
                    print(f"[FLMBackend] Server successfully started (Attempt {i+1})")
                    return True
            except requests.RequestException:
                if i % 5 == 0:
                    print(f"  ... waiting ({i}s)")
                continue
        
        print("[FLMBackend] VALIDATION FAILED: Server did not respond after 45s.")
        return False

    
    def generate(
        self,
        messages: List[Dict] = None,
        prompt: str = None,
        images: List[Any] = None,
        max_tokens: int = 500,
        temperature: float = 0.7,
        stop: List[str] = None,
        **kwargs
    ) -> str:
        """
        Generate text using FLM server.
        
        Args:
            messages: Chat messages in OpenAI format
            prompt: Simple text prompt (converted to messages)
            images: Optional images (PIL Image or base64 strings)
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text string
        """
        # Build messages if not provided
        if messages is None:
            if prompt:
                messages = [{"role": "user", "content": prompt}]
            else:
                raise ValueError("Either messages or prompt required")
        
        # Handle images in messages
        if images:
            messages = self._inject_images(messages, images)
        
        # Build request
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }
        
        if stop:
            payload["stop"] = stop
        
        # Call FLM API
        try:
            resp = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.config.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            
            # Extract response
            choices = data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "")
                return content
            
            return ""
            
        except requests.exceptions.Timeout:
            print(f"[FLMBackend] Request timed out after {self.config.timeout}s")
            return ""
        except Exception as e:
            print(f"[FLMBackend] Error: {e}")
            return ""
    
    def _inject_images(self, messages: List[Dict], images: List[Any]) -> List[Dict]:
        """Inject base64 images into messages for vision models."""
        from PIL import Image
        import io
        
        # Convert images to base64
        image_b64_list = []
        for img in images:
            if isinstance(img, str):
                # Already base64
                image_b64_list.append(img)
            elif isinstance(img, Image.Image):
                # PIL Image - convert to base64
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                image_b64_list.append(f"data:image/png;base64,{b64}")
            elif hasattr(img, "tobytes"):
                # Numpy array or similar
                pil_img = Image.fromarray(img)
                buffer = io.BytesIO()
                pil_img.save(buffer, format="PNG")
                b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
                image_b64_list.append(f"data:image/png;base64,{b64}")
        
        # Inject into last user message
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                content = messages[i].get("content", "")
                
                # Convert to multimodal format
                if isinstance(content, str):
                    new_content = []
                    for b64 in image_b64_list:
                        new_content.append({
                            "type": "image_url",
                            "image_url": {"url": b64}
                        })
                    new_content.append({"type": "text", "text": content})
                    messages[i]["content"] = new_content
                break
        
        return messages
    
    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 500,
        **kwargs
    ) -> str:
        """Simple chat interface with system + user prompts."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        return self.generate(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
    
    def to(self, device):
        """Compatibility stub."""
        return self


# Convenience function
def create_flm_backend(
    host: str = "localhost",
    port: int = 52625,
    model: str = "qwen3vl-it:4b"
) -> FLMBackend:
    """Create an FLM backend instance."""
    config = FLMConfig(host=host, port=port, model=model)
    return FLMBackend(config)
