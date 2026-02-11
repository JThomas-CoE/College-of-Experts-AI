
from typing import List, Dict, Tuple, Any, Optional
from PIL import Image
from .model_factory import ModelCapabilities

class UniversalChatFormatter:
    """
    Handles prompt formatting for different model architectures (e.g. Qwen vs LFM).
    Abstracts differences in:
    - Image token syntax (<|vision_start|> vs <image>)
    - Input structure (list of dicts vs strings)
    - Processor call signatures
    """
    
    def __init__(self, processor: Any, capabilities: ModelCapabilities):
        self.processor = processor
        self.capabilities = capabilities
    
    def format_messages(self, messages: List[Dict]) -> Tuple[str, List[Image.Image]]:
        """
        Convert structured messages (CoE format) into model-compatible inputs.
        
        Args:
            messages: List of {"role": "...", "content": [...]}
            
        Returns:
            Tuple(prompt_text, list_of_images)
        """
        # --- PRE-FLIGHT: Normalize for text-only models ---
        # If the model doesn't support vision, we MUST flatten structured content to strings
        # or the chat template (especially for Qwen Coder / Llama) will fail with TypeError.
        if not self.capabilities.has_vision or self.capabilities.model_family == "generic":
            messages = self._flatten_messages(messages)

        # Qwen-VL handles structured list-of-dicts natively via its template
        if self.capabilities.model_family == "qwen":
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            # Standard extraction
            images = self._extract_images(messages)
            return text, images

        # LFM-VL (and others) usually prefer explicit image tokens in text
        elif self.capabilities.model_family == "lfm":
            token = self.capabilities.image_token or "<image>"
            formatted_messages = []
            all_images = []
            
            for msg in messages:
                role = msg["role"]
                content_list = msg["content"]
                
                # Robustness: handles both flattened strings and list-of-dicts
                if isinstance(content_list, str):
                    content_list = [{"type": "text", "text": content_list}]
                
                parts = []
                for item in content_list:
                    if item["type"] == "text":
                        parts.append(item["text"])
                    elif item["type"] == "image":
                        parts.append(token)
                        if "image" in item:
                            all_images.append(item["image"])
                
                # Join content
                full_content = "\n".join(parts)
                formatted_messages.append({"role": role, "content": full_content})
            
            try:
                text = self.processor.apply_chat_template(
                    formatted_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            except Exception as e:
                # Fallback to manual ChatML if template fails/missing
                print(f"[ChatFormatter] Template failed ({e}), using manual ChatML...")
                text = ""
                for msg in formatted_messages:
                    role = msg["role"]
                    content = msg["content"]
                    if role == "system":
                        text += f"<|im_start|>system\n{content}<|im_end|>\n"
                    elif role == "user":
                        text += f"<|im_start|>user\n{content}<|im_end|>\n"
                    elif role == "assistant":
                        text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
                text += "<|im_start|>assistant\n"
            
            return text, all_images
            
        else:
            # Fallback (flattened text only)
            if not isinstance(messages[0]["content"], str):
                messages = self._flatten_messages(messages)
            
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return text, []

    def _flatten_messages(self, messages: List[Dict]) -> List[Dict]:
        """Convert list-of-dicts content into simple strings for text-only models."""
        flattened = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    # Images are ignored for text-only flattening
                content = "\n".join(text_parts)
            
            flattened.append({"role": msg["role"], "content": content})
        return flattened

    def _extract_images(self, messages: List[Dict]) -> List[Image.Image]:
        images = []
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "image":
                        images.append(item.get("image"))
        return images

    def prepare_inputs(self, prompt: str, images: List[Image.Image], device: str) -> Dict[str, Any]:
        """
        Prepare the final inputs dict for the model.
        """
        if self.capabilities.model_family == "qwen":
            inputs = self.processor(
                text=[prompt],
                images=images if images else None,
                return_tensors="pt",
                padding=True
            )
        elif self.capabilities.model_family == "lfm":
            # LFM 1.6B VL likely uses 'images' kwarg too, similar to Qwen/Llava
            # Note: If no images, pass None
            if not images:
                inputs = self.processor(text=prompt, return_tensors="pt")
            else:
                inputs = self.processor(
                    text=prompt,
                    images=images, 
                    return_tensors="pt"
                )
        else:
            # Text only / generic
            inputs = self.processor(text=prompt, return_tensors="pt")
            
        # Remove token_type_ids if present (LFM/Llama don't use them)
        if "token_type_ids" in inputs:
            del inputs["token_type_ids"]
            
        return {k: v.to(device) for k, v in inputs.items()}
