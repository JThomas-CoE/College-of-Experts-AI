"""
NPU Backend for LiquidAI LFM2.5-VL ONNX Model

Uses ONNX Runtime with VitisAIExecutionProvider for AMD Ryzen AI NPU.
Implements manual generation loop with KV-cache for split ONNX model.
"""

import os
import numpy as np
from typing import Optional, Dict, Any, List
from transformers import AutoProcessor


class NPUBackend:
    """
    ONNX Runtime backend for NPU inference.
    Loads: embed_images, embed_tokens, decoder (split model architecture).
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = "npu"
        
        try:
            import onnxruntime as ort
            self.ort = ort
        except ImportError:
            raise ImportError("onnxruntime required. Install: pip install onnxruntime")
        
        print(f"[NPUBackend] Initializing from {model_path}")
        
        # Verify ONNX directory exists
        onnx_dir = os.path.join(model_path, "onnx")
        if not os.path.exists(onnx_dir):
            raise FileNotFoundError(f"ONNX directory not found: {onnx_dir}")
        
        # Define model paths (fp16 encoder + q4 decoder per README recommendation)
        self.paths = {
            "embed_images": os.path.join(onnx_dir, "embed_images_fp16.onnx"),
            "embed_tokens": os.path.join(onnx_dir, "embed_tokens_fp16.onnx"),
            "decoder": os.path.join(onnx_dir, "decoder_q4.onnx")
        }
        
        # Verify files exist
        for name, path in self.paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing {name}: {path}")
        
        # Configure providers - NPU priority
        available = ort.get_available_providers()
        print(f"[NPUBackend] Available providers: {available}")
        
        # VitisAI for Ryzen AI NPU
        providers = []
        if 'VitisAIExecutionProvider' in available:
            providers.append('VitisAIExecutionProvider')
        if 'DmlExecutionProvider' in available:
            providers.append('DmlExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        print(f"[NPUBackend] Using providers: {providers}")
        
        # Load sessions
        print("[NPUBackend] Loading embed_images...")
        self.sess_embed_images = ort.InferenceSession(self.paths["embed_images"], providers=providers)
        
        print("[NPUBackend] Loading embed_tokens...")
        self.sess_embed_tokens = ort.InferenceSession(self.paths["embed_tokens"], providers=providers)
        
        print("[NPUBackend] Loading decoder...")
        self.sess_decoder = ort.InferenceSession(self.paths["decoder"], providers=providers)
        
        # Report active provider
        active = self.sess_decoder.get_providers()
        print(f"[NPUBackend] Active Providers: {active}")
        if 'VitisAIExecutionProvider' in active:
            print("[NPUBackend] Running on Ryzen AI NPU (Vitis AI)")
        elif 'DmlExecutionProvider' in active:
            print("[NPUBackend] Running on NPU/iGPU (DirectML)")
        else:
            print("[NPUBackend] WARNING: Running on CPU only!")

        
        # Load tokenizer (ONNX model uses tokenizers backend, not standard HF)
        print("[NPUBackend] Loading tokenizer...")
        try:
            from transformers import PreTrainedTokenizerFast
            tokenizer_path = os.path.join(model_path, "tokenizer.json")
            self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_path)
            # Set special tokens from config
            self.tokenizer.bos_token = "<|startoftext|>"
            self.tokenizer.eos_token = "<|im_end|>"
            self.tokenizer.pad_token = "<|pad|>"
        except Exception as e:
            print(f"[NPUBackend] Tokenizer fallback: {e}")
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        self.processor = self.tokenizer  # Alias for compatibility
        
        # Cache dtype mapping
        self.ONNX_DTYPE = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(int64)": np.int64
        }
        
        print("[NPUBackend] Ready")
    
    def _init_kv_cache(self) -> Dict[str, np.ndarray]:
        """Initialize empty KV cache for decoder."""
        cache = {}
        for inp in self.sess_decoder.get_inputs():
            name = inp.name
            if name in {"inputs_embeds", "attention_mask", "position_ids"}:
                continue
            
            # Build shape with 0 sequence length
            shape = []
            for d in inp.shape:
                if isinstance(d, int):
                    shape.append(d)
                elif "sequence" in str(d).lower():
                    shape.append(0)
                else:
                    shape.append(1)  # batch
            
            dtype = self.ONNX_DTYPE.get(inp.type, np.float32)
            cache[name] = np.zeros(shape, dtype=dtype)
        
        return cache
    
    def _update_kv_cache(self, cache: Dict, outputs: List, output_names: List[str]):
        """Update cache from decoder outputs."""
        for i, name in enumerate(output_names):
            if name == "logits":
                continue
            # Map present -> past
            new_name = name.replace("present_conv", "past_conv").replace("present.", "past_key_values.")
            if new_name in cache:
                cache[new_name] = outputs[i]
    
    def generate(
        self,
        input_ids=None,
        pixel_values=None,
        pixel_attention_mask=None,
        spatial_shapes=None,
        max_new_tokens: int = 500,
        temperature: float = 0.7,
        do_sample: bool = True,
        repetition_penalty: float = 1.0,
        **kwargs
    ) -> List[List[int]]:
        """
        Generate tokens using ONNX Runtime.
        
        Args:
            input_ids: Token IDs [batch, seq]
            pixel_values: Image tensor [batch, channels, h, w] (optional)
            max_new_tokens: Max tokens to generate
            temperature: Sampling temperature
            
        Returns:
            List of token ID sequences (including input)
        """
        # Handle inputs from kwargs
        if input_ids is None and "inputs" in kwargs:
            inputs = kwargs["inputs"]
            input_ids = inputs.get("input_ids")
            pixel_values = inputs.get("pixel_values", pixel_values)
            pixel_attention_mask = inputs.get("pixel_attention_mask", pixel_attention_mask)
            spatial_shapes = inputs.get("spatial_shapes", spatial_shapes)
        
        # Convert to numpy
        if hasattr(input_ids, "cpu"):
            input_ids = input_ids.cpu().numpy()
        input_ids = np.array(input_ids, dtype=np.int64)
        
        if input_ids.ndim == 1:
            input_ids = np.expand_dims(input_ids, 0)
        
        # Get token embeddings
        token_outputs = self.sess_embed_tokens.run(None, {"input_ids": input_ids})
        inputs_embeds = token_outputs[0].astype(np.float32)
        
        # Handle vision if provided
        if pixel_values is not None:
            if hasattr(pixel_values, "cpu"):
                pixel_values = pixel_values.cpu().numpy()
            if hasattr(pixel_attention_mask, "cpu"):
                pixel_attention_mask = pixel_attention_mask.cpu().numpy()
            if hasattr(spatial_shapes, "cpu"):
                spatial_shapes = spatial_shapes.cpu().numpy()
            
            # Run vision encoder
            vision_feed = {
                "pixel_values": pixel_values.astype(np.float32),
                "pixel_attention_mask": pixel_attention_mask.astype(np.int64),
                "spatial_shapes": spatial_shapes.astype(np.int64)
            }
            # Filter to valid inputs
            valid_vision_inputs = [i.name for i in self.sess_embed_images.get_inputs()]
            vision_feed = {k: v for k, v in vision_feed.items() if k in valid_vision_inputs and v is not None}
            
            try:
                image_outputs = self.sess_embed_images.run(None, vision_feed)
                image_embeds = image_outputs[0]
                
                # Replace <image> tokens with image embeddings
                if hasattr(self.processor, 'image_token_id'):
                    img_token_id = self.processor.image_token_id
                else:
                    img_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
                
                img_positions = np.where(input_ids[0] == img_token_id)[0]
                for i, pos in enumerate(img_positions):
                    if i < len(image_embeds):
                        inputs_embeds[0, pos] = image_embeds[i]
            except Exception as e:
                print(f"[NPUBackend] Vision encoding failed: {e}")
        
        # Initialize KV cache
        cache = self._init_kv_cache()
        
        # Generation loop
        generated = input_ids[0].tolist()
        seq_len = inputs_embeds.shape[1]
        curr_embeds = inputs_embeds
        
        output_names = [o.name for o in self.sess_decoder.get_outputs()]
        valid_decoder_inputs = [i.name for i in self.sess_decoder.get_inputs()]
        
        for step in range(max_new_tokens):
            # Build attention mask
            attn_mask = np.ones((1, seq_len), dtype=np.int64)
            
            # Decoder feed
            feed = {
                "inputs_embeds": curr_embeds,
                "attention_mask": attn_mask
            }
            feed.update(cache)
            
            # Filter to valid inputs only
            feed = {k: v for k, v in feed.items() if k in valid_decoder_inputs}
            
            # Run decoder
            outputs = self.sess_decoder.run(None, feed)
            logits = outputs[0]
            
            # Get next token logits
            next_logits = logits[:, -1, :].astype(np.float64)
            
            # Temperature scaling
            if temperature > 0 and temperature != 1.0:
                next_logits = next_logits / temperature
            
            # Repetition penalty
            if repetition_penalty != 1.0:
                for token in set(generated):
                    if next_logits[0, token] < 0:
                        next_logits[0, token] *= repetition_penalty
                    else:
                        next_logits[0, token] /= repetition_penalty
            
            # Sample or greedy
            if do_sample and temperature > 0:
                probs = np.exp(next_logits - np.max(next_logits))
                probs = probs / probs.sum()
                next_token = int(np.random.choice(len(probs[0]), p=probs[0]))
            else:
                next_token = int(np.argmax(next_logits))
            
            generated.append(next_token)
            
            # Check EOS
            if next_token == self.tokenizer.eos_token_id:
                break
            
            # Update cache
            self._update_kv_cache(cache, outputs, output_names)
            
            # Embed next token
            next_ids = np.array([[next_token]], dtype=np.int64)
            next_embed = self.sess_embed_tokens.run(None, {"input_ids": next_ids})
            curr_embeds = next_embed[0].astype(np.float32)
            seq_len += 1
        
        return [generated]
    
    def to(self, device):
        """Compatibility stub."""
        return self
