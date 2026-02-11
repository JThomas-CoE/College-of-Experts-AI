
import os
import json
import torch
from dataclasses import dataclass
from typing import Optional, Tuple, Any
from transformers import (
    AutoModelForCausalLM, 
    AutoProcessor, 
    AutoConfig
)

@dataclass
class ModelCapabilities:
    has_vision: bool
    image_token: Optional[str] = None
    model_family: str = "unknown"  # "qwen", "lfm", "generic"

class ModelFactory:
    @staticmethod
    def detect_architecture(model_path: str) -> ModelCapabilities:
        """
        Inspect config.json to determine model architecture and capabilities.
        """
        try:
            # Check for OGA/ONNX first
            if os.path.exists(os.path.join(model_path, "genai_config.json")):
                return ModelCapabilities(has_vision=False, model_family="oga")

            config_path = os.path.join(model_path, "config.json")
            if not os.path.exists(config_path):
                # Fallback to loading config via transformers if local file missing
                config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            else:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_dict = json.load(f)
                # Create a temporary config object for easy attribute access if needed
                # or just parse the dict.
                
                archs = config_dict.get("architectures", [])
                model_type = config_dict.get("model_type", "")
                
                # Check for Qwen family
                if "Qwen3VLForConditionalGeneration" in archs or model_type == "qwen3_vl":
                    return ModelCapabilities(
                        has_vision=True,
                        image_token="<|vision_start|>", 
                        model_family="qwen"
                    )
                
                if "Qwen2" in str(archs) or "Qwen2ForCausalLM" in archs or model_type == "qwen2":
                    return ModelCapabilities(
                        has_vision=False,
                        model_family="qwen"
                    )
                
                # Check for LFM-VL
                if "Lfm2VlForConditionalGeneration" in archs or model_type == "lfm2_vl":
                    return ModelCapabilities(
                        has_vision=True,
                        image_token="<image>",
                        model_family="lfm"
                    )
                
                # Check for LFM (Text Only)
                if "Lfm2ForCausalLM" in archs or model_type == "lfm2":
                    return ModelCapabilities(
                        has_vision=False,
                        model_family="lfm"
                    )

            return ModelCapabilities(has_vision=False, model_family="generic")

        except Exception as e:
            print(f"[ModelFactory] Error detecting architecture: {e}")
            return ModelCapabilities(has_vision=False, model_family="unknown")

    # Class-level cache for CPU models (Legacy)
    _CPU_CACHE = {} 
    # V11: Class-level cache for OGA/DirectML models (Crucial for VRAM deduplication)
    _OGA_CACHE = {}

    @staticmethod
    def load_model_for_slot(
        slot_id: int, 
        model_path: str, 
        device: str = "cuda:0",
        use_compile: bool = False
    ) -> Tuple[Any, Any, ModelCapabilities]:
        """
        V8 Hardware Strategy:
        - NPU (flm): Hardlocked to qwen3vl-it:4b for Router/Supervisor.
        - GPU (DirectML): Savant experts load benchmarked OGA/DML models.
        """
        # --- 1. NPU ROUTING (FLM) ---
        if device == "flm":
            print(f"[ModelFactory] Slot {slot_id}: NPU Locked -> gpt-oss-sg:20b")
            from src.backends.flm_backend import FLMBackend, FLMConfig
            # NPU ONLY runs the Supervisor model in V8 architecture
            model_name = "gpt-oss-sg:20b"
            config = FLMConfig(model=model_name)
            backend = FLMBackend(config)
            return backend, None, ModelCapabilities(has_vision=True, model_family="flm")

        # --- 2. GPU SAVANT ROUTING (DirectML) ---
        if "-DML" in model_path or "-OGA" in model_path or "-AMD" in model_path:
            # V11 WEIGHT DEDUPLICATION (Crucial fix for 45GB spike)
            if model_path in ModelFactory._OGA_CACHE:
                print(f"[ModelFactory] Slot {slot_id}: OGA Cache Hit for {model_path}. Sharing VRAM.")
                backend = ModelFactory._OGA_CACHE[model_path]
            else:
                print(f"[ModelFactory] Slot {slot_id}: OGA Cache Miss. Loading {model_path} into VRAM...")
                from src.backends.oga_backend import OGABackend
                backend = OGABackend(model_path)
                ModelFactory._OGA_CACHE[model_path] = backend
            
            return backend, None, ModelCapabilities(has_vision=False, model_family="oga")

        # Standard Transformers Discovery
        caps = ModelFactory.detect_architecture(model_path)
        print(f"[ModelFactory] Slot {slot_id}: Loading {caps.model_family} Generic (Vision={caps.has_vision})...")

        
        # --- GPU PATH ---
        if caps.model_family == "qwen":
            processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        elif caps.model_family == "lfm":
            # LFM known issues with AutoProcessor
            from transformers import PreTrainedTokenizerFast
            processor = PreTrainedTokenizerFast.from_pretrained(model_path, trust_remote_code=True)
            if not hasattr(processor, "image_token"):
                processor.image_token = "<image>"
        else: # Generic
            try:
                processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            except Exception:
                from transformers import PreTrainedTokenizerFast
                processor = PreTrainedTokenizerFast.from_pretrained(model_path, trust_remote_code=True)

        # 2. Check Cache for Model
        import copy
        cached_model = ModelFactory._CPU_CACHE.get(model_path)
        
        if cached_model is None:
            print(f"[ModelFactory] Cache miss. Loading master copy from disk to RAM...")
            # Load to CPU first
            # CRITICAL: low_cpu_mem_usage=True prevents slow random initialization
            if caps.model_family == "qwen":
                if caps.has_vision:
                    cached_model = Qwen3VLForConditionalGeneration.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        torch_dtype=torch.bfloat16,
                        device_map="cpu",
                        low_cpu_mem_usage=True
                    )
                else:
                    from transformers import AutoModelForCausalLM
                    cached_model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        torch_dtype=torch.bfloat16,
                        device_map="cpu",
                        low_cpu_mem_usage=True
                    )
            else:
                from transformers import AutoModelForImageTextToText, AutoModelForCausalLM
                try:
                    # Try vision-text class first
                    cached_model = AutoModelForImageTextToText.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        torch_dtype=torch.bfloat16,
                        device_map="cpu",
                        low_cpu_mem_usage=True
                    )
                except Exception:
                     print(f"[ModelFactory] Failed to load as ImageTextToText, trying CausalLM...")
                     try:
                         # Fallback to pure causal LM (for text experts)
                         cached_model = AutoModelForCausalLM.from_pretrained(
                            model_path,
                            trust_remote_code=True,
                            torch_dtype=torch.bfloat16,
                            device_map="cpu",
                            low_cpu_mem_usage=True
                         )
                     except Exception as e2:
                        print(f"[ModelFactory] Failed to load as CausalLM ({e2}), falling back to base AutoModel...")
                        from transformers import AutoModel
                        cached_model = AutoModel.from_pretrained(
                            model_path,
                            trust_remote_code=True,
                            torch_dtype=torch.bfloat16,
                            device_map="cpu",
                            low_cpu_mem_usage=True
                        )
            ModelFactory._CPU_CACHE[model_path] = cached_model
            print(f"[ModelFactory] Master copy cached in RAM.")
        else:
            print(f"[ModelFactory] Cache hit! Cloning from RAM...")

        # 3. Create GPU Instance
        try:
            # print(f"[ModelFactory] Cloning model to {device}...")
            # Using state_dict copy is safer/cleaner than deepcopy for PyTorch modules if structure is complex
            # But deepcopy is the most robust generic way.
            model_copy = copy.deepcopy(cached_model)
            model = model_copy.to(device)
        except Exception as e:
            print(f"[ModelFactory] Cloning failed ({e}). Reverting to fresh load.")
            # Fallback
            try:
                from transformers import AutoModelForImageTextToText
                model = AutoModelForImageTextToText.from_pretrained(
                     model_path, 
                     trust_remote_code=True, 
                     torch_dtype=torch.bfloat16, 
                     device_map={"": device},
                     low_cpu_mem_usage=True
                )
            except ImportError:
                 # Fallback if specific class not found (unlikely in new transformers)
                 from transformers import AutoModel
                 model = AutoModel.from_pretrained(
                     model_path, 
                     trust_remote_code=True, 
                     torch_dtype=torch.bfloat16, 
                     device_map={"": device},
                     low_cpu_mem_usage=True
                 )

        # Optional Compilation
        if use_compile and hasattr(torch, "compile"):
            try:
                # model = torch.compile(model) # Default mode
                pass 
            except Exception as e:
                print(f"[Factory] Compile skipped: {e}")
                
        return model, processor, caps
