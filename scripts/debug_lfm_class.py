
import torch
from transformers import AutoModelForVision2Seq, AutoModelForCausalLM, AutoConfig

MODEL_PATH = r"c:\RyzenAI\college of experts\models\LiquidAI_LFM2.5-1.6B-VL"

print(f"Diagnostics for: {MODEL_PATH}")

def test_load(param_class, name):
    print(f"\n--- Testing {name} ---")
    try:
        model = param_class.from_pretrained(
            MODEL_PATH, 
            trust_remote_code=True, 
            torch_dtype=torch.float16, 
            device_map="cpu", 
            low_cpu_mem_usage=True
        )
        print(f"SUCCESS: Loaded {type(model)}")
        if hasattr(model, "generate"):
            print("VERIFIED: Model has .generate() method.")
            return True
        else:
            print("FAILURE: Model loaded but MISSING .generate() method.")
            return False
    except Exception as e:
        print(f"CRASH: {e}")
        return False

# Test 1: Vision2Seq (Standard for standard VLMs like Llava)
if test_load(AutoModelForVision2Seq, "AutoModelForVision2Seq"):
    exit(0)

# Test 2: CausalLM (Standard for pure LLMs or some VLMs like Qwen)
if test_load(AutoModelForCausalLM, "AutoModelForCausalLM"):
    exit(0)

print("\nALL FAILED.")
