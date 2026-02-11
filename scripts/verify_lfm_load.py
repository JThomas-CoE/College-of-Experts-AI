
import os
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast

MODEL_PATH = r"c:\RyzenAI\college of experts\models\LiquidAI_LFM2.5-1.6B-VL"

print(f"--- LFM Verification (Low-Level): {MODEL_PATH} ---")

print("1. Loading Tokenizer...")
try:
    # Try generic fast tokenizer directly
    tokenizer = PreTrainedTokenizerFast.from_pretrained(MODEL_PATH)
    # Monkey-patch image_token if missing, just to bypass the processor check if needed
    if not hasattr(tokenizer, "image_token"):
        tokenizer.image_token = "<image>"
    print("   Tokenizer Loaded.")
    print(f"   Class: {type(tokenizer)}")
except Exception as e:
    print(f"   Tokenizer Load Failed: {e}")
    exit(1)

print("2. Loading Model...")
try:
    model = AutoModel.from_pretrained(
        MODEL_PATH, 
        trust_remote_code=True, 
        torch_dtype=torch.float16,
        device_map={"": "cuda:0"}
    )
    print("   Model Loaded Successfully!")
    print(f"   Model Class: {type(model)}")
except Exception as e:
    print(f"   Model Load Failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
    
print("\n--- Verification Complete: SUCCESS ---")
