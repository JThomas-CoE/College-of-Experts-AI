"""
Quantize FP16 models to NF4 and save to disk.
This reduces storage from ~15GB to ~5GB per 7B model.

Usage:
    python quantize_and_save.py models/Qwen2.5-Coder-7B-Instruct
"""

import sys
import os
import torch
import shutil
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def quantize_and_save(model_path: str, output_suffix: str = "-NF4"):
    """Load model, quantize to NF4, and save."""
    
    model_path = Path(model_path)
    if not model_path.exists():
        print(f"Error: {model_path} does not exist")
        return False
    
    output_path = Path(str(model_path) + output_suffix)
    
    print(f"=" * 60)
    print(f"Quantizing: {model_path.name}")
    print(f"Output: {output_path.name}")
    print(f"=" * 60)
    
    # Configure NF4 quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,  # Extra compression
    )
    
    print("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    print("[2/4] Loading and quantizing model (this takes 20-40 sec)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    print("[3/4] Saving quantized model...")
    # Save the quantized model
    model.save_pretrained(output_path, safe_serialization=True)
    tokenizer.save_pretrained(output_path)
    
    # Copy config files
    for config_file in ["config.json", "generation_config.json", "tokenizer_config.json"]:
        src = model_path / config_file
        if src.exists():
            shutil.copy(src, output_path / config_file)
    
    print("[4/4] Validating...")
    # Quick validation - try to load it back
    try:
        test_model = AutoModelForCausalLM.from_pretrained(
            output_path,
            device_map="auto",
            trust_remote_code=True,
        )
        del test_model
        torch.cuda.empty_cache()
        print(f"\n✓ SUCCESS! Quantized model saved to: {output_path}")
        
        # Report sizes
        orig_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
        new_size = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
        print(f"  Original: {orig_size / 1e9:.2f} GB")
        print(f"  Quantized: {new_size / 1e9:.2f} GB")
        print(f"  Savings: {(orig_size - new_size) / 1e9:.2f} GB ({100 * (1 - new_size/orig_size):.1f}%)")
        
        return True
    except Exception as e:
        print(f"\n✗ Validation failed: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python quantize_and_save.py <model_path>")
        print("\nAvailable models:")
        models_dir = Path("models")
        for d in models_dir.iterdir():
            if d.is_dir() and not d.name.endswith("-NF4"):
                print(f"  {d}")
        sys.exit(1)
    
    model_path = sys.argv[1]
    success = quantize_and_save(model_path)
    sys.exit(0 if success else 1)
