
import os
import argparse
import subprocess
from huggingface_hub import snapshot_download

def setup_deepseek_reasoner():
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
    raw_model_dir = os.path.join("models", "DeepSeek-R1-Distill-Qwen-7B")
    quant_model_dir = os.path.join("models", "DeepSeek-R1-Distill-Qwen-7B-DML")
    
    # 1. Download from HuggingFace
    if not os.path.exists(raw_model_dir):
        print(f"[1/2] Downloading {model_id}...")
        snapshot_download(repo_id=model_id, local_dir=raw_model_dir)
    else:
        print(f"[1/2] Model already downloaded at {raw_model_dir}")
        
    # 2. Convert to ONNX 4-bit
    print(f"[2/2] Converting to ONNX (Int4 DML)... This will take a while.")
    
    # We reuse the existing build_oga_model logic but call it directly
    cmd = [
        "python", os.path.join(os.path.dirname(__file__), "build_oga_model.py"),
        raw_model_dir,
        quant_model_dir,
        "--ep", "dml"
    ]
    
    subprocess.run(cmd, check=True)
    print(f"\nSUCCESS. Model ready at: {quant_model_dir}")

if __name__ == "__main__":
    setup_deepseek_reasoner()
