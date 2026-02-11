"""
Download Qwen3-VL-4B-Instruct model from HuggingFace to local models directory.
"""
import os
from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
LOCAL_DIR = Path("models/Qwen3-VL-4B-Instruct")

print(f"Downloading {MODEL_ID} to {LOCAL_DIR}")
print("This will download approximately 8GB...")

# Create directory if needed
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

# Download the model
snapshot_download(
    repo_id=MODEL_ID,
    local_dir=LOCAL_DIR,
    local_dir_use_symlinks=False,
    resume_download=True
)

print(f"\n✓ Download complete: {LOCAL_DIR}")
