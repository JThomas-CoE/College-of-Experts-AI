
from huggingface_hub import snapshot_download
import os

model_id = "LiquidAI/LFM2.5-VL-1.6B"
local_dir = r"c:\RyzenAI\college of experts\models\LiquidAI_LFM2.5-1.6B-VL"

print(f"Starting download of {model_id} to {local_dir}...")
try:
    snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,  # Windows often prefers actual files
        resume_download=True
    )
    print("Download complete successfully.")
except Exception as e:
    print(f"Download failed: {e}")
