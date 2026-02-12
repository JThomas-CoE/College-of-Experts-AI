"""
Download models for College of Experts (v4).

Fetches the pre-quantized (INT4) models from the JThomas-CoE Hugging Face collection
and places them in the ./models directory structure expected by the demo.
"""

import argparse
import os
import shutil
from pathlib import Path
from huggingface_hub import snapshot_download

# Map friendly names to HF Repo IDs and local folder names
MODELS = {
    "supervisor": {
        "repo": "JThomas-CoE/DeepSeek-R1-Distill-Qwen-7B-INT4-DML",
        "local": "DeepSeek-R1-Distill-Qwen-7B-DML",
        "desc": "Router & Supervisor (5GB)"
    },
    "code": {
        "repo": "JThomas-CoE/Qwen2.5-Coder-7B-INT4-DML",
        "local": "Qwen2.5-Coder-7B-DML",
        "desc": "Code Specialist (5GB)"
    },
    "math": {
        "repo": "JThomas-CoE/Qwen2.5-Math-7B-INT4-DML",
        "local": "Qwen2.5-Math-7B-DML",
        "desc": "Math Specialist (4GB)"
    },
    "medical": {
        "repo": "JThomas-CoE/BioMistral-7B-INT4-DML",
        "local": "BioMistral-7B-DML",
        "desc": "Medical Specialist (5GB)"
    },
    "sql": {
        "repo": "JThomas-CoE/sqlcoder-7b-2-INT4-DML",
        "local": "sqlcoder-7b-2-DML",
        "desc": "SQL Specialist (5GB)"
    },
    "legal": {
        "repo": "JThomas-CoE/law-LLM-INT4-DML",
        "local": "law-LLM-DML",
        "desc": "Legal Specialist (4GB)"
    }
}

DEFAULT_MODELS_DIR = Path("models").absolute()

def download_model(key: str, repo_id: str, folder_name: str, base_dir: Path):
    """Download a single model."""
    target_dir = base_dir / folder_name
    
    print(f"\n[{key.upper()}] Downloading {repo_id}...")
    print(f" -> Target: {target_dir}")
    
    try:
        # Download directly to the target folder
        path = snapshot_download(
            repo_id=repo_id,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,  # We want actual files for portability
            resume_download=True
        )
        print(f"✓ Successfully downloaded {key}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {key}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download CoE models")
    parser.add_argument("--dir", type=str, default=str(DEFAULT_MODELS_DIR), 
                        help="Target directory (default: ./models)")
    parser.add_argument("--only", type=str, choices=MODELS.keys(), 
                        help="Download only a specific model")
    args = parser.parse_args()

    base_dir = Path(args.dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"College of Experts - Model Downloader")
    print(f"Target Directory: {base_dir}\n")

    selection = MODELS.items()
    if args.only:
        selection = [(args.only, MODELS[args.only])]

    print("Models to download:")
    for key, info in selection:
        p = base_dir / info['local']
        status = " (Exists)" if p.exists() else ""
        print(f" - {info['desc']}{status}")

    if not args.only:
        confirm = input("\nStart download? (y/n): ")
        if confirm.lower() != 'y':
            print("Cancelled.")
            return

    for key, info in selection:
        download_model(key, info['repo'], info['local'], base_dir)

    print("\n✓ Process complete.")

if __name__ == "__main__":
    main()
