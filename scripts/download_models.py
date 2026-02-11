"""
Download models for College of Experts.

Saves models to a local folder for easy management.
At 100 Mbit/s, expect ~80 seconds per GB.

Models:
- Qwen/Qwen2.5-0.5B-Instruct   (~1 GB)   - Small test model
- LiquidAI/LFM2.5-1.2B-Instruct (~2.5 GB) - Faux expert base  
- tiiuae/Falcon-H1R-7B         (~14 GB)  - Router/HRM (optional)
"""

import argparse
import os
from pathlib import Path

# Default local models directory
DEFAULT_MODELS_DIR = Path(__file__).parent / "models"


def download_model(model_id: str, local_dir: Path):
    """Download a model from HuggingFace to local directory."""
    from huggingface_hub import snapshot_download
    
    # Create model-specific subdirectory
    model_name = model_id.replace("/", "_")
    model_path = local_dir / model_name
    
    print(f"\n{'='*60}")
    print(f"Downloading: {model_id}")
    print(f"Destination: {model_path}")
    print(f"{'='*60}")
    
    try:
        path = snapshot_download(
            repo_id=model_id,
            local_dir=str(model_path),
            local_dir_use_symlinks=False,  # Actual files, not symlinks
            resume_download=True,
        )
        print(f"✓ Downloaded to: {path}")
        return path
    except Exception as e:
        print(f"✗ Failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Download models for College of Experts")
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument("--test", action="store_true", help="Download small test model only")
    parser.add_argument("--expert", action="store_true", help="Download LFM2.5 expert model")
    parser.add_argument("--router", action="store_true", help="Download Falcon router model")
    parser.add_argument("--dir", type=str, default=str(DEFAULT_MODELS_DIR),
                        help=f"Models directory (default: {DEFAULT_MODELS_DIR})")
    
    args = parser.parse_args()
    
    # Default: download router and expert models
    if not any([args.all, args.test, args.expert, args.router]):
        args.expert = True
        args.router = True
    
    models_to_download = []
    
    if args.test or args.all:
        models_to_download.append(("Qwen/Qwen2.5-0.5B-Instruct", 1.0))
    
    if args.expert or args.all:
        models_to_download.append(("LiquidAI/LFM2.5-1.2B-Instruct", 2.5))
    
    if args.router or args.all:
        models_to_download.append(("tiiuae/Falcon-H1R-7B", 14.0))
    
    # Setup directory
    local_dir = Path(args.dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("College of Experts - Model Downloader")
    print("="*60)
    print(f"\nModels directory: {local_dir.absolute()}")
    print("\nModels to download:")
    for model, size in models_to_download:
        print(f"  • {model} (~{size:.1f} GB)")
    
    total_size = sum(size for _, size in models_to_download)
    estimated_time = total_size * 80  # ~80 seconds per GB at 100 Mbit/s
    print(f"\nEstimated total: ~{total_size:.1f} GB")
    print(f"Estimated time at 100 Mbit/s: ~{estimated_time/60:.1f} minutes")
    
    input("\nPress Enter to start downloads (Ctrl+C to cancel)...")
    
    print("\nStarting downloads...")
    
    results = {}
    for model, size in models_to_download:
        path = download_model(model, local_dir)
        results[model] = path
    
    # Summary
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    for model, path in results.items():
        status = "✓ OK" if path else "✗ FAILED"
        print(f"  {status}: {model}")
    
    print(f"\nModels saved to: {local_dir.absolute()}")
    
    # Create a config file listing the models
    config_path = local_dir / "models.txt"
    with open(config_path, "w") as f:
        f.write("# Downloaded models for College of Experts\n")
        for model, path in results.items():
            if path:
                model_name = model.replace("/", "_")
                f.write(f"{model}={local_dir / model_name}\n")
    print(f"Model paths saved to: {config_path}")
    
    if all(results.values()):
        print("\n✓ All models downloaded successfully!")
    else:
        print("\n⚠ Some downloads failed. Run again to retry.")


if __name__ == "__main__":
    main()
