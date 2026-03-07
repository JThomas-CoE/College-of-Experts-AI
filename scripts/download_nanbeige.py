import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


DEFAULT_MODEL_ID = "JThomas-CoE/Nanbeige4.1-3B-ONNX-INT4"
DEFAULT_TARGET_DIR = Path("models/Nanbeige4.1-3B-ONNX-INT4").absolute()


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the Nanbeige supervisor model for the stable root demo")
    parser.add_argument(
        "--repo",
        default=DEFAULT_MODEL_ID,
        help=f"Hugging Face repo ID to download (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--dir",
        default=str(DEFAULT_TARGET_DIR),
        help=f"Target directory (default: {DEFAULT_TARGET_DIR})",
    )
    args = parser.parse_args()

    target_dir = Path(args.dir).absolute()
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {args.repo} ...")
    print(f"Target: {target_dir}")
    print("Expected local folder name for demo.py: Nanbeige4.1-3B-ONNX-INT4")

    try:
        path = snapshot_download(
            repo_id=args.repo,
            local_dir=str(target_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"\n✓ Download complete: {path}")
    except Exception as e:
        print(f"\n✗ Download failed: {e}")
        print("If the model is not public yet, publish it to the Hugging Face hub first or pass --repo with the correct repo ID.")

if __name__ == "__main__":
    main()
