from huggingface_hub import HfApi, snapshot_download
import os

def download_model():
    api = HfApi()
    print("Searching for LFM2.5-VL-1.6B-ONNX...")
    
    # Try probable IDs directly first to save time
    candidates = [
        "liquid-ai/LFM2.5-VL-1.6B-ONNX",
        "LiquidAI/LFM2.5-VL-1.6B-ONNX"
    ]
    
    target_model = None
    
    # Check candidates
    for cand in candidates:
        try:
            api.model_info(cand)
            target_model = cand
            break
        except Exception:
            continue
            
    if not target_model:
        # Fallback to search
        models = api.list_models(search="LFM2.5-VL-1.6B-ONNX")
        for m in models:
            if "LFM2.5" in m.modelId and "ONNX" in m.modelId and "1.6B" in m.modelId:
                target_model = m.modelId
                break
    
    if target_model:
        print(f"Found model: {target_model}")
        output_dir = r"C:\RyzenAI\college of experts\models\LiquidAI_LFM2.5-1.6B-VL-ONNX"
        print(f"Downloading to {output_dir}...")
        
        snapshot_download(
            repo_id=target_model, 
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        print("Download complete.")
    else:
        print("Model not found on Hugging Face.")

if __name__ == "__main__":
    download_model()
