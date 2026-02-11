"""
Test Falcon H1R 7B router model.
"""
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
import sys

log_file = Path(__file__).parent / "test_falcon_output.txt"

with open(log_file, "w") as f:
    sys.stdout = f
    
    print("="*60)
    print("College of Experts - Falcon H1R 7B Router Test")
    print("="*60)

    MODELS_DIR = Path(__file__).parent / "models"
    ROUTER_MODEL = MODELS_DIR / "tiiuae_Falcon-H1R-7B"

    print(f"\nRouter model path: {ROUTER_MODEL}")
    print(f"Exists: {ROUTER_MODEL.exists()}")
    
    # List contents
    if ROUTER_MODEL.exists():
        print(f"Contents: {[f.name for f in ROUTER_MODEL.iterdir()][:10]}")

    from src.backends import TransformersBackend, GenerationConfig

    print("\n1. Creating backend...")
    backend = TransformersBackend(device="auto")
    device = backend._detect_best_device()
    print(f"   Device: {device}")

    print(f"\n2. Loading Falcon H1R 7B router model...")
    print(f"   (This is a 14GB model, may take a moment...)")
    try:
        info = backend.load_model(
            model_id="router",
            model_path=str(ROUTER_MODEL)
        )
        print(f"   Status: Loaded")
        print(f"   Model ID: {info.model_id}")
        print(f"   Memory: {backend.get_memory_usage()}")
        
        print(f"\n3. Testing basic generation...")
        response = backend.generate(
            model_id="router",
            messages=[{"role": "user", "content": "Hello! Say 'Router ready' in one sentence."}],
            config=GenerationConfig(max_tokens=30, temperature=0.3)
        )
        print(f"   Response: '{response}'")
        
        print(f"\n4. Cleanup...")
        backend.unload_model("router")
        print(f"   Model unloaded")
        
        print("\n" + "="*60)
        print("FALCON ROUTER TEST PASSED")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "="*60)
        print("TEST FAILED")
        print("="*60)

sys.stdout = sys.__stdout__
print(f"Output written to: {log_file}")
print("\n--- Test Output ---")
print(open(log_file).read())
