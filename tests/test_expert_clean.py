"""
Test LFM2.5 expert model with output to file.
"""
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
import sys

# Redirect output to file for clean capture
log_file = Path(__file__).parent / "test_output.txt"
original_stdout = sys.stdout

with open(log_file, "w") as f:
    sys.stdout = f
    
    print("="*60)
    print("College of Experts - LFM2.5 Expert Model Test")
    print("="*60)

    MODELS_DIR = Path(__file__).parent / "models"
    EXPERT_MODEL = MODELS_DIR / "LiquidAI_LFM2.5-1.2B-Instruct"

    print(f"\nExpert model path: {EXPERT_MODEL}")
    print(f"Exists: {EXPERT_MODEL.exists()}")

    from src.backends import TransformersBackend, GenerationConfig

    print("\n1. Creating backend...")
    backend = TransformersBackend(device="auto")
    device = backend._detect_best_device()
    print(f"   Device: {device}")

    print(f"\n2. Loading LFM2.5 expert model...")
    try:
        info = backend.load_model(
            model_id="expert_base",
            model_path=str(EXPERT_MODEL)
        )
        print(f"   Status: Loaded")
        print(f"   Model ID: {info.model_id}")
        print(f"   Memory: {backend.get_memory_usage()}")
        
        print(f"\n3. Testing basic generation...")
        response = backend.generate(
            model_id="expert_base",
            messages=[{"role": "user", "content": "What is 2+2? Answer in one word."}],
            config=GenerationConfig(max_tokens=20, temperature=0.3)
        )
        print(f"   Response: '{response}'")
        
        print(f"\n4. Testing with Python expert persona...")
        python_prompt = "You are a Python expert. Be concise."
        response2 = backend.generate(
            model_id="expert_base",
            messages=[{"role": "user", "content": "What is a list comprehension in Python?"}],
            system_prompt=python_prompt,
            config=GenerationConfig(max_tokens=150, temperature=0.7)
        )
        print(f"   Response:\n{response2}")
        
        print(f"\n5. Cleanup...")
        backend.unload_model("expert_base")
        print(f"   Model unloaded")
        
        print("\n" + "="*60)
        print("TEST PASSED")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\n" + "="*60)
        print("TEST FAILED")
        print("="*60)

sys.stdout = original_stdout
print(f"Output written to: {log_file}")
print("\n--- Test Output ---")
print(open(log_file).read())
