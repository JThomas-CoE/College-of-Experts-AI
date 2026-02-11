"""
Test loading just the LFM2.5 expert model.
"""
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path

print("="*60)
print("College of Experts - LFM2.5 Expert Model Test")
print("="*60)

# Local model path
MODELS_DIR = Path(__file__).parent / "models"
EXPERT_MODEL = MODELS_DIR / "LiquidAI_LFM2.5-1.2B-Instruct"

print(f"\nExpert model path: {EXPERT_MODEL}")
print(f"Exists: {EXPERT_MODEL.exists()}")

from src.backends import TransformersBackend, GenerationConfig

# Initialize backend
print("\n1. Creating backend...")
backend = TransformersBackend(device="auto")
device = backend._detect_best_device()
print(f"   Device: {device}")

# Test LFM2.5
print(f"\n2. Loading LFM2.5 expert model...")
try:
    info = backend.load_model(
        model_id="expert_base",
        model_path=str(EXPERT_MODEL)
    )
    print(f"   ✓ Loaded: {info.model_id}")
    print(f"   Memory: {backend.get_memory_usage()}")
    
    # Quick generation test
    print(f"\n3. Testing generation...")
    response = backend.generate(
        model_id="expert_base",
        messages=[{"role": "user", "content": "What is 2+2? Answer in one word."}],
        config=GenerationConfig(max_tokens=20, temperature=0.3)
    )
    print(f"   Response: {response}")
    
    # Test with expert persona
    print(f"\n4. Testing with Python expert persona...")
    python_prompt = "You are a Python expert. Be concise and technical."
    response2 = backend.generate(
        model_id="expert_base",
        messages=[{"role": "user", "content": "What is a list comprehension?"}],
        system_prompt=python_prompt,
        config=GenerationConfig(max_tokens=100, temperature=0.7)
    )
    print(f"   Response: {response2[:200]}...")
    
    # Unload
    print(f"\n5. Cleanup...")
    backend.unload_model("expert_base")
    print(f"   ✓ Model unloaded")
    
    print("\n" + "="*60)
    print("LFM2.5 EXPERT MODEL TEST PASSED")
    print("="*60)
    
except Exception as e:
    print(f"\n   ✗ Error: {e}")
    import traceback
    traceback.print_exc()
    print("\n" + "="*60)
    print("TEST FAILED")
    print("="*60)
