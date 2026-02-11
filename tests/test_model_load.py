"""Minimal test: Load a small model and generate."""
import sys
import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("="*50)
print("College of Experts - Model Load Test")
print("="*50)

from src.backends import TransformersBackend, GenerationConfig

# Initialize backend
print("\n1. Creating backend...")
backend = TransformersBackend(device="auto")
device = backend._detect_best_device()
print(f"   Device: {device}")

# Use a small model for testing
TEST_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

print(f"\n2. Loading model: {TEST_MODEL}")

try:
    info = backend.load_model(
        model_id="test_model",
        model_path=TEST_MODEL
    )
    print(f"   Status: Loaded successfully")
    mem = backend.get_memory_usage()
    print(f"   Memory: {mem}")
    
    # Try generation
    print(f"\n3. Generating test response...")
    response = backend.generate(
        model_id="test_model",
        messages=[{"role": "user", "content": "Say 'Hello, College of Experts!' in one short sentence."}],
        config=GenerationConfig(max_tokens=50, temperature=0.7)
    )
    print(f"   Response: {response}")
    
    # Cleanup
    print(f"\n4. Cleanup...")
    backend.unload_model("test_model")
    print(f"   Model unloaded")
    
    print("\n" + "="*50)
    print("TEST PASSED")
    print("="*50)
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    print("\n" + "="*50)
    print("TEST FAILED")
    print("="*50)

