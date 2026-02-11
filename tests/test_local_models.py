"""
Test loading the downloaded models from local directory.
"""
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path

print("="*60)
print("College of Experts - Local Model Test")
print("="*60)

# Local model paths
MODELS_DIR = Path(__file__).parent / "models"
EXPERT_MODEL = MODELS_DIR / "LiquidAI_LFM2.5-1.2B-Instruct"
ROUTER_MODEL = MODELS_DIR / "tiiuae_Falcon-H1R-7B"

print(f"\nModels directory: {MODELS_DIR}")
print(f"Expert model: {EXPERT_MODEL.exists()} - {EXPERT_MODEL}")
print(f"Router model: {ROUTER_MODEL.exists()} - {ROUTER_MODEL}")

from src.backends import TransformersBackend, GenerationConfig

# Initialize backend
print("\n1. Creating backend...")
backend = TransformersBackend(device="auto")
device = backend._detect_best_device()
print(f"   Device: {device}")

# Test LFM2.5 (expert model - smaller, faster to test)
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
        messages=[{"role": "user", "content": "Hello! Say 'Expert model ready' in one sentence."}],
        config=GenerationConfig(max_tokens=30, temperature=0.7)
    )
    print(f"   Response: {response}")
    
    # Unload
    backend.unload_model("expert_base")
    print(f"   ✓ Expert model unloaded")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test Falcon (router model - larger)
print(f"\n4. Loading Falcon router model...")
try:
    info = backend.load_model(
        model_id="router",
        model_path=str(ROUTER_MODEL)
    )
    print(f"   ✓ Loaded: {info.model_id}")
    print(f"   Memory: {backend.get_memory_usage()}")
    
    # Quick generation test
    print(f"\n5. Testing router generation...")
    response = backend.generate(
        model_id="router",
        messages=[{"role": "user", "content": "Hello! Say 'Router model ready' in one sentence."}],
        config=GenerationConfig(max_tokens=30, temperature=0.7)
    )
    print(f"   Response: {response}")
    
    # Unload
    backend.unload_model("router")
    print(f"   ✓ Router model unloaded")
    
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("LOCAL MODEL TEST COMPLETE")
print("="*60)
