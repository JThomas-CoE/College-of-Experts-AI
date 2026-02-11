"""
Test Nemotron Nano 30B via Ollama backend.

This is a safe test that won't crash like Transformers backend with hybrid models.
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 60)
print("College of Experts - Nemotron Nano 30B via Ollama Test")
print("=" * 60)

# Test 1: Direct Ollama backend test
print("\n1. Testing OllamaBackend directly...")
from src.backends import BackendType
from src.backends.ollama_backend import OllamaBackend
from src.backends.base import GenerationConfig

backend = OllamaBackend(auto_pull=False)

if not backend.is_ollama_running():
    print("   ERROR: Ollama is not running!")
    print("   Start with: ollama serve")
    sys.exit(1)

print("   Ollama is running ✓")

# List available models
models = backend.list_ollama_models()
print(f"   Available models: {len(models)}")
for model in models:
    if "nemotron" in model.lower():
        print(f"   ✓ {model}")

# Load Nemotron as router
print("\n2. Loading Nemotron Nano 30B as 'router'...")
try:
    info = backend.load_model(
        model_id="router",
        model_path="nemotron-3-nano:30b"
    )
    print(f"   Model ID: {info.model_id}")
    print(f"   Backend: {info.backend_type.value}")
    print(f"   Loaded: {info.is_loaded}")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

# Test generation
print("\n3. Testing generation...")
try:
    response = backend.generate(
        model_id="router",
        messages=[{"role": "user", "content": "Say 'Router ready!' in one sentence."}],
        config=GenerationConfig(max_tokens=50, temperature=0.3)
    )
    print(f"   Response: '{response}'")
except Exception as e:
    print(f"   ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test with system prompt (expert persona simulation)
print("\n4. Testing with system prompt (expert persona)...")
try:
    response = backend.generate(
        model_id="router",
        messages=[{"role": "user", "content": "What is a Python decorator?"}],
        system_prompt="You are a Python expert. Be concise and helpful.",
        config=GenerationConfig(max_tokens=150, temperature=0.5)
    )
    print(f"   Response (first 200 chars):\n   {response[:200]}...")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

# Cleanup
print("\n5. Cleanup...")
backend.unload_model("router")
print("   Model unlinked ✓")

print("\n" + "=" * 60)
print("NEMOTRON OLLAMA TEST PASSED ✓")
print("=" * 60)
print("\nYou can now run the demo with:")
print("  python demo.py --backend ollama --router-model nemotron-3-nano:30b")
