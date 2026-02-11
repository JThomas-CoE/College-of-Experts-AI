"""Test FLM Backend."""
import sys
sys.path.insert(0, ".")

from src.backends.flm_backend import FLMBackend, FLMConfig

print("=" * 50)
print("FLM Backend Test")
print("=" * 50)

# Create backend
config = FLMConfig(model="qwen3vl-it:4b")
backend = FLMBackend(config)

# Test simple generation
print("\n[TEST 1] Simple prompt...")
response = backend.generate(prompt="What is the capital of France?", max_tokens=50)
print(f"Response: {response}")

# Test chat interface
print("\n[TEST 2] Chat interface...")
response = backend.chat(
    system_prompt="You are a helpful assistant.",
    user_prompt="Explain quantum computing in one sentence.",
    temperature=0.7,
    max_tokens=100
)
print(f"Response: {response}")

# Test with different temperatures (simulating council)
print("\n[TEST 3] Temperature variation (council simulation)...")
for temp in [0.3, 0.5, 0.7]:
    response = backend.generate(
        prompt="Name a random color.",
        temperature=temp,
        max_tokens=10
    )
    print(f"  T={temp}: {response}")

print("\n" + "=" * 50)
print("[DONE] FLM Backend tests complete")
