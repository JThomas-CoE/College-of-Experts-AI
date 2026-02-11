"""Longer NPU Backend test for Task Manager verification."""
import sys
import time
sys.path.insert(0, ".")

from src.backends.npu_backend import NPUBackend

model_path = "models/LiquidAI_LFM2.5-1.6B-VL-ONNX"

print("=" * 50)
print("NPU Backend Extended Test")
print("=" * 50)

backend = NPUBackend(model_path)
print("\n[SUCCESS] NPU Backend loaded!")

# Long generation test
print("\n[TEST] Extended generation (200 tokens)...")
print("       Watch Task Manager for NPU/GPU activity!")
print("-" * 50)

prompts = [
    "Explain the theory of relativity in simple terms:",
    "Write a short poem about artificial intelligence:",
    "What are the benefits of renewable energy?"
]

for i, prompt in enumerate(prompts):
    print(f"\n[{i+1}/3] Prompt: '{prompt[:50]}...'")
    start = time.time()
    
    test_input = backend.tokenizer.encode(prompt, return_tensors="np")
    result = backend.generate(
        input_ids=test_input, 
        max_new_tokens=150, 
        temperature=0.7, 
        repetition_penalty=1.2
    )
    
    elapsed = time.time() - start
    decoded = backend.tokenizer.decode(result[0], skip_special_tokens=True)
    tokens_generated = len(result[0]) - len(test_input[0])
    
    print(f"Output ({tokens_generated} tokens in {elapsed:.1f}s = {tokens_generated/elapsed:.1f} tok/s):")
    print(decoded[:300] + "..." if len(decoded) > 300 else decoded)

print("\n" + "=" * 50)
print("[DONE] Extended test complete")
