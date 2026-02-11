"""Quick test for NPU Backend loading."""
import sys
sys.path.insert(0, ".")

from src.backends.npu_backend import NPUBackend

model_path = "models/LiquidAI_LFM2.5-1.6B-VL-ONNX"

print("=" * 50)
print("NPU Backend Loading Test")
print("=" * 50)

try:
    backend = NPUBackend(model_path)
    print("\n[SUCCESS] NPU Backend loaded!")
    print(f"  Processor: {type(backend.processor).__name__}")
    print(f"  Tokenizer vocab size: {len(backend.tokenizer)}")
    
    # Quick generation test
    print("\n[TEST] Quick generation test...")
    test_input = backend.tokenizer.encode("Hello, I am", return_tensors="np")
    result = backend.generate(input_ids=test_input, max_new_tokens=15, temperature=0.7, repetition_penalty=1.2)
    decoded = backend.tokenizer.decode(result[0], skip_special_tokens=True)
    print(f"  Input: 'Hello, I am'")
    print(f"  Output: '{decoded}'")
    
    print("\n[SUCCESS] All tests passed!")
    
except Exception as e:
    print(f"\n[ERROR] {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
