import onnxruntime_genai as og

print("Checking OGA Device Support...")
print(f"GenAI version: {og.__version__ if hasattr(og, '__version__') else 'unknown'}")

try:
    # This is a bit of a hack to see what devices are available
    # og.Model doesn't have a simple 'is_supported' but we can check the error when loading a dummy
    print("Testing DML support...")
    # We can't easily check without a model, but we can look for the library
    import onnxruntime as ort
    print(f"ONNX Runtime Providers: {ort.get_available_providers()}")
except Exception as e:
    print(f"Error checking: {e}")
