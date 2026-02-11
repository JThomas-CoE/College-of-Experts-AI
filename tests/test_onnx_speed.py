import onnxruntime as ort
import numpy as np
import time
import os
from transformers import AutoTokenizer

def test_onnx_speed():
    model_dir = "models/LiquidAI_LFM2.5-1.6B-VL-ONNX"
    # The 4-bit quantized model
    onnx_path = os.path.join(model_dir, "onnx", "decoder_q4.onnx")
    
    if not os.path.exists(onnx_path):
        print(f"Error: {onnx_path} not found")
        return

    print(f"Loading 4-bit ONNX model with DirectML: {onnx_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    
    sess_options = ort.SessionOptions()
    providers = [('DmlExecutionProvider', {'device_id': 0}), 'CPUExecutionProvider']
    
    try:
        session = ort.InferenceSession(onnx_path, sess_options, providers=providers)
        print(f"Using Provider: {session.get_providers()[0]}")
    except Exception as e:
        print(f"Error loading ONNX: {e}")
        return

    # LiquidAI LFM input names
    prompt = "Write a short poem about AI:"
    inputs = tokenizer(prompt, return_tensors="np")
    
    # We need to map inputs to what the ONNX model expects
    # In decoder-only ONNX, it usually expects input_ids and position_ids
    print(f"Inputs expected by model: {[i.name for i in session.get_inputs()]}")
    
    # Simple forward pass test
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # Handle optional inputs
    feed = {"input_ids": input_ids}
    if "attention_mask" in [i.name for i in session.get_inputs()]:
        feed["attention_mask"] = attention_mask

    print("\nRunning speed test (generation simulation)...")
    start = time.time()
    # Simulate generating 50 tokens sequentially
    # Real generation involves KV cache management, but this tests raw throughput
    for _ in range(50):
        _ = session.run(None, feed)
    elapsed = time.time() - start
    
    print(f"Throughput (simulated): {50/elapsed:.2f} tokens/s")

if __name__ == "__main__":
    test_onnx_speed()
