import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_bf16_speed():
    model_path = "models/Qwen2.5-Coder-7B-Instruct"
    print(f"Loading {model_path} in BF16...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Warmup
    print("Warmup run...")
    prompt = "def hello():"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=10)
    
    print("\nStarting timed inference...")
    # Longer generation to amortize any overhead
    start_inf = time.time()
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            use_cache=True
        )
    elapsed = time.time() - start_inf
    
    tokens = len(outputs[0]) - len(inputs[0])
    print("-" * 40)
    print(f"Inference Time: {elapsed:.2f}s")
    print(f"Tokens generated: {tokens}")
    print(f"Speed: {tokens/elapsed:.2f} tok/s")
    print("-" * 40)
    print(f"Theoretical (200GB/s / 15GB): {200/15:.2f} tok/s")
    print("-" * 40)

if __name__ == "__main__":
    test_bf16_speed()
