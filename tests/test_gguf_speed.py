import time
from llama_cpp import Llama, llama_cpp
import torch

def test_speed():
    model_path = "models/sqlcoder-7b-2/sqlcoder-7b-q5_k_m.gguf"
    
    # Print system info to check for HIP/BLAS
    try:
        sys_info = llama_cpp.llama_print_system_info().decode('utf-8')
        print(f"System Info: {sys_info}")
    except:
        print("Could not get system info")

    print(f"Loading {model_path}...")
    
    # n_gpu_layers=35 to offload everything to GPU if possible
    m = Llama(model_path=model_path, n_ctx=2048, n_gpu_layers=35)
    
    prompt = "Write a SQL query to find all customers who ordered more than 5 times:"
    print(f"\nPrompt: {prompt}\n")
    
    # Warmup
    _ = m(prompt, max_tokens=10)
    
    print("Timed run...")
    start = time.time()
    output = m(prompt, max_tokens=100, temperature=0.3)
    elapsed = time.time() - start
    
    response = output["choices"][0]["text"]
    tokens = output["usage"]["completion_tokens"]
    tps = tokens / elapsed if elapsed > 0 else 0
    
    print("-" * 40)
    print(f"Response: {response}")
    print("-" * 40)
    print(f"Total Time: {elapsed:.2f}s")
    print(f"Tokens: {tokens}")
    print(f"Throughput: {tps:.2f} tok/s")
    print("-" * 40)
    print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

if __name__ == "__main__":
    test_speed()
