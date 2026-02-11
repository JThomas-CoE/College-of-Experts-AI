import onnxruntime_genai as og
import time
import os

def test_oga_speed():
    model_path = "models/Qwen2.5-Coder-7B-Instruct-OGA"
    print(f"Loading OGA model from {model_path}...")
    
    try:
        model = og.Model(model_path)
        tokenizer = og.Tokenizer(model)
        tokenizer_stream = tokenizer.create_stream()
        print("Model Loaded.")
    except Exception as e:
        print(f"Error loading: {e}")
        return

    prompt = "def quicksort(arr):"
    input_tokens = tokenizer.encode(prompt)
    
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=150)
    
    generator = og.Generator(model, params)
    generator.append_tokens(input_tokens)
    
    print("\nStarting generation...")
    start_gen = time.time()
    generated_tokens = 0
    
    while not generator.is_done():
        generator.generate_next_token()
        generated_tokens += 1
        print(tokenizer_stream.decode(generator.get_next_tokens()[0]), end="", flush=True)
        
    elapsed = time.time() - start_gen
    
    print(f"\n\nThroughput: {generated_tokens/elapsed:.2f} tokens/s")

if __name__ == "__main__":
    test_oga_speed()
