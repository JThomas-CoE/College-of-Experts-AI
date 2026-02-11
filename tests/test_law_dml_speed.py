import onnxruntime_genai as og
import time
import os

def test_law_dml_speed():
    model_path = "models/law-LLM-DML"
    print(f"Loading Law DML OGA model from {model_path}...")
    
    try:
        model = og.Model(model_path)
        tokenizer = og.Tokenizer(model)
        tokenizer_stream = tokenizer.create_stream()
        print("Model Loaded.")
    except Exception as e:
        print(f"Error loading: {e}")
        return

    prompt = "What are the common elements of a breach of contract claim?"
    input_tokens = tokenizer.encode(prompt)
    
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=200, temperature=0.7)
    
    generator = og.Generator(model, params)
    generator.append_tokens(input_tokens)
    
    print("\nStarting generation...")
    start_gen = time.time()
    generated_tokens = 0
    
    while not generator.is_done():
        generator.generate_next_token()
        generated_tokens += 1
        token = generator.get_next_tokens()[0]
        print(tokenizer_stream.decode(token), end="", flush=True)
        
    elapsed = time.time() - start_gen
    
    print(f"\n\nThroughput: {generated_tokens/elapsed:.2f} tokens/s")

if __name__ == "__main__":
    test_law_dml_speed()
