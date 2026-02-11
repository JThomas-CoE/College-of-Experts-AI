import onnxruntime_genai as og
import time
import os

def test_bio_oga():
    model_path = "models/BioMistral-7B-DML"

    print(f"Loading Medical Specialist OGA model from {model_path}...")
    
    try:
        model = og.Model(model_path)
        tokenizer = og.Tokenizer(model)
        tokenizer_stream = tokenizer.create_stream()
        print("Model Loaded successfully.")
    except Exception as e:
        print(f"Error loading: {e}")
        return

    prompt = "<s>[INST] What are the primary symptoms of diabetes? [/INST]"
    print(f"\nPrompt: {prompt}")
    
    input_tokens = tokenizer.encode(prompt)
    
    params = og.GeneratorParams(model)
    params.set_search_options(max_length=200, temperature=0.3)
    
    generator = og.Generator(model, params)
    generator.append_tokens(input_tokens)
    
    print("\nStarting generation...")
    start_gen = time.time()
    generated_tokens = 0
    
    try:
        while not generator.is_done():
            generator.generate_next_token()
            generated_tokens += 1
            print(tokenizer_stream.decode(generator.get_next_tokens()[0]), end="", flush=True)
            
        elapsed = time.time() - start_gen
        if elapsed > 0:
            print(f"\n\nThroughput: {generated_tokens/elapsed:.2f} tokens/s")
        else:
            print(f"\n\nThroughput: 0 tokens/s (Elapsed time was 0)")
    except Exception as e:
        import traceback
        print(f"\nError during generation: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_bio_oga()
