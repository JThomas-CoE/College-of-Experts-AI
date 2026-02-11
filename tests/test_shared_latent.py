
import os
import torch
import time
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from src.shared_latent import SharedLatentSpace, SharedLatentConfig

# Disable parallelism warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    print("=== Shared Latent Space Diagnostic ===")
    
    # 1. Load Model
    model_path = "models/Qwen3-VL-4B-Instruct"
    print(f"Loading model from {model_path}...")
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, 
            trust_remote_code=True, 
            torch_dtype=torch.bfloat16, 
            device_map="auto"
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"FAILED to load model: {e}")
        return

    # 2. Define Shared Context
    SHARED_PROMPT = "You are a helpful AI assistant focused on Python security."
    system_messages = [{"role": "system", "content": SHARED_PROMPT}]
    # Get raw text for the system prompt
    system_text = processor.apply_chat_template(system_messages, tokenize=False)
    print(f"\nShared System Text ({len(system_text)} chars):\n{repr(system_text)}")

    # 3. Initialize Shared Space
    shared_space = SharedLatentSpace(model, processor.tokenizer, SharedLatentConfig(enabled=True))
    
    print("Computing KV Cache...")
    start = time.time()
    shared_space.set_shared_context(system_text)
    print(f"Cache computation took {time.time() - start:.2f}s")
    
    # Verify Cache
    pkv = shared_space.get_cloned_past_key_values()
    if pkv is None:
        print("CRITICAL: Cache is None!")
        return
    
    cache_len = pkv[0][0].shape[2]
    print(f"✓ Cache Valid. Cached Tokens: {cache_len}")

    # 4. Simulate Expert Generation
    print("\n--- Simulating Expert Generation ---")
    
    user_query = "How do I secure a Flask app?"
    # Construct FULL prompt (System + User)
    full_messages = [
        {"role": "system", "content": SHARED_PROMPT},
        {"role": "user", "content": user_query}
    ]
    full_text = processor.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(text=[full_text], return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    full_len = input_ids.shape[1]
    
    print(f"Full Input Length: {full_len} tokens")
    
    # Verify Prefix Match
    # We need to ensure the full input starts EXACTLY with the tokenized shared text
    # Note: Tokenizers are tricky. "A" + "B" tokens != "AB" tokens sometimes.
    # We'll check if we can reuse the cache.
    
    if full_len > cache_len:
        print(f"Input is longer than cache ({full_len} > {cache_len}). Attempting generation...")
        
        # SLICE INPUT
        # Pass only the NEW tokens (suffix)
        input_suffix = input_ids[:, cache_len:]
        print(f"Suffix Length: {input_suffix.shape[1]} tokens")
        
        print("Generating...")
        with torch.no_grad():
            output = model.generate(
                input_ids=input_suffix,
                past_key_values=pkv, # Pass the shared cache
                max_new_tokens=50
            )
        
        generated_ids = output[0] # output includes input_suffix + new tokens usually?
        # Qwen generate with past_key_values returns only new tokens? Or suffix + new?
        # Let's decode to find out.
        
        response = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
        print(f"\nGenerated Response:\n{response}")
        print("\nSUCCESS: Shared Latent Space was used!")
    else:
        print(f"ERROR: Input shorter than cache? {full_len} vs {cache_len}")

if __name__ == "__main__":
    main()
