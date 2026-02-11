"""
Test Qwen2.5-1.5B-Instruct for multi-expert with shared KV cache.

This is a pure text model with standard transformer architecture.
Should work perfectly with KV cache sharing.
"""
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from pathlib import Path

print("=" * 60)
print("Qwen2.5-1.5B Multi-Expert Test")
print("=" * 60)

# Use a small, reliable text-only model
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

# Step 1: Load model
print(f"\n1. Loading {MODEL_NAME}...")
print("   (This will download ~3GB on first run)")

from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"   Model type: {type(model).__name__}")
    print(f"   Device: {next(model.parameters()).device}")
    print("   ✓ Model loaded!")
except Exception as e:
    print(f"   ✗ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 2: Test basic generation
print("\n2. Testing basic generation...")
messages = [{"role": "user", "content": "Say hello in one sentence."}]

try:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"   Response: {response}")
    print("   ✓ Basic generation works!")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Step 3: Test KV cache extraction
print("\n3. Testing KV cache extraction...")
try:
    inputs = tokenizer("You are a Python expert.", return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(
            **inputs,
            use_cache=True,
            return_dict=True
        )
    
    past_kv = outputs.past_key_values
    print(f"   past_key_values type: {type(past_kv)}")
    if past_kv is not None:
        print(f"   Number of layers: {len(past_kv)}")
        if len(past_kv) > 0 and past_kv[0] is not None:
            print(f"   First layer K shape: {past_kv[0][0].shape}")
        print("   ✓ KV cache works!")
    else:
        print("   ✗ past_key_values is None")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Step 4: Test generation WITH past_key_values  
print("\n4. Testing generation with pre-computed KV cache...")
try:
    # Compute shared context KV cache
    shared_context = "System: You are a Python expert helping with authentication."
    context_inputs = tokenizer(shared_context, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        context_outputs = model(
            **context_inputs,
            use_cache=True,
            return_dict=True
        )
    
    cached_kv = context_outputs.past_key_values
    context_len = context_inputs['input_ids'].shape[1]
    print(f"   Cached context: '{shared_context}' ({context_len} tokens)")
    
    # Generate continuation using cached KV
    continuation = "\nUser: What is JWT?\nAssistant:"
    new_inputs = tokenizer(continuation, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=new_inputs['input_ids'],
            attention_mask=new_inputs['attention_mask'],
            past_key_values=cached_kv,
            max_new_tokens=100,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   Response: {response[:300]}...")
    print("   ✓ KV cache continuation works!")
except Exception as e:
    print(f"   ✗ Error with KV continuation: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
