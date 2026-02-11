"""
Simple test to verify LiquidAI LFM2 model works with basic generation.
Step 1: Test basic generation
Step 2: Test with use_cache=True
Step 3: Test with past_key_values for shared cache
"""
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from pathlib import Path

print("=" * 60)
print("LFM2 Model Basic Test")
print("=" * 60)

MODEL_PATH = Path("models/LiquidAI_LFM2.5-1.2B-Instruct")

# Step 1: Load model and tokenizer
print("\n1. Loading model and tokenizer...")
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
print(f"   Model type: {type(model).__name__}")
print(f"   Model config: {model.config.model_type}")
print(f"   Device: {next(model.parameters()).device}")

# Set pad token if needed
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Step 2: Test basic generation (without cache)
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

# Step 3: Test forward pass with use_cache to get past_key_values
print("\n3. Testing forward pass with use_cache=True...")
try:
    inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(
            **inputs,
            use_cache=True,
            return_dict=True
        )
    
    past_kv = outputs.past_key_values
    print(f"   past_key_values type: {type(past_kv)}")
    if past_kv is not None:
        if hasattr(past_kv, '__len__'):
            print(f"   Number of layers: {len(past_kv)}")
            if len(past_kv) > 0 and past_kv[0] is not None:
                if isinstance(past_kv[0], tuple):
                    print(f"   First layer K shape: {past_kv[0][0].shape if past_kv[0][0] is not None else 'None'}")
                else:
                    print(f"   First layer type: {type(past_kv[0])}")
        print("   ✓ past_key_values returned!")
    else:
        print("   ✗ past_key_values is None")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Step 4: Test generation WITH past_key_values (continuation)
print("\n4. Testing generation with past_key_values (continuation)...")
try:
    # First, encode a "shared context"
    shared_context = "You are a Python expert helping with code."
    context_inputs = tokenizer(shared_context, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        context_outputs = model(
            **context_inputs,
            use_cache=True,
            return_dict=True
        )
    
    cached_kv = context_outputs.past_key_values
    print(f"   Cached context: '{shared_context}' ({context_inputs['input_ids'].shape[1]} tokens)")
    
    # Now generate new tokens using the cached context
    new_text = " What is a decorator?"
    new_inputs = tokenizer(new_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=new_inputs['input_ids'],
            attention_mask=new_inputs['attention_mask'],
            past_key_values=cached_kv,
            max_new_tokens=50,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   Response with cached context: {response[:200]}...")
    print("   ✓ Generation with past_key_values works!")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
