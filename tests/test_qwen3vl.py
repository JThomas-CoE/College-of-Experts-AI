"""
Test Qwen3-VL-4B-Instruct for multi-expert with shared KV cache.

Uses the correct VL model loading with Qwen3VLForConditionalGeneration.
Can do both text and vision tasks.
"""
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch

print("=" * 60)
print("Qwen3-VL-4B Multi-Expert Test")
print("=" * 60)

MODEL_NAME = "models/Qwen3-VL-4B-Instruct"

# Step 1: Load model with correct VL classes
print(f"\n1. Loading {MODEL_NAME}...")
print("   (This will download ~8GB on first run)")

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

try:
    # Use AutoProcessor for VL models (handles both text and images)
    processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print(f"   Model type: {type(model).__name__}")
    print(f"   Device: {next(model.parameters()).device}")
    print("   ✓ Model loaded!")
except Exception as e:
    print(f"   ✗ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Step 2: Test text-only generation (no images)
print("\n2. Testing text-only generation...")
messages = [
    {"role": "user", "content": [{"type": "text", "text": "Say hello in one sentence."}]}
]

try:
    # Process without images
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            use_cache=True
        )
    
    response = processor.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
    print(f"   Response: {response}")
    print("   ✓ Text generation works!")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Step 3: Test KV cache extraction
print("\n3. Testing KV cache extraction...")
try:
    text = "You are a Python expert."
    inputs = processor(text=[text], return_tensors="pt").to(model.device)
    
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
            print(f"   First layer shape: {past_kv[0][0].shape if isinstance(past_kv[0], tuple) else type(past_kv[0])}")
        print("   ✓ KV cache works!")
    else:
        print("   ✗ past_key_values is None")
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Step 4: Test multiple expert personas with text
print("\n4. Testing multiple expert personas...")

EXPERTS = {
    "python_expert": "You are a Python expert. Be concise.",
    "security_expert": "You are a security expert. Focus on vulnerabilities.",
    "architecture_expert": "You are a software architect. Discuss trade-offs."
}

user_question = "What is JWT authentication?"

for expert_name, system_prompt in EXPERTS.items():
    print(f"\n→ {expert_name.upper()}")
    
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [{"type": "text", "text": user_question}]}
    ]
    
    try:
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7
            )
        
        response = processor.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
        print(f"   Response: {response[:200]}...")
    except Exception as e:
        print(f"   ✗ Error: {e}")

print("\n" + "=" * 60)
print("QWEN3-VL TEST COMPLETE")
print("=" * 60)
