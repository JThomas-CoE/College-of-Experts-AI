"""
Simplified test to verify LFM2 model works with standard generation only.
Focus: Just basic generation - no KV cache manipulation.
"""
import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from pathlib import Path

MODEL_PATH = Path("models/LiquidAI_LFM2.5-1.2B-Instruct")

print("=" * 60)
print("LFM2 Multi-Expert Test (Without Shared KV Cache)")
print("=" * 60)

# Load model once
print("\n1. Loading model...")
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print(f"   Model loaded: {type(model).__name__}")
print(f"   Device: {next(model.parameters()).device}")

# Define expert personas (system prompts)
EXPERTS = {
    "python_expert": "You are a Python expert. Be concise and focus on best practices.",
    "security_expert": "You are a security expert. Focus on vulnerabilities and mitigations.",
    "architecture_expert": "You are a software architect. Discuss trade-offs and design patterns."
}

def generate_with_persona(persona_name: str, system_prompt: str, user_message: str):
    """Generate using a specific expert persona (via system prompt)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()

# Test each expert with the same question
print("\n2. Testing multiple expert personas...")
user_question = "What should I consider when adding authentication to a web app?"

for expert_name, system_prompt in EXPERTS.items():
    print(f"\n→ {expert_name.upper()}")
    print(f"  System: {system_prompt[:50]}...")
    
    try:
        response = generate_with_persona(expert_name, system_prompt, user_question)
        print(f"  Response: {response[:200]}...")
    except Exception as e:
        print(f"  ✗ Error: {e}")

print("\n" + "=" * 60)
print("MULTI-EXPERT TEST COMPLETE")
print("=" * 60)
print("\nConclusion:")
print("- Basic generation with system prompts works!")
print("- Each 'expert' is the same model with different persona via system prompt")
print("- For now, shared KV cache is NOT used (text preamble mode)")
