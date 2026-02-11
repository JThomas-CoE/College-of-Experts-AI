
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_path = r"c:\RyzenAI\college of experts\models\LiquidAI_LFM2.5-1.2B-Instruct"

def test_lfm():
    print(f"Testing LFM model at {model_path}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda:0"}
        )
        print("Model and Tokenizer loaded successfully.")
        
        # Test chat template compatibility
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, explain what a Liquid Foundation Model is."}
        ]
        
        # Note: LFM might not support the {"type": "text", "content": ...} structure.
        # We test standard role/content strings first.
        try:
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            print("Chat template applied successfully.")
            print(f"Template preview: {text[:100]}...")
        except Exception as e:
            print(f"Chat template error: {e}")
            return

        inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)
            
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"Response: {response}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_lfm()
