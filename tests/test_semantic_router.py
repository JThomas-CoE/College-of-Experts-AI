
import time
import sys
import torch

# PATCH: Fix for custom AMD PyTorch builds missing distributed.is_initialized
if not hasattr(torch, 'distributed'):
    class MockDistributed:
        def is_initialized(self): return False
        def get_rank(self): return 0
        def get_world_size(self): return 1
    torch.distributed = MockDistributed()
elif not hasattr(torch.distributed, 'is_initialized'):
    torch.distributed.is_initialized = lambda: False

from sentence_transformers import SentenceTransformer, util

def test_semantic_routing():
    print("Initializing Embedding Model (BAAI/bge-m3)...")
    # We use 'BAAI/bge-m3' for high quality, or 'all-MiniLM-L6-v2' for speed.
    # Let's try to pull the good one.
    try:
        model = SentenceTransformer('BAAI/bge-m3')
    except Exception as e:
        print(f"Failed to load BAAI/bge-m3: {e}")
        print("Falling back to 'all-MiniLM-L6-v2'...")
        model = SentenceTransformer('all-MiniLM-L6-v2')

    print(f"Model loaded. Device: {model.device}")

    # 1. Define Experts (The "Catalog" Index)
    experts = {
        "python_expert": "Software Engineering, Python code, API development, Scripting, Debugging, Classes, Functions.",
        "legal_expert": "Legal compliance, Regulations, HIPAA, GDPR, Lawsuits, Liability, Privacy Policy, Disclaimers.",
        "medical_expert": "Clinical diagnosis, Anatomy, Physiology, Pharmacology, Diseases, Treatment protocols, Patient care.",
        "security_expert": "Cybersecurity, Authentication, Encryption, Auditing, Vulnerability scanning, Hashing, OAuth."
    }
    
    expert_names = list(experts.keys())
    expert_descriptions = list(experts.values())
    
    print("\nEncoding Expert Profiles...")
    expert_embeddings = model.encode(expert_descriptions, convert_to_tensor=True)
    
    # 2. Simulate User Tasks (The "Decomposition" chunks)
    tasks = [
        "Write a Python script to mask patient IDs",
        "Define the HIPAA requirements for data storage",
        "Diagnose the symptoms of Sepsis",
        "Ensure the database password is hashed using bcrypt"
    ]
    
    print("\n--- Routing Test ---")
    for task in tasks:
        print(f"\nTask: '{task}'")
        task_emb = model.encode(task, convert_to_tensor=True)
        
        # Compute Cosine Similarities
        scores = util.cos_sim(task_emb, expert_embeddings)[0]
        
        # Find Winner
        best_score_idx = scores.argmax()
        best_expert = expert_names[best_score_idx]
        best_score = scores[best_score_idx]
        
        print(f"  -> Assigned to: {best_expert} (Score: {best_score:.4f})")
        
        # Show all scores for debugging
        for i, name in enumerate(expert_names):
            print(f"     - {name}: {scores[i]:.4f}")
    
    print("\n[Test Complete] Forcing exit to clear background threads...")
    import os
    os._exit(0)

if __name__ == "__main__":
    test_semantic_routing()
