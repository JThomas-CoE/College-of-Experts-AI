"""
Test Grounding + Model - Verifies that RAG and OGA can coexist in the same process.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import onnxruntime_genai as og
from src.vector_backbone import VectorBackbone

def test_coexistence():
    print("Testing Coexistence of RAG and OGA...")
    
    # 1. Load OGA Model first
    print("Loading OGA Model...")
    try:
        model = og.Model("models/law-LLM-DML")
        print("SUCCESS: OGA Model loaded with DML.")
    except Exception as e:
        print(f"FAILURE: OGA Model load failed: {e}")
        return

    # 2. Initialize Vector Store
    print("\nInitializing Vector Store...")
    try:
        vb = VectorBackbone("data/vector_db")
        results = vb.query("legal", "hearsay")
        print(f"SUCCESS: Vector retrieval works: {len(results['documents'][0])} results.")
    except Exception as e:
        print(f"FAILURE: Vector Store error: {e}")
        return

    print("\nEverything is working together!")

if __name__ == "__main__":
    test_coexistence()
