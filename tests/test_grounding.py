"""
Test Grounding - Verifies that the Harness correctly retrieves and injects context.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.harness import Harness, HarnessConfig
from src.router import RouterConfig

def test_grounded_legal_query():
    print("Testing Grounded Legal Query...")
    harness = Harness()
    session = harness.create_session()
    
    # We'll use a query that should trigger the 'hearsay' grounding
    user_input = "What exactly is hearsay and what are the main exceptions?"
    
    # We'll mock the expert response part or just look at the logs if possible
    # For this test, we want to see if the grounding logic is triggered.
    # I'll just call the internal _get_grounded_context for a quick check.
    
    context = harness._get_grounded_context(user_input, "legal")
    print(f"\nRetrieved Context:\n{context}")
    
    if "out-of-court statement" in context:
        print("\nSUCCESS: Correct grounded context retrieved for Legal.")
    else:
        print("\nFAILURE: Grounding retrieval failed for Legal.")

def test_grounded_python_query():
    print("\nTesting Grounded Python Query...")
    harness = Harness()
    
    # Query about OGA
    user_input = "How do I run inference with onnxruntime-genai in Python?"
    context = harness._get_grounded_context(user_input, "python")
    print(f"\nRetrieved Context:\n{context}")
    
    if "onnxruntime-genai" in context or "OGA" in context:
        print("\nSUCCESS: Correct grounded context retrieved for Python.")
    else:
        print("\nFAILURE: Grounding retrieval failed for Python.")

if __name__ == "__main__":
    test_grounded_legal_query()
    test_grounded_python_query()
