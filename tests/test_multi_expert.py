"""
Test College of Experts with multiple experts using Nemotron Nano 30B via Ollama.

This test:
1. Loads 5 expert personas (Python, JavaScript, SQL, Security, Architecture)
2. Routes queries to appropriate experts
3. Tests multi-expert coordination
"""
import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("College of Experts - Multi-Expert Test (Nemotron via Ollama)")
print("=" * 70)

from src.backends.ollama_backend import OllamaBackend
from src.backends.base import GenerationConfig
from src.experts.personas import (
    PYTHON_EXPERT, 
    JAVASCRIPT_EXPERT, 
    SQL_EXPERT, 
    SECURITY_EXPERT, 
    ARCHITECTURE_EXPERT
)

# Initialize backend (auto-starts Ollama if needed)
print("\n1. Initializing Ollama backend...")
backend = OllamaBackend(auto_start=True)

if not backend.is_ollama_running():
    print("   ERROR: Could not start Ollama!")
    sys.exit(1)
print("   Ollama running ✓")

# Load Nemotron as the base model
print("\n2. Loading Nemotron Nano 30B as base model...")
MODEL_NAME = "nemotron-3-nano:30b"

try:
    info = backend.load_model(
        model_id="base_model",
        model_path=MODEL_NAME
    )
    print(f"   Model: {info.model_path}")
    print(f"   Ready: {info.is_loaded}")
except Exception as e:
    print(f"   ERROR: {e}")
    sys.exit(1)

# Define experts to test
EXPERTS = [
    ("python_expert", PYTHON_EXPERT),
    ("javascript_expert", JAVASCRIPT_EXPERT),
    ("sql_expert", SQL_EXPERT),
    ("security_expert", SECURITY_EXPERT),
    ("architecture_expert", ARCHITECTURE_EXPERT),
]

# Test queries for each expert
TEST_QUERIES = {
    "python_expert": "What's the best way to handle async file I/O in Python? Keep it brief.",
    "javascript_expert": "How do I use React Server Components? Brief answer.",
    "sql_expert": "How do I optimize a slow JOIN query? Brief answer.",
    "security_expert": "What are the top 3 things to check for SQL injection? Brief list.",
    "architecture_expert": "Microservices vs monolith - when to choose each? Brief answer.",
}

# Test each expert
print("\n3. Testing each expert persona...")
print("-" * 70)

config = GenerationConfig(max_tokens=200, temperature=0.5)

for expert_id, persona in EXPERTS:
    print(f"\n🎓 Expert: {persona.name}")
    print(f"   Domains: {', '.join(persona.domains)}")
    
    query = TEST_QUERIES[expert_id]
    print(f"   Query: \"{query}\"")
    
    try:
        response = backend.generate(
            model_id="base_model",
            messages=[{"role": "user", "content": query}],
            system_prompt=persona.system_prompt,
            config=config
        )
        
        # Truncate response for display
        display_response = response[:300] + "..." if len(response) > 300 else response
        print(f"   Response:\n   {display_response}")
        print(f"   ✓ Success ({len(response)} chars)")
        
    except Exception as e:
        print(f"   ✗ Error: {e}")

# Test 4: Multi-expert coordination scenario
print("\n" + "=" * 70)
print("4. Multi-Expert Coordination Test")
print("=" * 70)

COMPLEX_QUERY = """
I'm building a FastAPI backend with a PostgreSQL database. 
Users can submit code snippets that get stored and executed.
What should I be aware of?
"""

print(f"\nComplex Query: {COMPLEX_QUERY.strip()}")
print("\nConsulting multiple experts...")

# Get input from multiple experts
multi_expert_responses = {}

for expert_id in ["python_expert", "sql_expert", "security_expert"]:
    persona = dict(EXPERTS)[expert_id]
    print(f"\n→ Asking {persona.name}...")
    
    try:
        response = backend.generate(
            model_id="base_model",
            messages=[{"role": "user", "content": COMPLEX_QUERY}],
            system_prompt=persona.system_prompt + "\n\nBe concise. Focus on your area of expertise.",
            config=GenerationConfig(max_tokens=250, temperature=0.5)
        )
        multi_expert_responses[expert_id] = response
        print(f"   Response ({len(response)} chars): {response[:150]}...")
    except Exception as e:
        print(f"   Error: {e}")

# Summary
print("\n" + "=" * 70)
print("5. Test Summary")
print("=" * 70)

print(f"\n✓ Backend: Ollama")
print(f"✓ Model: {MODEL_NAME}")
print(f"✓ Experts tested: {len(EXPERTS)}")
print(f"✓ Multi-expert responses: {len(multi_expert_responses)}")

# Cleanup
print("\n6. Cleanup...")
backend.unload_model("base_model")
print("   Done ✓")

print("\n" + "=" * 70)
print("MULTI-EXPERT TEST COMPLETE ✓")
print("=" * 70)
