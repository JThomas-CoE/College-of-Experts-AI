import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.memory_backbone import MemoryBackbone, MemoryConfig
from src.expert_catalog import load_catalog
from src.expert_slots import ExpertSlotManager
from src.backends.oga_backend import OGABackend

def test_memory_aware_generation():
    print("Testing Memory-Aware Generation...")
    
    # 1. Setup Memory
    memory = MemoryBackbone(MemoryConfig(db_path=Path("data/test_shared_memory.db")))
    memory.clear_tier("working")
    memory.clear_tier("semantic")
    
    # Set a user preference
    memory.set_user_preference("coding_style", "Strictly use functional programming and type hints.")
    
    # Set a working context
    memory.write("task.context", {
        "project": "Savant AI",
        "current_module": "Memory Controller",
        "existing_code": "def process(data): return data"
    }, tier="working")
    
    # 2. Setup Expert
    catalog = load_catalog()
    python_expert_def = catalog.get_expert("python_expert")
    
    # Load model (mocking some parts for speed test if needed, but let's try real load if possible)
    # Actually, let's just test the prompt construction for now to avoid 7GB load in a unit test
    # unless we really want to see it work.
    
    # 3. Construct Context-Aware Prompt
    user_query = "Refactor the existing code to be more robust."
    
    # Fetch from memory
    style_pref = memory.get_user_preference("coding_style")
    context = memory.read("task.context", tier="working").value
    
    augmented_prompt = f"""
USER PREFERENCES:
{style_pref}

PROJECT CONTEXT:
Project: {context['project']}
Module: {context['current_module']}
Existing Code: {context['existing_code']}

TASK:
{user_query}
"""
    print("\n--- Augmented Prompt ---")
    print(augmented_prompt)
    print("------------------------")
    
    if "functional programming" in augmented_prompt and "Savant AI" in augmented_prompt:
        print("\nSUCCESS: Memory successfully injected into prompt context.")
    else:
        print("\nFAILURE: Memory injection failed.")

if __name__ == "__main__":
    test_memory_aware_generation()
