"""Integration test - FLM Backend with SlotManager."""
import sys
sys.path.insert(0, ".")

from src.expert_slots import ExpertSlotManager
from src.expert_catalog import load_catalog

print("=" * 50)
print("FLM + SlotManager Integration Test")
print("=" * 50)

# Load catalog
catalog = load_catalog("config/profile_lfm.json")
print(f"Catalog: {len(catalog)} experts, Model: {catalog.base_model}")

# Create slot manager with NPU enabled
slot_manager = ExpertSlotManager(
    catalog=catalog,
    num_slots=4,  # 3 NPU + 1 GPU (minimal test)
    max_vram_mb=64000,
    preload=False,
    multi_instance=True,
    use_npu_for_router=True,
    npu_slots=[0, 1, 2],  # NPU slots
    onnx_model_path=None  # Not needed for FLM
)

# Load an expert on NPU slot
print("\n[TEST] Loading expert on NPU slot 0...")
instance = slot_manager.load_expert("general_reasoner", slot_id=0)

print(f"  Model type: {type(instance.model).__name__}")
print(f"  Is API backend: {getattr(instance.model, 'is_api_backend', False)}")

# Test generation
print("\n[TEST] Generating response...")
response = instance.model.generate(
    prompt="What is 2 + 2? Answer briefly.",
    max_tokens=50,
    temperature=0.5
)
print(f"Response: {response}")

print("\n" + "=" * 50)
print("[DONE] Integration test complete")
